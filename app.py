from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import os
import glob
import json
from detect_api import OysterDetector
import subprocess
import signal

app = Flask(__name__)
CORS(app)

gps_proc = None
detector = None
detection_thread = None

@app.route('/api/files')
def get_files():
    """Get available models and videos"""
    models = glob.glob('models/*.pt') + glob.glob('models/*.onnx')
    videos = glob.glob('videos/*.mp4') + glob.glob('videos/*.avi') + glob.glob('videos/*.mov')
    
    return jsonify({
        'models': [os.path.basename(m) for m in models],
        'videos': [os.path.basename(v) for v in videos],
        'defaults': {
            'conf': 0.25,
            'iou': 0.45,
            'output': 'outputs/'
        }
    })

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection with given parameters"""
    global detector, detection_thread
    
    if detector and detector.is_running():
        return jsonify({'ok': False, 'error': 'Detection already running'})
    
    config = request.json
    
    try:
        # Validate required fields
        if not config.get('model') or not config.get('left'):
            return jsonify({'ok': False, 'error': 'Missing required fields'})
        
        if config['mode'] == 'stereo' and not config.get('right'):
            return jsonify({'ok': False, 'error': 'Right video required for stereo mode'})
        
        # Create detector instance
        detector = OysterDetector(
            model_path=os.path.join('models', config['model']),
            mode=config['mode'],
            left_video=os.path.join('videos', config['left']) if config['left'] else None,
            right_video=os.path.join('videos', config['right']) if config.get('right') else None,
            confidence=config.get('conf', 0.25),
            iou=config.get('iou', 0.45),
            output_dir=config.get('output', 'outputs/')
        )
        
        # Start detection in separate thread
        detection_thread = threading.Thread(target=detector.run)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({'ok': True})
        
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop current detection"""
    global detector
    
    if detector:
        detector.stop()
    
    return jsonify({'ok': True})

@app.route('/api/status')
def get_status():
    """Get current detection status"""
    global detector
    
    if not detector:
        return jsonify({
            'running': False,
            'frames': 0,
            'total_frames': 0,
            'fps': 0.0,
            'oyster_count': 0
        })
    
    return jsonify(detector.get_status())

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    """Serve output files"""
    return send_from_directory('outputs', filename)

@app.route('/tools')
def tools_page():
    return send_from_directory('.', 'data_tools.html')

@app.route('/api/last_frame')
def last_frame():
    base = 'outputs'
    # Find latest run directory by mtime (directories starting with 'run_')
    runs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d.startswith('run_')]
    if runs:
        latest_run = max(runs, key=os.path.getmtime)
        frames = sorted([f for f in os.listdir(latest_run) if f.endswith('.jpg')])
        if not frames:
            return ('', 204)
        return send_from_directory(latest_run, frames[-1])
    else:
        # backward-compat: search directly in outputs/
        frames = sorted([f for f in os.listdir(base) if f.endswith('.jpg')])
        if not frames:
            return ('', 204)
        return send_from_directory(base, frames[-1])

@app.route('/api/clear_frames', methods=['POST'])
def clear_frames():
    out_dir = 'outputs'
    removed = 0
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.lower().endswith('.jpg'):
            try:
                os.remove(os.path.join(out_dir, name))
                removed += 1
            except Exception:
                pass
    return jsonify({'ok': True, 'removed': removed})

# MAPS ROUTE E CONFIGURAZIONE
@app.route('/api/tools/files')
def tools_files():
    """Elenco utile per la UI: cartelle immagini e CSV disponibili."""
    try:
        images_dirs = [d for d in glob.glob('outputs/*') if os.path.isdir(d)]
        csvs = glob.glob('*.csv') + glob.glob('outputs/**/*.csv', recursive=True) + glob.glob('maps/*.csv')
        return jsonify({
            'images_dirs': images_dirs,
            'csvs': csvs,
            'defaults': {
                'mavlink_base': 'http://192.168.2.2:6040',
                'gps_hz': 5,
                'tolerance_ms': 500,
                'offset_ms': 0,
                'method': 'interp'
            }
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/tools/gps/start', methods=['POST'])
def gps_start():
    """
    Avvia il logger GPS (maps/get_gps.py) come sottoprocesso.
    Body JSON (opzionali): { "base": "...", "out": "gps.csv", "hz": 5 }
    """
    global gps_proc
    if gps_proc and gps_proc.poll() is None:
        return jsonify({'ok': False, 'error': 'GPS logger already running'})

    data = request.json or {}
    base = data.get('base', 'http://192.168.2.2:6040')
    out  = data.get('out',  'gps.csv')
    hz   = str(data.get('hz', 5))

    # Adegua gli argomenti a quelli previsti da maps/get_gps.py (se diversi, cambiali qui)
    cmd = ['python', 'maps/get_gps.py', '--base', base, '--out', out, '--hz', hz]

    try:
        gps_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return jsonify({'ok': True, 'pid': gps_proc.pid, 'cmd': ' '.join(cmd)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/tools/gps/stop', methods=['POST'])
def gps_stop():
    """Ferma il logger GPS se attivo."""
    global gps_proc
    if gps_proc and gps_proc.poll() is None:
        try:
            gps_proc.terminate()
            try:
                gps_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                gps_proc.kill()
        finally:
            gps_proc = None
        return jsonify({'ok': True})
    return jsonify({'ok': False, 'error': 'GPS logger not running'})


@app.route('/api/tools/frames/generate', methods=['POST'])
def frames_generate():
    """
    Genera frames.csv dalla cartella immagini con maps/frames_from_images.py
    Body JSON: { "images_dir": "...", "out": "frames.csv" }
    """
    data = request.json or {}
    images_dir = data.get('images_dir')
    out = data.get('out', 'frames.csv')

    if not images_dir:
        return jsonify({'ok': False, 'error': 'images_dir required'}), 400

    cmd = ['python', 'maps/frames_from_images.py', '--images-dir', images_dir, '--out', out]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return jsonify({
            'ok': res.returncode == 0,
            'stdout': res.stdout,
            'stderr': res.stderr,
            'cmd': ' '.join(cmd),
            'out': out
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/tools/combine/run', methods=['POST'])
def combine_run():
    """
    Esegue il merge/allineamento con maps/combine_csv.py
    Body JSON: {
      "frames": "frames.csv",
      "gps": "gps.csv",
      "detections": "outputs/run_.../detections.csv" (opzionale),
      "out": "final.csv",
      "method": "nearest|interp" (opzionale),
      "tolerance_ms": 500 (opzionale),
      "offset_ms": 0 (opzionale)
    }
    """
    data = request.json or {}
    frames = data.get('frames')
    gps = data.get('gps')
    detections = data.get('detections')
    out = data.get('out', 'final.csv')
    method = data.get('method', 'interp')
    tolerance_ms = str(data.get('tolerance_ms', 500))
    offset_ms = str(data.get('offset_ms', 0))

    if not frames or not gps:
        return jsonify({'ok': False, 'error': 'frames and gps are required'}), 400

    # Adegua gli argomenti a come è implementato maps/combine_csv.py
    cmd = ['python', 'maps/combine_csv.py',
           '--frames', frames,
           '--gps', gps,
           '--out', out,
           '--method', method,
           '--tolerance-ms', tolerance_ms,
           '--offset-ms', offset_ms]
    if detections:
        cmd += ['--detections', detections]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return jsonify({
            'ok': res.returncode == 0,
            'stdout': res.stdout,
            'stderr': res.stderr,
            'cmd': ' '.join(cmd),
            'out': out
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
