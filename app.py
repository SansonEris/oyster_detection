from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import os
import glob
import json
from detect_api import OysterDetector

app = Flask(__name__)
CORS(app)

# Global detector instance
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

@app.route('/api/last_frame')
def last_frame():
    frames = sorted([f for f in os.listdir('outputs') if f.endswith('.jpg')])
    if not frames:
        return ('', 204)
    return send_from_directory('outputs', frames[-1])

# >>> AGGIUNTA: endpoint per cancellare tutte le .jpg in outputs/ <<<
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
# <<< FINE AGGIUNTA >>>

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
