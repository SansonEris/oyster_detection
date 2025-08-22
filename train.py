# === SETUP AMBIENTE COLAB PER YOLOV5 OYSTER TRAINING ===
# Esegui questo setup PRIMA del tuo codice principale

# STEP 1: Installa Conda su Colab
print("🔧 Installing Conda on Colab...")
!pip install -q condacolab
import condacolab
condacolab.install()

# STEP 2: Downgrade a Python 3.8 (compatibile con le tue dipendenze)
print("🐍 Setting up Python 3.8...")
!mamba install -y "python=3.8"

# STEP 3: Aggiorna tooling base
print("🛠️ Updating base tools...")
!python -m pip install -U "pip>=24.0" "setuptools>=68.0.0" "wheel>=0.41.0"

# STEP 4: Installa PyTorch compatibile
print("🔥 Installing PyTorch...")
# Per GPU (default in Colab)
!mamba install -y cudatoolkit=11.6
!python -m pip install "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"

# Se vuoi forzare CPU only, usa invece:
# !python -m pip install "torch==1.13.1+cpu" "torchvision==0.14.1+cpu" "torchaudio==0.13.1" -f https://download.pytorch.org/whl/cpu/torch_stable.html

# STEP 5: Dipendenze pin (identiche al tuo Docker)
print("📦 Installing pinned dependencies...")
!python -m pip install gdown PyYAML "pandas==2.2.2" "numpy==1.24.4" "pydantic<2.0" "thop==0.1.1.post2209072238" "albumentations==1.3.1"

# STEP 6: Stack ONNX compatibile con Python 3.8
print("🤖 Installing ONNX stack...")
!python -m pip install "onnx==1.15.0" "onnxruntime==1.17.3" "onnxsim==0.4.36"

# STEP 7: Librerie per visualizzazioni
print("📊 Installing visualization libraries...")
!python -m pip install matplotlib seaborn

# STEP 8: Verifica environment
print("\n✅ ENVIRONMENT VERIFICATION:")
import sys, torch, numpy, pandas
print(f"Python version: {sys.version}")
print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")

print("\n🎉 Environment setup completed! Now you can run your YOLOv5 training code.")

# Enhanced YOLOv5 Training for Oyster Detection - Full Drive Integration
# Tutto su Drive con resume training e export separato

import os, sys, time, zipfile, shutil, subprocess, shlex, json, re, glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import torch
import numpy as np

# =========================
# CONFIGURAZIONE DRIVE-FIRST
# =========================
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Config dataset e paths - TUTTO SU DRIVE
PROJECT_NAME = "enhanced_oyster_training"
WORK_DRIVE = "/content/drive/MyDrive/enhanced_oyster_project"
DATASET_ZIP_DRIVE = f"{WORK_DRIVE}/dataset/REU_Oyster_2024_Improved.v3i.yolov5pytorch.zip"  # Dataset ZIP su Drive
DATASET_DIR_DRIVE = f"{WORK_DRIVE}/dataset/extracted"     # Dataset estratto su Drive
YOLO_DIR_DRIVE = f"{WORK_DRIVE}/yolov5"                  # YOLOv5 su Drive
RUNS_ROOT_DRIVE = f"{WORK_DRIVE}/runs/train"
EXPORT_DIR_DRIVE = f"{WORK_DRIVE}/exported_models"
LOGS_DIR_DRIVE = f"{WORK_DRIVE}/logs"
HYP_DIR_DRIVE = f"{WORK_DRIVE}/hyp"
RESUME_DIR_DRIVE = f"{WORK_DRIVE}/resume"

# PARAMETRI TRAINING
EPOCHS = 200
SAVE_EVERY = 5         # Salva ogni 5 epoche per resume
IMG_SIZE = 640
BATCH_SIZE = 16
SINGLE_CLASS = True
CACHE_MODE = "disk"
WORKERS = 2            # Ridotto per Drive I/O
PATIENCE = 50

dataset_yaml_path = None
custom_hyp_path = None

# =========================
# UTILITY FUNCTIONS
# =========================
def _ascii(s: str) -> str:
    try: return s.encode("ascii","ignore").decode("ascii","ignore")
    except Exception: return s

def run(cmd, check=True, cwd=None, capture=False):
    print("> " + _ascii(" ".join(shlex.quote(str(c)) for c in cmd)))
    res = subprocess.run(
        cmd, check=False, cwd=cwd,
        stdout=(subprocess.PIPE if capture else None),
        stderr=(subprocess.PIPE if capture else None),
        text=True,
    )
    if capture and (res.stdout or res.stderr):
        print(_ascii((res.stdout or "") + (res.stderr or "")))
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    return res

def mount_drive():
    from google.colab import drive
    for i in range(3):
        try:
            print(f"🔌 Mounting Google Drive (attempt {i+1}/3)...")
            drive.mount("/content/drive")
            if os.path.exists("/content/drive/MyDrive"):
                print("✅ Drive mounted successfully")
                return True
        except Exception as e:
            print("❌ Mount failed:", _ascii(str(e)))
        time.sleep(2)
    print("❌ Unable to mount Drive")
    return False

def ensure_dirs():
    """Crea tutte le directory necessarie su Drive"""
    dirs_to_create = [
        f"{WORK_DRIVE}/dataset",
        f"{WORK_DRIVE}/analysis",
        RUNS_ROOT_DRIVE,
        EXPORT_DIR_DRIVE,
        LOGS_DIR_DRIVE,
        HYP_DIR_DRIVE,
        RESUME_DIR_DRIVE,
        DATASET_DIR_DRIVE
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"📁 Created/verified: {d}")

def setup_enhanced_env():
    """Setup environment - le dipendenze sono già installate"""
    print("✅ Dependencies already installed via Conda setup")

    try:
        import torch, numpy, pandas, yaml
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ NumPy: {numpy.__version__}")
        print(f"✅ Pandas: {pandas.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

# =========================
# YOLOV5 SETUP SU DRIVE
# =========================
def setup_yolov5_on_drive():
    """Setup YOLOv5 direttamente su Drive"""
    print("⚙️ Setting up YOLOv5 on Drive...")

    if not (Path(YOLO_DIR_DRIVE) / "train.py").exists():
        print("📥 Cloning YOLOv5 to Drive...")
        run(["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", YOLO_DIR_DRIVE])
        print("✅ YOLOv5 cloned to Drive")
    else:
        print("✅ YOLOv5 already present on Drive")

    # Installa requirements filtrati
    req = Path(YOLO_DIR_DRIVE) / "requirements.txt"
    if req.exists():
        txt = req.read_text().splitlines()
        filtered = [ln for ln in txt if not any(skip in ln.lower() for skip in ["torch", "torchvision", "pillow", "pandas", "numpy"])]
        if filtered:
            # Installa solo le dipendenze che mancano
            cmd = [sys.executable, "-m", "pip", "install"] + filtered
            try:
                run(cmd, capture=True, check=False)
            except:
                print("⚠️ Some requirements failed to install, continuing...")

    print("✅ YOLOv5 ready on Drive")

# =========================
# DATASET MANAGEMENT SU DRIVE
# =========================
def extract_dataset_from_drive():
    """Estrae il dataset dal ZIP su Drive"""
    if not os.path.exists(DATASET_ZIP_DRIVE):
        print(f"❌ Dataset ZIP not found at: {DATASET_ZIP_DRIVE}")
        return False

    if validate_dataset_on_drive():
        print("✅ Dataset already extracted on Drive")
        return True

    print("📦 Extracting dataset from Drive...")
    try:
        with zipfile.ZipFile(DATASET_ZIP_DRIVE, "r") as z:
            z.testzip()  # Verifica integrità
            z.extractall(DATASET_DIR_DRIVE)
        print("✅ Dataset extracted successfully")
        return validate_dataset_on_drive()
    except Exception as e:
        print(f"❌ Dataset extraction error: {_ascii(str(e))}")
        return False

def validate_dataset_on_drive():
    """Valida la struttura del dataset su Drive (root annidata + valid presente).
       Se manca 'val', crea symlink 'val' -> 'valid' per compatibilità.
       Non copia dati (risparmia spazio su Drive)."""
    import os, shutil
    from pathlib import Path

    global DATASET_DIR_DRIVE
    base = Path(DATASET_DIR_DRIVE)
    if not base.exists():
        return False

    def _has_yolo(p: Path) -> bool:
        return (p / "train" / "images").is_dir() and (p / "train" / "labels").is_dir()

    # Se la root non ha la struttura YOLO, prova a trovare una cartella annidata
    if not _has_yolo(base):
        for child in base.iterdir():
            if child.is_dir() and _has_yolo(child):
                DATASET_DIR_DRIVE = str(child)
                base = child
                print(f"🔁 YOLO root adjusted to nested folder: {base}")
                break
        else:
            print("❌ Missing required train directories")
            return False

    # A questo punto train/ esiste. Gestione valid/val
    valid_imgs = base / "valid" / "images"
    valid_lbls = base / "valid" / "labels"
    val_imgs   = base / "val"   / "images"
    val_lbls   = base / "val"   / "labels"

    # Se hai già 'valid', assicura compatibilità creando 'val' -> 'valid' se manca
    if valid_imgs.exists() and valid_lbls.exists():
        if not val_imgs.exists() or not val_lbls.exists():
            try:
                val_imgs.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(valid_imgs, val_imgs)
                os.symlink(valid_lbls, val_lbls)
                print("🔗 Created symlink: val -> valid")
            except Exception as e:
                print(f"⚠️ Could not symlink val->valid: {e} (ok, il YAML userà 'valid')")
    else:
        # Se per qualche motivo 'valid' non è completa, prova a derivarla da 'val' o 'test' (solo symlink)
        if val_imgs.exists() and val_lbls.exists():
            try:
                valid_imgs.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(val_imgs, valid_imgs)
                os.symlink(val_lbls, valid_lbls)
                print("🔗 Created symlink: valid -> val")
            except Exception as e:
                print(f"⚠️ Could not symlink valid->val: {e}")
        else:
            test_imgs = base / "test" / "images"
            test_lbls = base / "test" / "labels"
            if test_imgs.exists() and test_lbls.exists():
                try:
                    valid_imgs.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(test_imgs, valid_imgs)
                    os.symlink(test_lbls, valid_lbls)
                    print("🔗 Created symlink: valid -> test (fallback)")
                except Exception as e:
                    print(f"⚠️ Could not symlink valid->test: {e} (userai il fallback nel YAML)")
            else:
                print("⚠️ Nessun set di validazione trovato (valid/val/test)")

    print("✅ Dataset structure validated on Drive")
    return True


def analyze_dataset_on_drive():
    """Analizza il dataset su Drive"""
    analysis = {
        'total_images': 0,
        'total_labels': 0,
        'splits': {},
        'class_distribution': {}
    }

    for split in ["train", "valid", "val", "test"]:
        img_dir = Path(DATASET_DIR_DRIVE) / split / "images"
        lbl_dir = Path(DATASET_DIR_DRIVE) / split / "labels"

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))

        analysis['splits'][split] = {
            'images': len(images),
            'labels': len(labels)
        }
        analysis['total_images'] += len(images)
        analysis['total_labels'] += len(labels)

    print(f"📊 Dataset Analysis (Drive):")
    print(f"   Total images: {analysis['total_images']}")
    print(f"   Total labels: {analysis['total_labels']}")
    print(f"   Splits: {analysis['splits']}")

    return analysis

# =========================
# HYPERPARAMETERS
# =========================
def create_enhanced_hyperparams():
    """Crea hyperparameters ottimizzati su Drive"""
    global custom_hyp_path

    base_hyp = Path(YOLO_DIR_DRIVE) / "data/hyps/hyp.scratch-med.yaml"
    with open(base_hyp, "r") as f:
        hyp = yaml.safe_load(f)

    enhanced_hyp = {
        'lr0': 0.008,
        'lrf': 0.12,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,
        'cls': 0.3,
        'cls_pw': 1.0,
        'obj': 0.7,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'fl_gamma': 0.0,
        'hsv_h': 0.010,
        'hsv_s': 0.65,
        'hsv_v': 0.35,
        'degrees': 12.0,
        'translate': 0.08,
        'scale': 0.7,
        'shear': 2.0,
        'perspective': 0.0002,
        'flipud': 0.4,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.08,
        'copy_paste': 0.15,
        'label_smoothing': 0.05,
        'nbs': 64,
    }

    hyp.update(enhanced_hyp)
    custom_hyp_path = str(Path(HYP_DIR_DRIVE) / "hyp_oyster_enhanced.yaml")

    with open(custom_hyp_path, "w") as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    print(f"⚙️ Enhanced hyperparameters created: {custom_hyp_path}")
    return custom_hyp_path

# =========================
# YAML SETUP
# =========================
def setup_dataset_yaml():
    """Setup YAML del dataset su Drive"""
    global dataset_yaml_path

    dataset_yaml_path = f"{WORK_DRIVE}/dataset_enhanced.yaml"

    config = {
        "path": DATASET_DIR_DRIVE,
        "train": "train/images",
        "val": "valid/images" if (Path(DATASET_DIR_DRIVE) / "valid/images").exists() else "val/images",
        "test": "test/images" if (Path(DATASET_DIR_DRIVE) / "test/images").exists() else None,
        "nc": 1 if SINGLE_CLASS else 4,
        "names": ["oyster"] if SINGLE_CLASS else ["oyster-closed", "oyster-open", "oyster-juvenile", "oyster-damaged"]
    }

    with open(dataset_yaml_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Converti a single class se richiesto
    if SINGLE_CLASS:
        convert_to_single_class()

    print(f"📄 Dataset YAML created: {dataset_yaml_path}")
    return dataset_yaml_path

def convert_to_single_class():
    """Converte tutte le labels a single class"""
    converted_count = 0
    for split in ["train", "valid", "val", "test"]:
        labels_dir = Path(DATASET_DIR_DRIVE) / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            try:
                if label_file.stat().st_size == 0:
                    continue

                lines = label_file.read_text().strip().split('\n')
                if not lines or lines == ['']:
                    continue

                new_lines = []
                changed = False

                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        if parts[0] != "0":
                            parts[0] = "0"
                            changed = True
                        new_lines.append(" ".join(parts))

                if changed:
                    label_file.write_text("\n".join(new_lines) + "\n")
                    converted_count += 1

            except Exception as e:
                print(f"⚠️ Error processing {label_file}: {e}")

    if converted_count > 0:
        print(f"🔄 Converted {converted_count} label files to single-class")

# =========================
# RESUME TRAINING FUNCTIONS
# =========================
def save_training_state(epoch, run_dir, metrics=None):
    """Salva lo stato del training per resume"""
    state = {
        'epoch': epoch,
        'run_dir': str(run_dir),
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics or {},
        'config': {
            'epochs': EPOCHS,
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'project_name': PROJECT_NAME
        }
    }

    state_file = Path(RESUME_DIR_DRIVE) / f"training_state_epoch_{epoch}.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"💾 Training state saved: {state_file}")

def find_latest_training_state():
    """Trova l'ultimo stato di training salvato"""
    resume_files = list(Path(RESUME_DIR_DRIVE).glob("training_state_epoch_*.json"))
    if not resume_files:
        return None

    # Trova il file più recente
    latest_file = max(resume_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(latest_file, 'r') as f:
            state = json.load(f)
        print(f"📂 Found training state: {latest_file}")
        return state
    except Exception as e:
        print(f"❌ Error reading state file: {e}")
        return None

def find_resume_weights(run_dir):
    """Trova i weights per il resume"""
    if not run_dir or not Path(run_dir).exists():
        return None

    weights_dir = Path(run_dir) / "weights"
    if not weights_dir.exists():
        return None

    # Cerca last.pt o il più recente epoch weight
    last_weights = weights_dir / "last.pt"
    if last_weights.exists():
        return str(last_weights)

    # Cerca epoch weights
    epoch_weights = list(weights_dir.glob("epoch*.pt"))
    if epoch_weights:
        latest_weight = max(epoch_weights, key=lambda f: f.stat().st_mtime)
        return str(latest_weight)

    return None

def cleanup_old_resume_states(keep_last_n=3):
    """Pulisce vecchi stati di resume mantenendo solo gli ultimi N"""
    resume_files = list(Path(RESUME_DIR_DRIVE).glob("training_state_epoch_*.json"))
    if len(resume_files) <= keep_last_n:
        return

    # Ordina per data di modifica e rimuovi i più vecchi
    resume_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    for old_file in resume_files[keep_last_n:]:
        try:
            old_file.unlink()
            print(f"🗑️ Cleaned old resume state: {old_file.name}")
        except Exception as e:
            print(f"⚠️ Failed to clean {old_file}: {e}")

# =========================
# EXPORT FUNCTION (SEPARATA)
# =========================
def export_models_only():
    """Funzione separata per esportare modelli esistenti"""
    print("🚀" + "="*60)
    print("MODEL EXPORT ONLY")
    print("="*61 + "🚀")

    if not mount_drive():
        print("❌ Cannot proceed without Drive access")
        return False

    # Trova run directories esistenti
    if not Path(RUNS_ROOT_DRIVE).exists():
        print("❌ No training runs found")
        return False

    run_dirs = [d for d in Path(RUNS_ROOT_DRIVE).iterdir() if d.is_dir() and d.name.startswith("enhanced_oyster")]

    if not run_dirs:
        print("❌ No oyster training runs found")
        return False

    # Usa l'ultimo run
    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    print(f"📁 Using latest run: {latest_run}")

    return enhanced_export_for_jetson(str(latest_run))

def enhanced_export_for_jetson(run_dir):
    """Export ottimizzato per Jetson"""
    if not run_dir or not Path(run_dir).exists():
        print("❌ Run directory not found")
        return False

    weights_dir = Path(run_dir) / "weights"
    if not weights_dir.exists():
        print("❌ Weights directory not found")
        return False

    # Setup YOLOv5 se necessario
    yolo_local = "/content/yolov5"
    if not Path(yolo_local).exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", yolo_local])

    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    weights_to_export = []
    if best_weights.exists():
        weights_to_export.append(("best", best_weights))
    if last_weights.exists():
        weights_to_export.append(("last", last_weights))

    if not weights_to_export:
        print("❌ No weights found for export")
        return False

    print("🚀 Starting enhanced export for Jetson...")
    os.chdir(yolo_local)
    Path(EXPORT_DIR_DRIVE).mkdir(parents=True, exist_ok=True)

    export_configs = [
        {
            "name": "jetson_nano",
            "formats": ["onnx"],
            "imgsz": 416,
            "opset": 11,
            "simplify": True,
            "half": False
        },
        {
            "name": "jetson_xavier",
            "formats": ["onnx"],
            "imgsz": 640,
            "opset": 12,
            "simplify": True,
            "half": True
        },
        {
            "name": "production",
            "formats": ["onnx", "torchscript"],
            "imgsz": 640,
            "opset": 12,
            "simplify": True,
            "half": False
        }
    ]

    exported_count = 0

    for weight_name, weight_path in weights_to_export:
        print(f"\n📦 Exporting {weight_name}.pt...")

        for config in export_configs:
            try:
                for fmt in config["formats"]:
                    print(f"  🔄 Creating {config['name']} {fmt.upper()} model...")

                    cmd = [
                        sys.executable, "export.py",
                        "--weights", str(weight_path),
                        "--include", fmt,
                        "--imgsz", str(config["imgsz"]),
                        "--device", "0" if torch.cuda.is_available() else "cpu",
                    ]

                    if fmt == "onnx":
                        cmd.extend(["--opset", str(config["opset"])])
                        if config["simplify"]:
                            cmd.append("--simplify")

                    if config.get("half", False):
                        cmd.append("--half")

                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=yolo_local)

                    if result.returncode == 0:
                        # Trova e sposta file esportato
                        for ext in [f".{fmt}", f"_{fmt}"]:
                            potential_file = weight_path.with_suffix(ext)
                            if potential_file.exists():
                                final_name = f"oyster_{weight_name}_{config['name']}.{fmt}"
                                final_path = Path(EXPORT_DIR_DRIVE) / final_name
                                shutil.move(str(potential_file), str(final_path))

                                size_mb = final_path.stat().st_size / 1024 / 1024
                                print(f"    ✅ {final_name} ({size_mb:.1f} MB)")
                                exported_count += 1
                                break
                        else:
                            print(f"    ❌ Export file not found for {fmt}")
                    else:
                        print(f"    ❌ Export failed: {result.stderr}")

            except Exception as e:
                print(f"    ❌ Export error for {config['name']}: {_ascii(str(e))}")

    # Crea summary degli export
    if exported_count > 0:
        create_export_summary(weights_to_export, export_configs)
        print(f"✅ Successfully exported {exported_count} models to {EXPORT_DIR_DRIVE}")
        return True
    else:
        print("❌ No models were successfully exported")
        return False

def create_export_summary(weights_to_export, export_configs):
    """Crea un summary degli export effettuati"""
    summary = {
        'export_timestamp': datetime.now().isoformat(),
        'exported_weights': [w[0] for w in weights_to_export],
        'configurations': export_configs,
        'files': []
    }

    # Scansiona i file esportati
    for file in Path(EXPORT_DIR_DRIVE).glob("oyster_*.onnx"):
        summary['files'].append({
            'filename': file.name,
            'size_mb': file.stat().st_size / 1024 / 1024,
            'created': datetime.fromtimestamp(file.stat().st_ctime).isoformat()
        })

    summary_file = Path(EXPORT_DIR_DRIVE) / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"📄 Export summary created: {summary_file}")

# =========================
# TRAINING CON RESUME
# =========================
def create_training_command(resume_weights=None):
    """Crea comando di training con supporto resume"""
    os.chdir(YOLO_DIR_DRIVE)

    model_config = "models/yolov5m.yaml"
    pretrained_weights = resume_weights or "yolov5m.pt"

    cmd = [
        sys.executable, "train.py",
        "--data", dataset_yaml_path,
        "--cfg", model_config,
        "--weights", pretrained_weights,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--imgsz", str(IMG_SIZE),
        "--project", RUNS_ROOT_DRIVE,
        "--name", PROJECT_NAME,
        "--exist-ok",
        "--save-period", str(SAVE_EVERY),
        "--device", "0" if torch.cuda.is_available() else "cpu",
        "--workers", str(WORKERS),
        "--seed", "42",
        "--patience", str(PATIENCE),
        "--hyp", custom_hyp_path,
        "--cache", CACHE_MODE,
        "--multi-scale",
        "--cos-lr",
        "--label-smoothing", "0.05",
        "--optimizer", "Adam",   # 👈 rimpiazza il vecchio --adam
    ]

    if SINGLE_CLASS:
        cmd.append("--single-cls")

    if resume_weights and "last.pt" in resume_weights:
        cmd.append("--resume")

    return cmd

def train_with_resume_support():
    """Training con supporto completo per resume"""
    print("🚀 Starting training with resume support...")

    # Controlla se c'è un training precedente da riprendere
    resume_state = find_latest_training_state()
    resume_weights = None
    start_epoch = 0

    if resume_state:
        print(f"📂 Found previous training state:")
        print(f"   Last epoch: {resume_state['epoch']}")
        print(f"   Run directory: {resume_state['run_dir']}")

        response = input("🤔 Do you want to resume from this checkpoint? (y/n): ").lower().strip()
        if response == 'y':
            resume_weights = find_resume_weights(resume_state['run_dir'])
            if resume_weights:
                start_epoch = resume_state['epoch']
                print(f"🔄 Resuming from epoch {start_epoch} with weights: {resume_weights}")
            else:
                print("⚠️ Resume weights not found, starting fresh")

    cmd = create_training_command(resume_weights)

    print(f"📊 Dataset: Located on Drive")
    print(f"🎯 Target: mAP@0.5 > 0.75, mAP@0.5:0.95 > 0.45")
    print(f"⚙️ Model: YOLOv5m (enhanced)")
    print(f"🖼️ Image size: {IMG_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")

    print("\n" + "="*80)
    print("TRAINING COMMAND:")
    print(_ascii(" ".join(shlex.quote(str(c)) for c in cmd)))
    print("="*80 + "\n")

    # Esegui training
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=YOLO_DIR_DRIVE
    )

    # Variabili monitoring
    current_epoch = start_epoch
    best_map50 = 0.0
    best_map5095 = 0.0
    start_time = time.time()
    last_log_time = time.time()
    last_save_time = time.time()

    # CSV per tracking
    progress_csv = Path(WORK_DRIVE) / "training_progress.csv"
    progress_data = []

    try:
        for line in proc.stdout:
            line_clean = _ascii(line.rstrip())
            print(line_clean)

            # Parse epoch info
            if "Epoch" in line_clean and "/" in line_clean:
                try:
                    epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line_clean)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))

                        # Salva stato ogni SAVE_EVERY epoche
                        if current_epoch % SAVE_EVERY == 0:
                            run_dir = str(Path(RUNS_ROOT_DRIVE) / PROJECT_NAME)
                            save_training_state(current_epoch, run_dir, {
                                'best_map50': best_map50,
                                'best_map5095': best_map5095
                            })
                            last_save_time = time.time()
                            cleanup_old_resume_states()  # Pulisci vecchi stati
                except:
                    pass

            # Parse metriche e salva in CSV
            if "metrics/mAP_0.5:" in line_clean:
                try:
                    parts = line_clean.split()
                    metrics = {}
                    for i, part in enumerate(parts):
                        if "mAP_0.5:" in part and i + 1 < len(parts):
                            metrics['mAP_0.5:0.95'] = float(parts[i + 1])
                        elif "mAP_0.5" in part and "mAP_0.5:" not in part and i + 1 < len(parts):
                            metrics['mAP_0.5'] = float(parts[i + 1])

                    if metrics:
                        metrics['epoch'] = current_epoch
                        metrics['timestamp'] = datetime.now().isoformat()
                        progress_data.append(metrics)

                        # Aggiorna best scores
                        if 'mAP_0.5' in metrics and metrics['mAP_0.5'] > best_map50:
                            best_map50 = metrics['mAP_0.5']
                            print(f"🎯 NEW BEST mAP@0.5: {best_map50:.4f}")

                        if 'mAP_0.5:0.95' in metrics and metrics['mAP_0.5:0.95'] > best_map5095:
                            best_map5095 = metrics['mAP_0.5:0.95']
                            print(f"🎯 NEW BEST mAP@0.5:0.95: {best_map5095:.4f}")

                        # Salva CSV progress
                        if progress_data:
                            df = pd.DataFrame(progress_data)
                            df.to_csv(progress_csv, index=False)
                except:
                    pass

            # Log progress periodicamente
            current_time = time.time()
            if current_time - last_log_time > 300:  # Ogni 5 minuti
                elapsed = current_time - start_time
                if current_epoch > start_epoch:
                    eta = (elapsed / (current_epoch - start_epoch)) * (EPOCHS - current_epoch)
                    print(f"📈 Progress: Epoch {current_epoch}/{EPOCHS} | Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")
                    print(f"🏆 Best mAP@0.5: {best_map50:.4f} | mAP@0.5:0.95: {best_map5095:.4f}")
                last_log_time = current_time

        return_code = proc.wait()

        # Salva stato finale
        final_run_dir = str(Path(RUNS_ROOT_DRIVE) / PROJECT_NAME)
        save_training_state(current_epoch, final_run_dir, {
            'best_map50': best_map50,
            'best_map5095': best_map5095,
            'completed': True
        })

        if return_code == 0:
            print("✅ Training completed successfully!")

            # Crea summary finale
            create_training_summary(final_run_dir, progress_data, {
                'best_map50': best_map50,
                'best_map5095': best_map5095,
                'total_epochs': current_epoch,
                'training_time': time.time() - start_time
            })

            return True
        else:
            print(f"❌ Training failed with return code: {return_code}")
            return False

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        proc.terminate()

        # Salva stato di interruzione
        final_run_dir = str(Path(RUNS_ROOT_DRIVE) / PROJECT_NAME)
        save_training_state(current_epoch, final_run_dir, {
            'best_map50': best_map50,
            'best_map5095': best_map5095,
            'interrupted': True
        })

        return False

    except Exception as e:
        print(f"❌ Training error: {_ascii(str(e))}")
        proc.terminate()
        return False

def create_training_summary(run_dir, progress_data, final_metrics):
    """Crea un summary completo del training"""
    summary = {
        'training_completed': datetime.now().isoformat(),
        'run_directory': run_dir,
        'configuration': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'img_size': IMG_SIZE,
            'single_class': SINGLE_CLASS,
            'patience': PATIENCE
        },
        'final_metrics': final_metrics,
        'progress_data_points': len(progress_data),
        'training_time_hours': final_metrics.get('training_time', 0) / 3600
    }

    # Aggiungi statistiche sui pesi
    weights_dir = Path(run_dir) / "weights"
    if weights_dir.exists():
        weights_files = list(weights_dir.glob("*.pt"))
        summary['weights_saved'] = len(weights_files)
        summary['weight_files'] = [w.name for w in weights_files]

    summary_file = Path(LOGS_DIR_DRIVE) / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Training summary saved: {summary_file}")
    return summary_file

# =========================
# MONITORING E ANALISI
# =========================
def monitor_training_progress():
    """Monitora il progresso del training in corso"""
    progress_csv = Path(WORK_DRIVE) / "training_progress.csv"

    if not progress_csv.exists():
        print("❌ No training progress data found")
        return False

    try:
        df = pd.read_csv(progress_csv)

        if df.empty:
            print("❌ No progress data available")
            return False

        latest = df.iloc[-1]

        print("📊 CURRENT TRAINING STATUS")
        print("="*50)
        print(f"Current Epoch: {latest.get('epoch', 'N/A')}")
        print(f"Latest mAP@0.5: {latest.get('mAP_0.5', 'N/A'):.4f}")
        print(f"Latest mAP@0.5:0.95: {latest.get('mAP_0.5:0.95', 'N/A'):.4f}")
        print(f"Best mAP@0.5: {df['mAP_0.5'].max():.4f}")
        print(f"Best mAP@0.5:0.95: {df['mAP_0.5:0.95'].max():.4f}")
        print(f"Total Progress Points: {len(df)}")

        # Trend analysis
        if len(df) >= 5:
            recent_map50 = df['mAP_0.5'].tail(5).mean()
            earlier_map50 = df['mAP_0.5'].head(max(1, len(df)-10)).tail(5).mean()
            trend = "📈 Improving" if recent_map50 > earlier_map50 else "📉 Declining"
            print(f"Recent Trend: {trend}")

        return True

    except Exception as e:
        print(f"❌ Error reading progress: {e}")
        return False

def analyze_completed_training():
    """Analizza i risultati di training completati"""
    if not Path(RUNS_ROOT_DRIVE).exists():
        print("❌ No training runs found")
        return False

    run_dirs = [d for d in Path(RUNS_ROOT_DRIVE).iterdir() if d.is_dir()]

    if not run_dirs:
        print("❌ No run directories found")
        return False

    print("📊 COMPLETED TRAINING ANALYSIS")
    print("="*60)

    for run_dir in sorted(run_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue

        try:
            df = pd.read_csv(results_csv)
            if df.empty:
                continue

            best_row = df.loc[df['metrics/mAP_0.5'].idxmax()]

            print(f"\n📁 Run: {run_dir.name}")
            print(f"   Epochs completed: {len(df)}")
            print(f"   Best mAP@0.5: {best_row['metrics/mAP_0.5']:.4f} (epoch {best_row.name + 1})")
            print(f"   Best mAP@0.5:0.95: {best_row['metrics/mAP_0.5:0.95']:.4f}")
            print(f"   Final Loss: {df['train/box_loss'].iloc[-1]:.4f}")

            # Check weights
            weights_dir = run_dir / "weights"
            if weights_dir.exists():
                weights = list(weights_dir.glob("*.pt"))
                print(f"   Weights available: {len(weights)} files")

        except Exception as e:
            print(f"   ⚠️ Error analyzing {run_dir.name}: {e}")

    return True

# =========================
# MAIN EXECUTION FUNCTIONS
# =========================
def full_training_pipeline():
    """Pipeline completo di training"""
    print("🚀" + "="*60)
    print("ENHANCED YOLOV5 OYSTER TRAINING PIPELINE")
    print("="*61 + "🚀")

    # 1. Mount Drive
    if not mount_drive():
        print("❌ Cannot proceed without Drive access")
        return False

    # 2. Setup directories
    ensure_dirs()

    # 3. Setup environment
    if not setup_enhanced_env():
        print("❌ Environment setup failed")
        return False

    # 4. Setup YOLOv5 on Drive
    setup_yolov5_on_drive()

    # 5. Extract and validate dataset
    if not extract_dataset_from_drive():
        print("❌ Dataset setup failed")
        return False

    # 6. Analyze dataset
    analyze_dataset_on_drive()

    # 7. Setup YAML and hyperparameters
    setup_dataset_yaml()
    create_enhanced_hyperparams()

    # 8. Start training with resume support
    success = train_with_resume_support()

    if success:
        print("✅ Training pipeline completed successfully!")

        # 9. Optional: Export models immediately
        response = input("🤔 Do you want to export models now? (y/n): ").lower().strip()
        if response == 'y':
            export_models_only()

    return success

def resume_interrupted_training():
    """Resume specifico per training interrotti"""
    print("🔄" + "="*60)
    print("RESUME INTERRUPTED TRAINING")
    print("="*61 + "🔄")

    if not mount_drive():
        print("❌ Cannot proceed without Drive access")
        return False

    # Trova l'ultimo stato
    resume_state = find_latest_training_state()
    if not resume_state:
        print("❌ No interrupted training found")
        return False

    print(f"📂 Found interrupted training:")
    print(f"   Last epoch: {resume_state['epoch']}")
    print(f"   Run directory: {resume_state['run_dir']}")
    print(f"   Timestamp: {resume_state['timestamp']}")

    # Setup necessario
    ensure_dirs()
    setup_yolov5_on_drive()
    setup_dataset_yaml()
    create_enhanced_hyperparams()

    # Resume automatico
    return train_with_resume_support()

def export_only_pipeline():
    """Pipeline solo per export"""
    return export_models_only()

def status_check():
    """Controlla lo stato generale del sistema"""
    print("📊" + "="*60)
    print("SYSTEM STATUS CHECK")
    print("="*61 + "📊")

    if not mount_drive():
        print("❌ Drive not accessible")
        return False

    # Check directories
    print("\n📁 Directory Status:")
    for dir_name, dir_path in [
        ("Work Drive", WORK_DRIVE),
        ("Dataset", DATASET_DIR_DRIVE),
        ("YOLOv5", YOLO_DIR_DRIVE),
        ("Runs", RUNS_ROOT_DRIVE),
        ("Export", EXPORT_DIR_DRIVE),
        ("Resume", RESUME_DIR_DRIVE)
    ]:
        status = "✅ EXISTS" if Path(dir_path).exists() else "❌ MISSING"
        print(f"   {dir_name}: {status}")

    # Check dataset
    print(f"\n📦 Dataset Status:")
    if Path(DATASET_ZIP_DRIVE).exists():
        print("   ZIP: ✅ Found")
    else:
        print("   ZIP: ❌ Missing")

    if validate_dataset_on_drive():
        print("   Extracted: ✅ Valid")
        analyze_dataset_on_drive()
    else:
        print("   Extracted: ❌ Invalid/Missing")

    # Check training progress
    print(f"\n🏃 Training Status:")
    monitor_training_progress()

    # Check completed runs
    print(f"\n📈 Completed Runs:")
    analyze_completed_training()

    # Check exports
    if Path(EXPORT_DIR_DRIVE).exists():
        exports = list(Path(EXPORT_DIR_DRIVE).glob("*.onnx")) + list(Path(EXPORT_DIR_DRIVE).glob("*.pt"))
        print(f"\n📦 Exported Models: {len(exports)} files")
        for exp in exports[:5]:  # Show first 5
            size_mb = exp.stat().st_size / 1024 / 1024
            print(f"   {exp.name} ({size_mb:.1f} MB)")
        if len(exports) > 5:
            print(f"   ... and {len(exports) - 5} more")

    return True

# =========================
# GOOGLE COLAB ONLY
# =========================
def _check_colab():
    """Verifica che siamo su Google Colab"""
    try:
        import google.colab
        print("✅ Running on Google Colab")
        return True
    except ImportError:
        print("❌ This script is designed ONLY for Google Colab!")
        print("Please run this in Google Colab environment.")
        return False

# =========================
# CONFIGURATION FUNCTIONS
# =========================
def set_training_config(epochs=200, batch_size=16, img_size=640, single_class=True, patience=50):
    """Configura i parametri di training"""
    global EPOCHS, BATCH_SIZE, IMG_SIZE, SINGLE_CLASS, PATIENCE
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    IMG_SIZE = img_size
    SINGLE_CLASS = single_class
    PATIENCE = patience

    print(f"⚙️ Training Configuration Updated:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Single Class: {SINGLE_CLASS}")
    print(f"   Patience: {PATIENCE}")

# Auto-check solo per Colab
if not _check_colab():
    raise RuntimeError("This script requires Google Colab environment")

# =========================
# GOOGLE COLAB INTERFACE
# =========================
def run_full_training():
    """Training completo - Interfaccia per Google Colab"""
    print("🚀 Starting Full Training Pipeline...")
    return full_training_pipeline()

def run_resume_training():
    """Resume training interrotto - Interfaccia per Google Colab"""
    print("🔄 Starting Resume Training...")
    return resume_interrupted_training()

def run_export_only():
    """Solo esportazione modelli - Interfaccia per Google Colab"""
    print("📦 Starting Export Only...")
    return export_only_pipeline()

def run_status_check():
    """Controllo stato sistema - Interfaccia per Google Colab"""
    print("📊 Checking System Status...")
    return status_check()

def quick_setup():
    """Setup rapido per Google Colab"""
    print("⚡ QUICK SETUP FOR GOOGLE COLAB")
    print("="*50)

    print("1. Mounting Google Drive...")
    if not mount_drive():
        return False

    print("2. Creating directories...")
    ensure_dirs()

    print("3. Setting up environment...")
    setup_enhanced_env()

    print("✅ Quick setup completed! You can now run:")
    print("   • run_full_training() - Complete training")
    print("   • run_resume_training() - Resume interrupted training")
    print("   • run_export_only() - Export existing models")
    print("   • run_status_check() - Check system status")

    return True

# =========================
# COLAB HELPERS
# =========================
def show_training_options():
    """Mostra opzioni disponibili per Google Colab"""
    print("🎯 GOOGLE COLAB TRAINING OPTIONS")
    print("="*50)
    print("1. run_full_training()    - Complete training pipeline")
    print("2. run_resume_training()  - Resume interrupted training")
    print("3. run_export_only()      - Export models only")
    print("4. run_status_check()     - System status check")
    print("5. monitor_training_progress() - Current training progress")
    print("6. analyze_completed_training() - Analyze past runs")
    print()
    print("⚙️ CONFIGURATION:")
    print("   set_training_config(epochs=200, batch_size=16, img_size=640)")
    print()
    print("📊 MONITORING:")
    print("   monitor_training_progress() - Check current progress")
    print("   analyze_completed_training() - Review completed runs")

def colab_menu():
    """Menu interattivo per Google Colab"""
    print("🤖 ENHANCED YOLOV5 OYSTER TRAINING - GOOGLE COLAB")
    print("="*60)
    print("Choose an option:")
    print("1. Full Training Pipeline")
    print("2. Resume Interrupted Training")
    print("3. Export Models Only")
    print("4. System Status Check")
    print("5. Show All Options")
    print("0. Exit")

    try:
        choice = input("\nEnter your choice (0-5): ").strip()

        if choice == "1":
            return run_full_training()
        elif choice == "2":
            return run_resume_training()
        elif choice == "3":
            return run_export_only()
        elif choice == "4":
            return run_status_check()
        elif choice == "5":
            show_training_options()
            return True
        elif choice == "0":
            print("👋 Goodbye!")
            return True
        else:
            print("❌ Invalid choice")
            return False

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return True

# =========================
# GOOGLE COLAB INITIALIZATION
# =========================
print("🚀 Enhanced YOLOv5 Oyster Detection Training System")
print("="*60)
print("🐚 GOOGLE COLAB ONLY - Optimized for GPU training")
print()
print("📋 QUICK START:")
print("1. quick_setup()           # Initial setup")
print("2. run_full_training()     # Start training")
print()
print("🔧 OR CUSTOMIZE:")
print("1. set_training_config(epochs=300, batch_size=32)")
print("2. run_full_training()")
print()
print("📱 INTERACTIVE: colab_menu()")
print("="*60)

# Usage examples for Google Colab:
"""
🚀 GOOGLE COLAB USAGE GUIDE:

=== REQUIREMENTS ===
• Google Colab with GPU enabled
• Dataset ZIP at: /content/drive/MyDrive/enhanced_oyster_project/dataset/dataset.zip
• YOLO format: train/images, train/labels, val/images, val/labels

=== QUICK START ===
quick_setup()                    # Setup Google Drive e directories
run_full_training()             # Training completo

=== CUSTOM CONFIGURATION ===
set_training_config(
    epochs=300,          # Numero epoche
    batch_size=32,       # Batch size (adatta alla GPU)
    img_size=640,        # Dimensione immagini
    single_class=True,   # Single class mode
    patience=50          # Early stopping
)
run_full_training()

=== RESUME INTERRUPTED TRAINING ===
run_resume_training()           # Riprende automaticamente

=== EXPORT MODELS ONLY ===
run_export_only()              # Esporta per Jetson

=== MONITORING ===
run_status_check()             # Stato completo
monitor_training_progress()    # Progresso corrente
analyze_completed_training()   # Analisi runs completati

=== INTERACTIVE MODE ===
colab_menu()                   # Menu guidato

=== DRIVE STRUCTURE ===
/content/drive/MyDrive/enhanced_oyster_project/
├── dataset/dataset.zip        # IL TUO DATASET QUI
├── dataset/extracted/         # Estratto automaticamente
├── yolov5/                    # YOLOv5 clonato
├── runs/train/                # Training results
├── exported_models/           # ONNX/TorchScript per Jetson
├── logs/                      # Training logs
├── hyp/                       # Hyperparameters
└── resume/                    # Auto-resume states

🔥 OTTIMIZZATO PER GOOGLE COLAB GPU!
"""

c

quick_setup()

colab_menu()

run_full_training()
