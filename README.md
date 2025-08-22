# Stereo Oyster Detection

Sistema di rilevamento ostriche con YOLOv5 ONNX e Docker. Supporta video stereo, video singoli e streaming live.

## 🚀 Setup Rapido

### 1. Preparazione
```bash
# Clona il progetto
git clone <your-repo>
cd stereo-oyster-detection

# Rendi eseguibili gli script
chmod +x runDocker.sh setup.sh

# Setup automatico struttura progetto
./setup.sh
```

### 2. Aggiungi i tuoi file
```bash
# Copia il modello ONNX
cp /path/to/your/model.onnx models/best.onnx

# Copia i video di test (opzionale)
cp left_camera.mp4 videos/left.mp4
cp right_camera.mp4 videos/right.mp4
```

### 3. Avvio
```bash
# Menu interattivo (raccomandato)
./runDocker.sh

# Oppure comando diretto
./runDocker.sh --model /workspace/models/best.onnx \
               --left /workspace/videos/left.mp4 \
               --right /workspace/videos/right.mp4 \
               --http --save-video
```

## 📁 Struttura File
```
stereo-oyster-detection/
├── models/best.onnx          # Il tuo modello (obbligatorio)
├── videos/                   # Video di test
├── outputs/                  # Risultati generati
├── runDocker.sh              # Script principale
└── setup.sh                  # Setup automatico
```

## 🎮 Modalità di Utilizzo

### Menu Interattivo
```bash
./runDocker.sh
```
Scegli tra:
1. **Video stereo** (2 file video)
2. **Stream live** (URL RTSP/HTTP)  
3. **Video singolo** (1 file video)
4. **Shell** (accesso container)

### Comandi Diretti

**Video stereo:**
```bash
./runDocker.sh --model /workspace/models/best.onnx \
               --left /workspace/videos/left.mp4 \
               --right /workspace/videos/right.mp4 \
               --http --save-video
```

**Video singolo:**
```bash
./runDocker.sh --model /workspace/models/best.onnx \
               --left /workspace/videos/video.mp4 \
               --mono --http --save-video
```

**Stream live:**
```bash
./runDocker.sh --model /workspace/models/best.onnx \
               --left "rtsp://ip:port/stream1" \
               --right "rtsp://ip:port/stream2" \
               --http
```

## 📊 Visualizzazione

- **Console**: FPS e conteggi in tempo reale
- **HTTP Stream**: Apri http://localhost:5000 (se `--http` attivo)
- **Video salvato**: Disponibile in `outputs/` (se `--save-video` attivo)

## ⚙️ Parametri Principali

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--conf` | 0.25 | Soglia confidenza detection |
| `--size` | 640 | Risoluzione input (416 per Jetson Nano) |
| `--http` | - | Abilita streaming web |
| `--mono` | - | Modalità video singolo |
| `--save-video` | - | Salva video con detection |
| `--no-display` | - | Modalità headless |

## 🔧 Ottimizzazioni

**Performance migliori (Jetson Nano):**
```bash
./runDocker.sh --size 416 --conf 0.3
```

**Qualità maggiore (PC potente):**
```bash
./runDocker.sh --size 640 --conf 0.2
```

**Solo streaming (server headless):**
```bash
./runDocker.sh --http --no-display
```

## ❗ Risoluzione Problemi

**Modello non trovato:**
```bash
# Verifica esistenza
ls models/best.onnx
# Se manca, copialo nella cartella models/
```

**Video non si aprono:**
```bash
# Testa formato supportato
./runDocker.sh --shell
# Dentro il container: python -c "import cv2; print(cv2.__version__)"
```

**Container non parte:**
```bash
# Rebuild immagine
./runDocker.sh --rebuild
```

**Performance lente:**
```bash
# Riduci risoluzione
./runDocker.sh --size 416
```

## 📝 Note Veloci

- **Jetson Nano**: Usa `--size 416` sempre
- **Stream HTTP**: Porta 5000 di default
- **Output**: File salvati in `outputs/`
- **Logs**: Visibili in console durante esecuzione
- **Stop**: Premi `Ctrl+C` per fermare

## 🏁 Comandi Utili

```bash
# Help completo
./runDocker.sh --help

# Setup iniziale
./setup.sh

# Accesso shell container
./runDocker.sh --shell

# Rebuild forzato
./runDocker.sh --rebuild

# Test sistema
./runDocker.sh --system-check
```
