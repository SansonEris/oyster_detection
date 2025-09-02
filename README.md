# Oyster Detection System

Sistema di detection per ostriche utilizzando YOLO con interfaccia web per la configurazione dei parametri.

## Setup

### 1. Installazione dipendenze
```bash
pip install -r requirements.txt
```

### 2. Struttura cartelle
```
project/
├── app.py              # API Flask
├── detect_api.py       # Classe OysterDetector
├── detect.py           # Script standalone (opzionale)
├── index.html          # Interfaccia web
├── requirements.txt    # Dipendenze Python
├── models/            # Modelli YOLO (.pt, .onnx)
│   └── best_yolo8.onnx
├── videos/            # Video sorgente
│   ├── left.mp4
│   └── right.mp4
└── outputs/           # Output generati
```

### 3. Aggiunta files
- Posiziona i tuoi modelli YOLO nella cartella `models/`
- Posiziona i video da processare nella cartella `videos/`

### 4. Avvio applicazione web

```bash
python app.py
```

L'interfaccia sarà disponibile su: http://localhost:5000
