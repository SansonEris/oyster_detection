# Oyster Detection System — Guida rapida

Sistema per il rilevamento di ostriche con YOLO, interfaccia web per il setup e strumenti dati.

## Requisiti

* Python 3.10+
* Dipendenze: `pip install -r requirements.txt`

## File chiave

```
app.py            # API/Server Flask
index.html        # UI (Detection + Tools)
detect_api.py     # Motore di detection (usato da app.py)
detect.py         # CLI standalone (opzionale)
models/           # Modelli YOLO
videos/           # Sorgenti video
outputs/          # Risultati per run (frame + CSV)
```

## Struttura del progetto

```
.
├── README.md
├── requirements.txt
├── app.py
├── detect.py
├── detect_api.py
├── index.html
├── results.png
├── detections.csv
├── final.csv
├── frames.csv
├── gitssh
├── gitssh.pub
├── gps.csv
├── outputs
├── calibration
│   └── stereo_calibration_data.pkl
├── models
│   └── best_yolo8.pt
├── static
│   └── app.css
├── tools
│   ├── combine_csv.py
│   ├── data_tools.html
│   ├── frames_from_images.py
│   ├── get_gps.py
│   └── make_heatmap.py
└── videos
    ├── TEST_LEFT_2.mp4
    └── TEST_RIGHT_2.mp4 
```

## Modelli e video supportati

* **Modelli** (metti in `models/`): `yolo8.onnx`, `yolo8.pt`, `yolo11.onnx`, `yolo11.onnx` 
* **Video validi** (metti in `videos/`): `TEST_LEFT_2.mp4`, `TEST_RIGHT_2.mp4`

## Avvio (Web UI)

1. Avvia il server:

   ```bash
   python app.py
   ```
2. Apri `http://localhost:5000`.
3. Tab **Detection** → imposta:

   * **Mode**: `stereo` (o `mono` se vuoi un solo video)
   * **Model**: `best_yolo8.onnx` (o `best_yolo8.pt`)
   * **Primary/Secondary**: `TEST_LEFT_2.mp4` / `TEST_RIGHT_2.mp4`
   * **Conf/IoU**: default 0.25 / 0.45
   * **Output**: `outputs/`
4. **Start Detection**. L’anteprima dei frame appare durante l’elaborazione.

### Output per run

* Cartella: `outputs/run_YYYYmmdd_HHMMSS[_N]/`
* File: `frame_*.jpg`, `detections.csv`, `detection_log.txt`

## Avvio (CLI)

Esecuzione rapida senza UI:

```bash
python detect.py \
  --model models/best_yolo8.onnx \
  --mode stereo \
  --left videos/TEST_LEFT_2.mp4 \
  --right videos/TEST_RIGHT_2.mp4 \
  --conf 0.25 --iou 0.45 \
  --output outputs/
```

> Per modalità mono: rimuovi `--right` e usa `--mode mono`.

## Tools (GPS/Frames/Merge/Heatmap)

Apri il tab **Tools** (`/tools`).

* **GPS Logger** → genera `gps.csv` da `mavlink2rest` della blue boat (imposta base URL e Hz). 
* **Frames CSV** → da cartella immagini crea della rosbag `frames.csv` (richiede nomi `SSSSSSSSS.NNNNNNNNN.png/jpg`).
* **Align & Fuse** → allinea `frames.csv` + `gps.csv` (+ opzionale `detections.csv`) e produce `final.csv`.
* **Heatmap** → genera `outputs/heatmap.html` da `final.csv`.
