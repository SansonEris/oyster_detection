Patch bundle: stereo size estimation integrated in detection pipeline.

Key points:
- Stereo size estimation uses calibration from `stereo_calibration_data.pkl`
  (P1/P2/Q, rectification maps). Units follow your calibration (cm).
- Two streams (SINISTRA/DESTRA) retain red overlay style. The left stream
  shows size text above each detection.
- CSV `detections.csv` now includes extra columns per detection:
    Z,W,H,units
  (Right stream entries keep size fields empty to avoid duplication.)
- `detect_api.py` exposes the same API expected by `app.py` and `index.html`.
  No other behavior was changed.
- `detect.py` is the CLI twin with the same logic.
- `size_estimation_video.py` is kept for reference/testing.
- Calibration and logs included as provided.

Files included:
- detect.py (patched, full)
- detect_api.py (patched, full)
- size_estimation_video.py (reference)
- app.py, index.html (unchanged)
- CalibrationCamera.py, calibration_log.txt (reference)
- stereo_calibration_data.pkl (reference data)