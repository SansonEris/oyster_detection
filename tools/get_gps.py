import requests
import time
import math
import csv
from pathlib import Path

def log_gps_periodically(url, interval_s=0.2, csv_path="gps.csv"):
    """
    Polling REST di GLOBAL_POSITION_INT con timestamp midpoint.
    Salva: ts_req_ms, ts_resp_ms, ts_mid_ms, lat_deg, lon_deg, alt_m, rel_alt_m, hdg_deg, vx_mps, vy_mps, vz_mps
    """
    p = Path(csv_path)
    new_file = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts_req_ms","ts_resp_ms","ts_mid_ms",
                        "lat_deg","lon_deg","alt_m","rel_alt_m","hdg_deg",
                        "vx_mps","vy_mps","vz_mps"])

        print(f"Polling {url} ogni {interval_s}s... Ctrl+C per uscire.")
        try:
            while True:
                t_req = time.time()  
                try:
                    r = requests.get(url, timeout=2.0)
                except requests.RequestException as e:
                    print("Request error:", e)
                    time.sleep(interval_s)
                    continue
                t_resp = time.time()

                if r.status_code != 200:
                    print("HTTP", r.status_code)
                    time.sleep(interval_s)
                    continue

                try:
                    data = r.json()
                except ValueError:
                    print("JSON non valido")
                    time.sleep(interval_s)
                    continue

                msg = data.get("message", {}) or {}

                # Estrai e scala con fallback None-safe
                def get_scaled(key, div, default=None):
                    v = msg.get(key)
                    return (v / div) if isinstance(v, (int, float)) else default

                lat_deg = get_scaled("lat", 1e7)
                lon_deg = get_scaled("lon", 1e7)
                alt_m = get_scaled("alt", 1000)
                rel_alt_m = get_scaled("relative_alt", 1000)
                hdg_deg = get_scaled("hdg", 100)  # potrebbe mancare, dipende dal firmware
                vx_mps = get_scaled("vx", 100)
                vy_mps = get_scaled("vy", 100)
                vz_mps = get_scaled("vz", 100)

                ts_req_ms = int(t_req * 1000)
                ts_resp_ms = int(t_resp * 1000)
                ts_mid_ms = (ts_req_ms + ts_resp_ms) // 2  # timestamp stimato del fix

                w.writerow([ts_req_ms, ts_resp_ms, ts_mid_ms,
                            lat_deg, lon_deg, alt_m, rel_alt_m, hdg_deg,
                            vx_mps, vy_mps, vz_mps])
                f.flush()

                # Debug leggibile
                print(f"{time.strftime('%H:%M:%S')} mid={ts_mid_ms} lat={lat_deg} lon={lon_deg} hdg={hdg_deg}")

                # intervallo costante rispetto a fine ciclo
                remaining = interval_s - (time.time() - t_req)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("Stop.")

if __name__ == "__main__":
    base = "http://192.168.2.2:6040"
    url = f"{base}/v1/mavlink/vehicles/1/components/1/messages/GLOBAL_POSITION_INT"
    log_gps_periodically(url, interval_s=0.2, csv_path="gps.csv")  # 5 Hz
