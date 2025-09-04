import argparse, os, time, html

TEMPLATE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>Oyster Heatmap</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
  <script src="https://unpkg.com/papaparse@5.4.1/papaparse.min.js"></script>
  <style>
    html,body,#map{height:100%;margin:0}
    /* Legend */
    .legend {
      position:absolute; right:12px; bottom:12px; z-index:1000;
      background:#2f3540; color:#fff; border:1px solid #404652; border-radius:8px;
      padding:8px 10px; font:12px/1.2 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      box-shadow:0 4px 16px rgba(0,0,0,.25);
    }
    .legend .row { display:flex; align-items:center; gap:8px; }
    .legend .label { min-width:36px; text-align:center; opacity:.95 }
    .legend .bar {
      width:240px; height:12px; border-radius:6px; border:1px solid #404652; 
      background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
    }
    .legend .cap { margin-top:4px; opacity:.75 }
    /* Hover info */
    .hover-info{
      position:absolute; right:12px; top:12px; z-index:1000;
      background:#2f3540; color:#fff; border:1px solid #404652; border-radius:8px;
      padding:8px 10px; font:12px/1.2 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;
      box-shadow:0 4px 16px rgba(0,0,0,.25);
    }
    .hover-info .muted{opacity:.7}
  </style>
</head>
<body>
<div id="map"></div>

<!-- Hover readout -->
<div class="hover-info" id="hoverBox">
  <div><b>Posizione</b> <span class="muted">(lat, lon)</span></div>
  <div id="hoverLatLon">–</div>
  <div style="margin-top:6px"><b>Nearest</b> <span class="muted">(entro 30 px)</span></div>
  <div id="hoverNearest">–</div>
</div>

<!-- Legend -->
<div class="legend" id="legend" style="display:none">
  <div class="row">
    <span class="label" id="legMin">0</span>
    <div class="bar"></div>
    <span class="label" id="legMax">max</span>
  </div>
  <div class="cap">Intensità (oysters_total normalizzato)</div>
</div>

<script>
  const CSV_PATH = %CSV%;
  const RADIUS = %RADIUS%;
  const BLUR   = %BLUR%;

  const map = L.map('map', { preferCanvas: true });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 20, attribution: "&copy; OpenStreetMap"
  }).addTo(map);

  let rows = []; // conserveremo le righe per hover nearest

  function distPx(a, b){
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx*dx + dy*dy);
  }

  async function loadHeat(){
    const res = await fetch(CSV_PATH);
    const text = await res.text();
    const parsed = Papa.parse(text, { header:true, dynamicTyping:true, skipEmptyLines:true });
    rows = parsed.data.filter(r => r.lat_deg && r.lon_deg);

    if (!rows.length){ alert("CSV vuoto o senza lat/lon"); return; }

    // Valore massimo grezzo per legenda
    const vals = rows.map(r => +r.oysters_total || 0);
    const maxVal = Math.max(1, ...vals);

    // normalizza oysters_total in [0..1] con minimo 0.05 per visibilità
    const pts = rows.map(r => {
      const w = Math.min(1, Math.max(0.05, (+r.oysters_total||0) / maxVal));
      return [ +r.lat_deg, +r.lon_deg, w ];
    });

    const latlngs = rows.map(r => [ +r.lat_deg, +r.lon_deg ]);
    map.fitBounds(latlngs, { padding:[20,20] });

    // heat con gradient coerente con la barra della legenda
    const heat = L.heatLayer(pts, {
      radius: RADIUS, blur: BLUR, maxZoom: 17,
      gradient: { 0.0:'#0000ff', 0.25:'#00ffff', 0.5:'#00ff00', 0.75:'#ffff00', 1.0:'#ff0000' }
    }).addTo(map);

    // aggiorna legenda
    document.getElementById('legMin').textContent = '0';
    document.getElementById('legMax').textContent = String(Math.round(maxVal));
    document.getElementById('legend').style.display = 'block';

    // hover: mostra lat/lon e nearest point entro 30 px
    map.on('mousemove', (ev)=>{
      const {lat, lng} = ev.latlng;
      document.getElementById('hoverLatLon').textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;

      const p = map.latLngToLayerPoint(ev.latlng);
      let best = null, bestD = 1e9;

      for (const r of rows){
        const ll = L.latLng(+r.lat_deg, +r.lon_deg);
        const q = map.latLngToLayerPoint(ll);
        const d = distPx(p, q);
        if (d < bestD){
          bestD = d; best = r;
        }
      }
      if (best && bestD <= 30){
        const n = +best.oysters_total || 0;
        document.getElementById('hoverNearest').textContent =
          `${n} oysters @ (${(+best.lat_deg).toFixed(6)}, ${(+best.lon_deg).toFixed(6)})`;
      } else {
        document.getElementById('hoverNearest').textContent = '–';
      }
    });
  }

  loadHeat();
</script>
</body>
</html>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path del CSV (es: outputs/.../final.csv)")
    ap.add_argument("--out", required=False, default=None, help="HTML di output (default: heatmap_<ts>.html)")
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--blur", type=int, default=15)
    args = ap.parse_args()

    csv_path = args.csv.lstrip('/\\')
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV non trovato: {csv_path}")

    out = args.out or os.path.join("outputs", f"heatmap_{int(time.time())}.html")

    csv_http = '/' + csv_path.replace("\\", "/")   
    csv_js = '"/' + csv_path.replace("\\", "/") + '"'
    html_out = TEMPLATE.replace("%CSV%", csv_js).replace("%RADIUS%", str(args.radius)).replace("%BLUR%", str(args.blur))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(out)

if __name__ == "__main__":
    main()
