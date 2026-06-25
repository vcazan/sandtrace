# SandTrace

Turn any image into a single continuous path for an **Oasis / Sisyphus kinetic
sand table**. Upload a photo, logo, or line art; SandTrace traces it and exports
a `.thr` file the table's ball can draw in one stroke — without unnecessary
"fresh sand" crossings.

> Live demo: **[sandtrace.ink](https://sandtrace.ink)**

![SandTrace](web/favicon.svg)

---

## What it does

1. **Upload an image** — photo, logo, or line drawing.
2. **Pick a tracing** — SandTrace generates three readings (Silhouette, Edge
   detection, Threshold) and you choose the one closest to what you want drawn.
3. **Refine** — tweak detail, smoothing, mirror, and more. The outline updates
   live and the full sand path renders right beside it.
4. **Download `.thr`** — one click saves the path file, ready to drop onto your
   table. (An SVG of the path is available too.)

Under the hood it detects contours with OpenCV, then plans a continuous path in
three phases — **navigate → outline → draw** — so the ball reaches the artwork
and fills it with a single, mostly-uninterrupted stroke.

---

## Quick start (run locally)

**Requirements:** Python 3.10+

```bash
# 1. Clone
git clone https://github.com/vcazan/sandtrace.git
cd sandtrace

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python server.py
```

Then open **http://localhost:8080** in your browser.

That's it — uploads and generated files are handled in a temp directory and
cleaned up automatically; nothing is written into the repo.

---

## Project structure

```
SandArt/
├── server.py            # FastAPI web server + API endpoints
├── SandArt.py           # Core engine: contour detection + path planning
├── sandart_export.py    # .thr / .svg assembly and preview helpers
├── requirements.txt
├── web/                 # Front-end (vanilla HTML/CSS/JS — no build step)
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── favicon.svg
└── Dockerfile           # Container image (used for deployment)
```

No front-end build step — the `web/` files are served as-is.

---

## How the API works

The browser talks to a small FastAPI backend:

| Endpoint | Purpose |
|----------|---------|
| `POST /api/analyze` | Upload an image, get a session + auto settings |
| `POST /api/mode-compare` | Build the three tracing previews |
| `POST /api/preview` | Fast contour/outline preview for the current settings |
| `POST /api/preview-path` | Full sand-path preview (with playback data) |
| `POST /api/convert` | Generate the final `.thr` / `.svg` (returned inline) |
| `GET  /api/health` | Health check |

Sessions live in memory with automatic expiry, and the `.thr`/`.svg` are
returned inline (never persisted on the server), so the app is stateless enough
to run anywhere.

---

## Using the `.thr` file

`.thr` files are the path format used by Sisyphus-style tables (theta/rho polar
coordinates). Once downloaded, add it to your table's playlist the same way you
would any other track (e.g. via the Sisyphus app or by copying it to the
device, depending on your hardware).

---

## Run with Docker (optional)

A `Dockerfile` is included if you'd rather run it in a container:

```bash
docker build -t sandtrace .
docker run -p 8080:8080 sandtrace
```

The only requirement to host it anywhere is exposing port `8080`.

---

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Image / geometry:** OpenCV (headless), NumPy, Pillow
- **Frontend:** vanilla HTML / CSS / JavaScript (no framework, no build)

---

## Contributing

Issues and pull requests are welcome. To hack on it, follow the Quick start
above — the server auto-reloads on file changes, and the front-end is plain
static files, so you can just refresh the browser.

---

## License

[MIT](LICENSE) © Vlad Cazan
