"""SandTrace web server — convert images to Oasis/Sisyphus .thr path files."""

from __future__ import annotations

import asyncio
import functools
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import SandArt as core
from sandart_export import (
    build_final_thr,
    build_trace_segments,
    contours_to_svg_paths,
    polar_to_svg_path,
    thr_to_string,
    svg_to_string,
    _orient_for_display,
)

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"

# Uploads are transient working files only — kept on the (ephemeral) temp disk and
# auto-expired. Generated .thr/.svg are never written to disk; they are returned
# inline and downloaded straight to the user's device.
UPLOAD_DIR = Path(os.environ.get("SANDTRACE_TMP", Path(tempfile.gettempdir()) / "sandtrace"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# Storage hygiene: bound memory + disk so nothing accumulates forever.
# (The client also transparently re-establishes a session if its job is gone,
# so these limits never surface as a user-visible "session expired".)
JOB_TTL_SECONDS = 45 * 60       # drop a job 45 min after its last use
MAX_JOBS = 60                   # hard cap; evict least-recently-used beyond this
CLEANUP_INTERVAL = 3 * 60       # janitor sweep cadence

app = FastAPI(title="SandTrace", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cap concurrency: image processing is memory-heavy and os.cpu_count() on a
# shared VM reports the host's cores (can be 16+), which would let too many
# heavy jobs run at once and OOM the machine. A small fixed pool bounds RAM.
_MAX_WORKERS = int(os.environ.get("SANDTRACE_WORKERS", "2"))
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

# Cap the stored image size so every downstream op (and memory use) is bounded —
# phone photos are ~4000px which is needless work on a shared CPU.
MAX_IMAGE_SIDE = int(os.environ.get("SANDTRACE_MAX_SIDE", "1600"))


async def _offload(fn, *args, **kwargs):
    """Run blocking CPU work in the thread pool so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, functools.partial(fn, *args, **kwargs))
_jobs: dict[str, dict] = {}


def _remove_job(job_id: str) -> None:
    job = _jobs.pop(job_id, None)
    if not job:
        return
    path = job.get("image_path")
    if path:
        Path(path).unlink(missing_ok=True)


def _touch(job: dict) -> None:
    job["last_seen"] = time.time()


def _get_job(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    if job:
        _touch(job)
    return job


def _register_job(job_id: str, job: dict) -> None:
    job["created_at"] = job["last_seen"] = time.time()
    _jobs[job_id] = job
    # Evict least-recently-used jobs beyond the cap.
    while len(_jobs) > MAX_JOBS:
        oldest = min(_jobs, key=lambda k: _jobs[k].get("last_seen", 0))
        if oldest == job_id:
            break
        _remove_job(oldest)


def _sweep() -> None:
    now = time.time()
    stale = [
        jid for jid, j in list(_jobs.items())
        if now - j.get("last_seen", j.get("created_at", 0)) > JOB_TTL_SECONDS
    ]
    for jid in stale:
        _remove_job(jid)
    # Remove orphaned upload files (e.g. left by a crash/restart) past their TTL.
    live = set(_jobs)
    for f in UPLOAD_DIR.glob("*"):
        try:
            if f.stem not in live and (now - f.stat().st_mtime) > JOB_TTL_SECONDS:
                f.unlink(missing_ok=True)
        except OSError:
            pass


async def _janitor() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            _sweep()
        except Exception:
            pass


@app.on_event("startup")
async def _start_janitor() -> None:
    asyncio.create_task(_janitor())


@app.middleware("http")
async def _no_cache_assets(request, call_next):
    """Force the browser to revalidate the app shell so it never runs a stale build."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


class ConvertSettings(BaseModel):
    mode: str = "edges"
    blur: int = Field(default=3, ge=1, le=15)
    canny_low: int = Field(default=30, ge=5, le=250)
    canny_high: int = Field(default=100, ge=10, le=300)
    threshold: int = Field(default=128, ge=0, le=255)
    invert: bool = True
    min_area: float = Field(default=10.0, ge=1.0)
    min_length: float = Field(default=20.0, ge=1.0)
    smooth: int = Field(default=2, ge=0, le=5)
    thin: bool = True
    straighten: float = Field(default=0.90, ge=0.0, le=1.0)
    max_dim: int = Field(default=800, ge=400, le=1600)
    max_points: int = Field(default=15000, ge=2000, le=50000)
    fill: float = Field(default=1.0, ge=0.5, le=1.0)
    ball_start: str = "center"
    mirror: bool = False
    add_home: bool = False


def _save_upload(upload: UploadFile) -> tuple[str, Path]:
    ext = Path(upload.filename or "image.png").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    job_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    size = 0
    with dest.open("wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File too large (max 12 MB)")
            fh.write(chunk)
    _shrink_upload(dest)
    return job_id, dest


def _shrink_upload(path: Path, max_side: int = MAX_IMAGE_SIDE) -> None:
    """Downscale oversized uploads in place to keep processing fast and bounded."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return
        scale = max_side / longest
        resized = cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(str(path), resized)
    except Exception:
        # If anything goes wrong, leave the original; processing still works.
        pass


_PREVIEW_KEYS = {
    "mode", "blur", "canny_low", "canny_high", "threshold", "invert",
    "min_area", "min_length", "smooth", "thin", "straighten", "max_dim",
}

_CONVERT_KEYS = _PREVIEW_KEYS | {
    "max_points", "fill", "ball_start", "add_home",
}


def _preview_kwargs(settings: ConvertSettings) -> dict:
    data = settings.model_dump()
    return {k: data[k] for k in _PREVIEW_KEYS}


def _convert_kwargs(settings: ConvertSettings) -> dict:
    data = settings.model_dump(exclude={"mirror"})
    return {k: data[k] for k in _CONVERT_KEYS}


def _build_output(job: dict, settings: ConvertSettings) -> tuple[dict, np.ndarray, int]:
    """Run full pipeline; returns (raw result, final polar path, elapsed ms)."""
    started = time.perf_counter()
    kwargs = _convert_kwargs(settings)
    result = core.build_thr_path(job["image_path"], **kwargs)
    final = build_final_thr(
        result,
        mirror=settings.mirror,
        ball_start=settings.ball_start,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result, final, elapsed_ms


def _preview_layers_svg(result: dict, settings: ConvertSettings) -> list[dict]:
    """Tri-colour path preview matching the desktop app (nav / outline / draw)."""
    polar = np.asarray(result.get("polar"))
    if polar is None or len(polar) < 2:
        return []

    n_ol = max(0, min(int(result.get("n_outline", 0) or 0), len(polar)))
    navigate = result.get("navigate")
    layers: list[dict] = []

    def _orient(arr: np.ndarray) -> np.ndarray:
        return _orient_for_display(arr, settings.mirror)

    if navigate is not None and len(navigate) > 1 and "none" not in settings.ball_start.lower():
        layers.append({"kind": "navigate", "stroke": "#c45c26", "d": polar_to_svg_path(_orient(navigate))})

    if n_ol > 0:
        outline = np.vstack([polar[:n_ol], polar[0:1]])
        layers.append({"kind": "outline", "stroke": "#2b6cb0", "d": polar_to_svg_path(_orient(outline))})

    tail = polar[n_ol:] if n_ol > 0 else polar
    if len(tail) > 1:
        layers.append({"kind": "draw", "stroke": "#5c3317", "d": polar_to_svg_path(_orient(tail))})

    if not layers:
        layers.append({"kind": "path", "stroke": "#5c3317", "d": polar_to_svg_path(_orient(polar))})
    return layers


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "sandtrace"}


@app.get("/favicon.ico")
def favicon():
    svg = WEB_DIR / "favicon.svg"
    if svg.exists():
        return FileResponse(svg, media_type="image/svg+xml")
    raise HTTPException(404, "no favicon")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    job_id, image_path = _save_upload(file)
    started = time.perf_counter()
    auto = await _offload(core.compute_auto_settings, str(image_path), max_dim=800)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _register_job(job_id, {
        "image_path": str(image_path),
        "filename": file.filename or image_path.name,
    })
    return {
        "job_id": job_id,
        "filename": file.filename,
        "auto_settings": auto,
        "analyze_ms": elapsed_ms,
    }


MODE_DESCRIPTIONS = {
    "silhouette": "Solid shapes and photos",
    "edges": "Line art and fine detail",
    "threshold": "Logos and high contrast",
}


@app.post("/api/mode-compare")
async def mode_compare(job_id: str = Form(...)):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job — upload again")

    started = time.perf_counter()

    def _run():
        mode_data = core.compute_mode_settings(job["image_path"], max_dim=800)
        options = []
        for mode_id, settings in mode_data["modes"].items():
            cfg = ConvertSettings(**settings)
            kwargs = _preview_kwargs(cfg)
            _, _, raw, contours, w, h = core.extract_preview_data(
                job["image_path"], **kwargs
            )
            options.append({
                "mode": mode_id,
                "label": settings["preset_label"],
                "description": MODE_DESCRIPTIONS.get(mode_id, ""),
                "contour_count": len(contours),
                "raw_contour_count": len(raw),
                "contour_paths": contours_to_svg_paths(contours, w, h),
                "settings": settings,
                "width": w,
                "height": h,
            })
        return mode_data, options

    mode_data, options = await _offload(_run)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "job_id": job_id,
        "options": options,
        "suggested_mode": mode_data["suggested_mode"],
        "compare_ms": elapsed_ms,
    }


@app.post("/api/preview")
async def preview(
    job_id: str = Form(...),
    settings_json: str = Form("{}"),
):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job — upload again")
    settings = ConvertSettings.model_validate_json(settings_json)
    kwargs = _preview_kwargs(settings)
    started = time.perf_counter()
    gray, binary, raw, contours, w, h = await _offload(
        core.extract_preview_data, job["image_path"], **kwargs
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "job_id": job_id,
        "width": w,
        "height": h,
        "contour_count": len(contours),
        "raw_contour_count": len(raw),
        "contour_paths": contours_to_svg_paths(contours, w, h, mirror=settings.mirror),
        "preview_ms": elapsed_ms,
    }


@app.get("/api/image/{job_id}")
def get_image(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    path = Path(job["image_path"])
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.post("/api/preview-path")
async def preview_path(
    job_id: str = Form(...),
    settings_json: str = Form("{}"),
):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Session expired — please upload the image again")
    settings = ConvertSettings.model_validate_json(settings_json)

    def _run():
        try:
            result, final, elapsed_ms = _build_output(job, settings)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        display = build_final_thr(
            result,
            mirror=settings.mirror,
            ball_start=settings.ball_start,
            table_orientation=False,
        )
        layers = _preview_layers_svg(result, settings)
        payload = {
            "points": len(final),
            "n_outline": int(result.get("n_outline", 0) or 0),
            "path_layers": layers,
            "trace_segments": build_trace_segments(
                result, mirror=settings.mirror, ball_start=settings.ball_start
            ),
            "path_svg": polar_to_svg_path(display),
            "path_ms": elapsed_ms,
            "settings_key": settings.model_dump_json(),
        }
        # Cache ONLY what /api/convert needs to skip recompute. Do NOT keep the
        # heavy response payload (trace_segments, layers, svg) in memory — that
        # bloated each retained job and led to OOM under load.
        job["cached"] = {
            "settings_key": payload["settings_key"],
            "result": result,
            "final": final,
            "path_ms": elapsed_ms,
        }
        return payload

    loop = __import__("asyncio").get_event_loop()
    return await loop.run_in_executor(_executor, _run)


@app.post("/api/convert")
async def convert(
    job_id: str = Form(...),
    settings_json: str = Form("{}"),
):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Unknown job — upload again")
    settings = ConvertSettings.model_validate_json(settings_json)
    settings_key = settings.model_dump_json()

    def _run():
        cached = job.get("cached")
        if cached and cached.get("settings_key") == settings_key:
            result, final = cached["result"], cached["final"]
            elapsed_ms = cached["path_ms"]
        else:
            result, final, elapsed_ms = _build_output(job, settings)

        comment = (
            f"Generated by SandTrace\n"
            f"Source: {job['filename']}\n"
            f"Points: {len(final)}"
        )
        display = build_final_thr(
            result,
            mirror=settings.mirror,
            ball_start=settings.ball_start,
            table_orientation=False,
        )
        # No files on disk: the .thr/.svg content is returned inline and the
        # browser saves it directly to the user's device.
        return {
            "points": len(final),
            "n_outline": int(result.get("n_outline", 0) or 0),
            "path_layers": _preview_layers_svg(result, settings),
            "trace_segments": build_trace_segments(
                result, mirror=settings.mirror, ball_start=settings.ball_start
            ),
            "path_svg": polar_to_svg_path(display),
            "convert_ms": elapsed_ms,
            "thr_text": thr_to_string(final, comment=comment),
            "svg_text": svg_to_string(display, comment=comment),
        }

    payload = await asyncio.get_event_loop().run_in_executor(_executor, _run)
    return {"job_id": job_id, **payload}


@app.delete("/api/jobs/{job_id}")
def cleanup(job_id: str):
    if job_id not in _jobs:
        return {"deleted": False}
    _remove_job(job_id)
    return {"deleted": True}


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
