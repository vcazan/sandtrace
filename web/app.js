const state = {
  jobId: null,
  currentFile: null,
  settings: defaultSettings(),
  autoSettings: null,
  modeOptions: null,
  selectedMode: null,
  previewTimer: null,
  contourTimer: null,
  pathRequestId: 0,
  pathReady: false,
  pathStale: false,
  pathLayers: null,
  view: "outline",
  outlineCount: 0,
  waypointCount: 0,
};

const trace = {
  segments: [],
  totalPoints: 0,
  index: 0,
  playing: false,
  rafId: null,
  lastTs: 0,
  speed: 4,
  pointsPerSec: 180,
};

const $ = (sel) => document.querySelector(sel);

const CANVAS_DEFS = `<defs>
  <radialGradient id="ball-grad" cx="38%" cy="34%" r="68%">
    <stop offset="0%" stop-color="#eef0f2"/>
    <stop offset="35%" stop-color="#b9bcc0"/>
    <stop offset="100%" stop-color="#4c4843"/>
  </radialGradient>
</defs>`;
const BASE_CANVAS = `${CANVAS_DEFS}<circle cx="200" cy="200" r="192" class="table-ring"/>`;

function defaultSettings() {
  return {
    mode: "edges",
    blur: 3,
    canny_low: 30,
    canny_high: 100,
    threshold: 128,
    invert: true,
    min_area: 10,
    min_length: 20,
    smooth: 2,
    thin: true,
    straighten: 0.9,
    max_dim: 800,
    max_points: 15000,
    fill: 1.0,
    ball_start: "center",
    mirror: false,
    add_home: false,
    centerline: false,
  };
}

function readSettingsFromUI() {
  return {
    ...state.settings,
    mode: $("#mode").value,
    blur: Number($("#blur").value),
    canny_low: Number($("#canny-low").value),
    canny_high: Number($("#canny-high").value),
    threshold: Number($("#threshold").value),
    smooth: Number($("#smooth").value),
    min_length: Number($("#min-length").value),
    max_points: Number($("#max-points").value),
    ball_start: $("#ball-start").value,
    mirror: $("#mirror").checked,
    thin: $("#thin").checked,
    add_home: $("#add-home").checked,
    centerline: $("#centerline").checked,
  };
}

function applySettingsToUI(settings) {
  $("#mode").value = settings.mode || "edges";
  $("#blur").value = settings.blur ?? 3;
  $("#canny-low").value = settings.canny_low ?? 30;
  $("#canny-high").value = settings.canny_high ?? 100;
  $("#threshold").value = settings.threshold ?? 128;
  $("#smooth").value = settings.smooth ?? 2;
  $("#min-length").value = settings.min_length ?? 20;
  $("#max-points").value = settings.max_points ?? 15000;
  $("#ball-start").value = settings.ball_start || "center";
  $("#mirror").checked = !!settings.mirror;
  $("#thin").checked = settings.thin !== false;
  $("#add-home").checked = !!settings.add_home;
  $("#centerline").checked = !!settings.centerline;
  syncRangeOutputs();
  syncModeFields();
  state.settings = { ...state.settings, ...settings };
}

function syncRangeOutputs() {
  $("#blur-val").textContent = $("#blur").value;
  $("#canny-low-val").textContent = $("#canny-low").value;
  $("#canny-high-val").textContent = $("#canny-high").value;
  $("#threshold-val").textContent = $("#threshold").value;
  $("#smooth-val").textContent = $("#smooth").value;
  $("#min-length-val").textContent = $("#min-length").value;
  $("#max-points-val").textContent = $("#max-points").value;
}

function syncModeFields() {
  const mode = $("#mode").value;
  const edges = mode === "edges";
  const threshold = mode === "threshold";
  document.querySelectorAll(".setting-edges").forEach((el) => {
    el.classList.toggle("hidden", !edges);
  });
  document.querySelectorAll(".setting-threshold").forEach((el) => {
    el.classList.toggle("hidden", !threshold);
  });
  // "Draw each line once" applies to line-based modes (edges + threshold)
  document.querySelectorAll(".setting-line").forEach((el) => {
    el.classList.toggle("hidden", !(edges || threshold));
  });
}

function showAutoPreset(settings) {
  const pill = $("#auto-preset");
  const labels = {
    silhouette: "Silhouette",
    edges: "Edge detection",
    threshold: "Threshold",
  };
  const label = labels[settings?.mode] || settings?.preset_label;
  if (!label) {
    pill.classList.add("hidden");
    pill.textContent = "";
    return;
  }
  pill.textContent = label;
  pill.classList.remove("hidden");
}

function renderContoursSvg(svg, paths, width, height, emptyText) {
  if (!svg) return;
  if (!paths.length) {
    svg.innerHTML = `<text x="200" y="200" text-anchor="middle" fill="#9a8570" font-size="13">${emptyText || "No contours"}</text>`;
    return;
  }
  const scale = 400 / Math.max(width, height, 1);
  const ox = (400 - width * scale) / 2;
  const oy = (400 - height * scale) / 2;
  svg.innerHTML = `<g transform="translate(${ox},${oy}) scale(${scale})">` +
    paths.map((d) => `<path class="contour" d="${d}"/>`).join("") +
    `</g>`;
}

function renderContours(paths, width, height) {
  renderContoursSvg($("#outline-canvas"), paths, width, height, "No outline — adjust settings");
}

function setView(view) {
  state.view = view;
  $("#stage").dataset.view = view;
  $("#stage-state-label").textContent = view === "path" ? "Sand path" : "Outline";
  renderCaption();
}

// Show the path once it's ready & current; otherwise show the live outline.
function syncView() {
  setView(state.pathReady && !state.pathStale ? "path" : "outline");
}

function setBuilding(on) {
  $("#stage").classList.toggle("is-building", on);
}

function renderCaption() {
  const cap = $("#stage-caption");
  if (state.view === "outline") {
    const n = state.outlineCount;
    cap.textContent = n ? `${n} line${n === 1 ? "" : "s"}` : "detecting…";
  } else if (state.pathReady) {
    animateNumber(cap, state.waypointCount, (v) => `${v.toLocaleString()} waypoints`);
  } else {
    cap.textContent = "building…";
  }
}

function showPathUI(hasPath) {
  $("#trace-controls").classList.toggle("hidden", !hasPath);
  $("#convert-btn").disabled = !hasPath;
}

function animateNumber(el, target, fmt) {
  const from = Number(el.dataset.val || 0);
  if (from === target) { el.textContent = fmt(target); return; }
  const start = performance.now();
  const dur = 500;
  const step = (now) => {
    const t = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = Math.round(from + (target - from) * eased);
    el.textContent = fmt(val);
    if (t < 1) requestAnimationFrame(step);
    else el.dataset.val = String(target);
  };
  requestAnimationFrame(step);
}

function renderModeCards(options, suggestedMode) {
  const grid = $("#mode-grid");
  grid.innerHTML = options.map((opt) => {
    const suggested = opt.mode === suggestedMode
      ? `<span class="mode-card-suggested">Suggested</span>` : "";
    const n = opt.contour_count;
    return `
      <button type="button" class="mode-card${opt.mode === state.selectedMode ? " selected" : ""}" data-mode="${opt.mode}">
        <div class="mode-card-preview">
          <svg viewBox="0 0 400 400" aria-hidden="true"></svg>
        </div>
        <div class="mode-card-body">
          ${suggested}
          <p class="mode-card-title">${opt.label}</p>
          <p class="mode-card-desc">${opt.description}</p>
          <p class="mode-card-meta">${n} line${n === 1 ? "" : "s"}</p>
        </div>
      </button>`;
  }).join("");

  options.forEach((opt) => {
    const card = grid.querySelector(`[data-mode="${opt.mode}"]`);
    const svg = card?.querySelector("svg");
    renderContoursSvg(svg, opt.contour_paths, opt.width, opt.height, "No contours");
    card?.addEventListener("click", () => selectMode(opt.mode));
  });
}

function selectMode(mode) {
  state.selectedMode = mode;
  document.querySelectorAll(".mode-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.mode === mode);
  });
  $("#mode-continue").disabled = false;
}

async function fetchModeCompare() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 120000);
  const post = () => {
    const form = new FormData();
    form.append("job_id", state.jobId);
    return fetch("/api/mode-compare", { method: "POST", body: form, signal: ctrl.signal });
  };
  try {
    let res = await post();
    if (res.status === 404 && state.currentFile) {
      await reestablishJob();
      res = await post();
    }
    if (!res.ok) throw new Error(await apiError(res));
    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("This is taking too long — try again");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function showModePicker(filename) {
  $("#empty-state").classList.add("hidden");
  $("#workspace").classList.add("hidden");
  $("#mode-picker").classList.remove("hidden");
  $("#mode-picker-filename").textContent = filename || "";
  $("#mode-picker-loading").classList.remove("hidden");
  $("#mode-grid").innerHTML = "";
  $("#mode-continue").disabled = true;
  state.selectedMode = null;

  try {
    const data = await fetchModeCompare();
    state.modeOptions = data.options;
    renderModeCards(data.options, data.suggested_mode);
    selectMode(data.suggested_mode || data.options[0]?.mode || "edges");
  } catch (err) {
    $("#mode-grid").innerHTML =
      `<div class="mode-error">
         <p>Couldn't build previews.</p>
         <p class="mode-error-detail">${err.message || "Something went wrong"}</p>
         <button type="button" class="primary-btn" id="mode-retry">Try again</button>
       </div>`;
    $("#mode-retry")?.addEventListener("click", () => {
      showModePicker(filename).catch(() => {});
    });
  } finally {
    $("#mode-picker-loading").classList.add("hidden");
  }
}

function settingsForMode(mode) {
  const opt = state.modeOptions?.find((o) => o.mode === mode);
  return opt?.settings || { ...defaultSettings(), mode };
}

async function enterWorkspaceFromMode(mode) {
  const settings = settingsForMode(mode);
  state.autoSettings = { ...settings };
  applySettingsToUI({ ...defaultSettings(), ...settings });
  showAutoPreset(settings);

  $("#mode-picker").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "auto" });

  state.pathReady = false;
  state.pathStale = true;
  state.pathLayers = null;
  renderPathLayers([]);
  $("#download-row").classList.add("hidden");
  syncView();

  await runContourPreview().catch((err) => setStatus(err.message));
  setStatus("Adjust the look — it draws the path automatically.");

  // Build the path; the view fades from outline to the finished drawing.
  runPathPreview().catch((err) => {
    setBuilding(false);
    setStatus(err.message);
  });
}

function resetToUpload() {
  state.jobId = null;
  state.modeOptions = null;
  state.selectedMode = null;
  state.pathReady = false;
  state.pathLayers = null;
  $("#mode-picker").classList.add("hidden");
  $("#workspace").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  setStatus("");
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 900px)").matches;
}

function syncTuningMode(open) {
  $("#workspace").classList.toggle("tuning", open && isMobileLayout());
}

function toggleAdvancedPanel(forceOpen) {
  const body = $("#settings-advanced");
  const btn = $("#toggle-advanced");
  const open = forceOpen ?? body.classList.contains("collapsed");
  body.classList.toggle("collapsed", !open);
  btn.textContent = open ? "Hide advanced options" : "Advanced options";
  btn.setAttribute("aria-expanded", String(open));
  syncTuningMode(open);
}

function backToModePicker() {
  stopTrace();
  const filename = $("#filename").textContent;
  showModePicker(filename).catch((e) => setStatus(e.message));
}

function setStatus(msg) {
  $("#status-line").textContent = msg || "";
  const err = $("#empty-error");
  if (msg && $("#workspace").classList.contains("hidden")) {
    err.textContent = msg;
    err.classList.remove("hidden");
  } else {
    err.classList.add("hidden");
  }
}

function isImageFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("image/")) return true;
  return /\.(png|jpe?g|webp|bmp|gif)$/i.test(file.name || "");
}

function apiError(res) {
  return res.json().then((body) => {
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((d) => d.msg || d).join(", ");
    return "Request failed";
  });
}

function layerClass(kind) {
  if (kind === "navigate") return "layer-navigate";
  if (kind === "outline") return "layer-outline";
  return "layer-draw";
}

function renderPathLayers(layers) {
  if (trace.playing) return;
  const svg = $("#path-canvas");
  const paths = (layers || [])
    .filter((layer) => layer.d && layer.d.trim())
    .map((layer) => `<path class="${layerClass(layer.kind)}" d="${layer.d}"/>`)
    .join("");
  svg.innerHTML = `${BASE_CANVAS}${paths}`;
}

function loadTrace(segments) {
  stopTrace();
  trace.segments = segments || [];
  trace.totalPoints = trace.segments.reduce(
    (sum, seg) => sum + (seg.points?.length || 0), 0
  );
  trace.index = 0;
  $("#trace-scrub").value = "0";
  const canPlay = trace.totalPoints >= 2;
  $("#trace-play").disabled = !canPlay;
  $("#trace-pause").disabled = true;
}

function pointAt(index) {
  let remaining = Math.max(0, index);
  for (const seg of trace.segments) {
    const pts = seg.points || [];
    if (remaining < pts.length && pts[remaining]) {
      return { x: pts[remaining][0], y: pts[remaining][1], kind: seg.kind };
    }
    remaining -= pts.length;
  }
  const lastSeg = trace.segments[trace.segments.length - 1];
  const lastPts = lastSeg?.points || [];
  const last = lastPts[lastPts.length - 1];
  return last ? { x: last[0], y: last[1], kind: lastSeg.kind } : null;
}

function drawTraceProgress(index) {
  const svg = $("#path-canvas");
  let remaining = index + 1;
  let parts = [BASE_CANVAS];

  for (const seg of trace.segments) {
    const pts = seg.points || [];
    const take = Math.min(Math.max(remaining, 0), pts.length);
    if (take >= 2) {
      const d = pts.slice(0, take)
        .map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`)
        .join(" ");
      parts.push(`<path class="${layerClass(seg.kind)}" d="${d}"/>`);
    }
    remaining -= pts.length;
    if (remaining <= 0) break;
  }

  const ball = pointAt(index);
  if (ball) {
    parts.push(`<circle class="trace-ball" cx="${ball.x}" cy="${ball.y}" r="6.5" fill="url(#ball-grad)"/>`);
  }
  svg.innerHTML = parts.join("");
}

function syncTraceScrub() {
  const max = Math.max(trace.totalPoints - 1, 0);
  $("#trace-scrub").value = String(max ? Math.round((trace.index / max) * 1000) : 0);
}

function stopTrace() {
  trace.playing = false;
  if (trace.rafId) cancelAnimationFrame(trace.rafId);
  trace.rafId = null;
  $("#trace-play").disabled = trace.totalPoints < 2;
  $("#trace-pause").disabled = true;
}

function startTrace() {
  if (trace.totalPoints < 2) return;
  if (trace.index >= trace.totalPoints - 1) trace.index = 0;
  trace.playing = true;
  trace.lastTs = 0; // initialized from the first rAF timestamp
  $("#trace-play").disabled = true;
  $("#trace-pause").disabled = false;
  trace.rafId = requestAnimationFrame(tickTrace);
}

function tickTrace(ts) {
  if (!trace.playing) return;
  // Use the rAF timestamp for both ends so we never mix time origins
  // (Safari/iOS rAF and performance.now can differ, which broke playback).
  if (!trace.lastTs) trace.lastTs = ts;
  const dt = Math.max(0, Math.min(100, ts - trace.lastTs));
  trace.lastTs = ts;
  trace.index = Math.max(0, Math.min(
    trace.totalPoints - 1,
    trace.index + (trace.pointsPerSec * trace.speed * dt) / 1000
  ));
  drawTraceProgress(Math.floor(trace.index));
  syncTraceScrub();
  if (trace.index >= trace.totalPoints - 1) {
    stopTrace();
    return;
  }
  trace.rafId = requestAnimationFrame(tickTrace);
}

function pauseTrace() {
  stopTrace();
}

function resetTrace() {
  stopTrace();
  trace.index = 0;
  syncTraceScrub();
  if (state.pathLayers) renderPathLayers(state.pathLayers);
}

function setTraceIndex(index) {
  trace.index = Math.max(0, Math.min(trace.totalPoints - 1, index));
  drawTraceProgress(Math.floor(trace.index));
  syncTraceScrub();
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  setStatus("Analyzing…");
  const res = await fetch("/api/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error(await apiError(res));
  return res.json();
}

// Re-create the server-side job from the image we still hold in the browser.
// The server keeps sessions in memory, so a restart or idle-expiry can drop a
// job; this silently re-establishes it so the user never sees a "session timeout".
async function reestablishJob() {
  if (!state.currentFile) throw new Error("Session expired — please re-upload");
  const data = await uploadFile(state.currentFile);
  state.jobId = data.job_id;
  return data.job_id;
}

// POST a job-scoped request; if the session is gone (404), re-establish once and retry.
async function jobPost(url, settingsObj) {
  const build = () => {
    const f = new FormData();
    f.append("job_id", state.jobId);
    if (settingsObj !== undefined) f.append("settings_json", JSON.stringify(settingsObj));
    return f;
  };
  let res = await fetch(url, { method: "POST", body: build() });
  if (res.status === 404 && state.currentFile) {
    await reestablishJob();
    res = await fetch(url, { method: "POST", body: build() });
  }
  if (!res.ok) throw new Error(await apiError(res));
  return res;
}

async function fetchContourPreview() {
  state.settings = readSettingsFromUI();
  const res = await jobPost("/api/preview", state.settings);
  return res.json();
}

async function fetchPathPreview(requestId, onProgress) {
  state.settings = readSettingsFromUI();
  const doFetch = () => {
    const f = new FormData();
    f.append("job_id", state.jobId);
    f.append("settings_json", JSON.stringify(state.settings));
    return fetch("/api/preview-path", { method: "POST", body: f });
  };
  let res = await doFetch();
  if (res.status === 404 && state.currentFile) {
    await reestablishJob();
    res = await doFetch();
  }
  if (!res.ok) throw new Error(await apiError(res));

  // Stream NDJSON: progress lines, then a final {"done": true, ...payload}.
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }
      if (msg.error) throw new Error(msg.error);
      if (msg.done) result = msg;
      else if (msg.progress !== undefined && onProgress) onProgress(msg.progress, msg.total);
    }
  }
  if (requestId !== state.pathRequestId) return null;
  return result;
}

function showPathProgress(on) {
  const el = $("#path-progress");
  if (!on) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.classList.add("indeterminate");
}

function setPathProgress(step, total) {
  const el = $("#path-progress");
  if (!total || total <= 0) { el.classList.add("indeterminate"); return; }
  el.classList.remove("indeterminate");
  const frac = Math.max(0, Math.min(1, step / total));
  const C = 201.06; // 2·π·32
  el.querySelector(".progress-ring-bar").style.strokeDashoffset = String(C * (1 - frac));
  $("#progress-pct").textContent = Math.round(frac * 100) + "%";
}

async function runContourPreview() {
  if (!state.jobId) return;
  const contourData = await fetchContourPreview();
  renderContours(contourData.contour_paths, contourData.width, contourData.height);
  state.outlineCount = contourData.contour_count;
  if (state.view === "outline") renderCaption();
  return contourData;
}

async function runPathPreview() {
  if (!state.jobId) return;
  const requestId = ++state.pathRequestId;
  setBuilding(true);

  // Designed progress loader appears only if the build runs longer than 3s.
  let shown = false;
  let last = null;
  const slowTimer = setTimeout(() => {
    shown = true;
    showPathProgress(true);
    if (last) setPathProgress(last.step, last.total);
  }, 3000);
  const onProgress = (step, total) => {
    last = { step, total };
    if (shown) setPathProgress(step, total);
  };

  try {
    const pathData = await fetchPathPreview(requestId, onProgress);
    if (!pathData) return;

    state.pathLayers = pathData.path_layers;
    renderPathLayers(pathData.path_layers);
    loadTrace(pathData.trace_segments);
    state.waypointCount = pathData.points;
    state.pathReady = true;
    state.pathStale = false;
    showPathUI(true);
    syncView();           // seamless: fade from outline to the finished path
    setStatus("Press play to watch it draw, then save.");
  } finally {
    clearTimeout(slowTimer);
    showPathProgress(false);
    if (requestId === state.pathRequestId) setBuilding(false);
  }
}

function schedulePreview() {
  stopTrace();
  clearTimeout(state.contourTimer);
  clearTimeout(state.previewTimer);
  state.pathReady = false;
  state.pathStale = true;
  $("#download-row").classList.add("hidden");
  $("#convert-btn").disabled = true;
  syncView();             // drop back to the instant outline while we rebuild

  state.contourTimer = setTimeout(() => {
    runContourPreview().catch((err) => setStatus(err.message));
  }, 90);

  state.previewTimer = setTimeout(() => {
    runPathPreview().catch((err) => {
      setBuilding(false);
      setStatus(err.message);
    });
  }, 500);
}

function downloadText(filename, text, mime) {
  const blob = new Blob([text], { type: mime });
  const objUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(objUrl), 2000);
}

async function saveThr() {
  if (!state.jobId) return;
  state.settings = readSettingsFromUI();
  const btn = $("#convert-btn");
  const spinner = btn.querySelector(".btn-spinner");
  btn.disabled = true;
  spinner.classList.remove("hidden");
  setStatus(state.pathReady ? "Saving your file…" : "Generating…");

  try {
    const res = await jobPost("/api/convert", state.settings);
    const data = await res.json();

    state.pathLayers = data.path_layers;
    state.pathReady = true;
    state.pathStale = false;
    state.waypointCount = data.points;
    renderPathLayers(data.path_layers);
    loadTrace(data.trace_segments);
    showPathUI(true);
    setView("path");

    const stem = $("#filename").textContent.replace(/\.[^.]+$/, "") || "sandtrace";
    state.lastStem = stem;
    state.lastSvgText = data.svg_text;

    // One click → the .thr lands in the browser's downloads, named after the image.
    downloadText(`${stem}.thr`, data.thr_text, "text/plain");

    // Keep the SVG available as a quiet secondary option.
    $("#download-row").classList.remove("hidden");

    btn.classList.add("saved");
    $("#convert-btn .btn-label").textContent = "Saved";
    setStatus(`Saved ${stem}.thr · ${data.points.toLocaleString()} waypoints`);
    setTimeout(() => {
      btn.classList.remove("saved");
      $("#convert-btn .btn-label").textContent = "Download";
    }, 2400);
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
  }
}

async function handleFile(file) {
  if (!file) return;
  if (!isImageFile(file)) {
    setStatus("Please choose a PNG, JPG, or WebP image.");
    return;
  }

  try {
    setStatus("Uploading…");
    state.currentFile = file;   // kept so we can transparently re-create the session
    const data = await uploadFile(file);
    state.jobId = data.job_id;

    $("#filename").textContent = data.filename;
    $("#original-image").src = `/api/image/${data.job_id}?t=${Date.now()}`;

    await showModePicker(data.filename);
    setStatus("Pick the detection mode that looks best.");
  } catch (err) {
    setStatus(err.message);
  }
}

// Events
const dropzone = $("#dropzone");
const fileInput = $("#file-input");

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  handleFile(file);
  e.target.value = "";
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

$("#mode-continue").addEventListener("click", () => {
  if (!state.selectedMode) return;
  enterWorkspaceFromMode(state.selectedMode).catch((e) => setStatus(e.message));
});

$("#mode-reupload").addEventListener("click", () => {
  resetToUpload();
  $("#file-input").click();
});

$("#back-to-modes").addEventListener("click", backToModePicker);

$("#mode").addEventListener("change", () => {
  syncModeFields();
  schedulePreview();
});

$("#toggle-advanced").addEventListener("click", () => toggleAdvancedPanel());

$("#reset-settings").addEventListener("click", () => {
  if (state.selectedMode) {
    applySettingsToUI({ ...defaultSettings(), ...settingsForMode(state.selectedMode) });
    showAutoPreset(settingsForMode(state.selectedMode));
  } else if (state.autoSettings) {
    applySettingsToUI({ ...defaultSettings(), ...state.autoSettings });
    showAutoPreset(state.autoSettings);
  } else {
    applySettingsToUI(defaultSettings());
    showAutoPreset(null);
  }
  schedulePreview();
});

["blur", "canny-low", "canny-high", "threshold", "smooth", "min-length", "max-points"].forEach((id) => {
  $(`#${id}`).addEventListener("input", () => {
    syncRangeOutputs();
    schedulePreview();
  });
});

["ball-start", "mirror", "thin", "add-home", "centerline"].forEach((id) => {
  $(`#${id}`).addEventListener("change", schedulePreview);
});

$("#convert-btn").addEventListener("click", () => saveThr().catch((e) => setStatus(e.message)));
$("#download-svg").addEventListener("click", () => {
  if (state.lastSvgText) {
    downloadText(`${state.lastStem || "sandtrace"}.svg`, state.lastSvgText, "image/svg+xml");
  }
});
$("#trace-play").addEventListener("click", startTrace);
$("#trace-pause").addEventListener("click", pauseTrace);
$("#trace-reset").addEventListener("click", resetTrace);
$("#trace-speed").addEventListener("input", (e) => { trace.speed = Number(e.target.value); });
$("#trace-scrub").addEventListener("input", (e) => {
  pauseTrace();
  const max = Math.max(trace.totalPoints - 1, 0);
  setTraceIndex(max ? Math.round((Number(e.target.value) / 1000) * max) : 0);
});

syncRangeOutputs();
syncModeFields();
showPathUI(false);
