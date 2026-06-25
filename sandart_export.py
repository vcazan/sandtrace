"""Assemble final .thr polar paths for the SandTrace pipeline."""

from __future__ import annotations

import math

import numpy as np

CENTER_START_RHO = 0.001
CENTER_START_THETA = -math.pi / 2
APPROACH_WEIGHT = 30


def remove_consecutive_duplicates(polar: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    if polar is None or len(polar) < 2:
        return polar
    out = [polar[0]]
    for row in polar[1:]:
        if not np.allclose(row, out[-1], atol=tol, rtol=0.0):
            out.append(row)
    return np.asarray(out, dtype=polar.dtype)


def remove_immediate_backtracks(polar: np.ndarray, q: int = 3) -> np.ndarray:
    """Collapse A→B→A spikes that only retrace fresh sand."""
    if polar is None or len(polar) < 3:
        return polar

    def qpt(row):
        theta, rho = float(row[0]), float(row[1])
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return (round(x, q), round(y, q))

    def undirected(a, b):
        if a == b:
            return None
        return (a, b) if a <= b else (b, a)

    out = [polar[0]]
    for row in polar[1:]:
        out.append(row)
        while len(out) >= 3:
            a, b, c = qpt(out[-3]), qpt(out[-2]), qpt(out[-1])
            e1 = undirected(a, b)
            e2 = undirected(b, c)
            if e1 is not None and e1 == e2:
                out.pop(-2)
            else:
                break
    return np.asarray(out, dtype=polar.dtype)


def compress_polar_path(polar: np.ndarray) -> np.ndarray:
    """Lightweight export cleanup — removes obvious redundant motion."""
    path = np.asarray(polar)
    path = remove_consecutive_duplicates(path)
    path = remove_immediate_backtracks(path)
    path = trim_duplicate_suffix_edges(path)
    return path


def trim_duplicate_suffix_edges(
    polar: np.ndarray,
    q: int = 3,
    window_size: int = 5,
    min_dups_in_window: int = 4,
) -> np.ndarray:
    if polar is None or len(polar) < 4:
        return polar

    def _qxy(row):
        theta = float(row[0])
        rho = float(row[1])
        x = rho * math.cos(theta)
        y = rho * math.sin(theta)
        return (round(x, q), round(y, q))

    pts = [_qxy(row) for row in polar]
    edges = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if a == b:
            edges.append(None)
        else:
            edges.append((a, b) if a <= b else (b, a))

    first_seen = {}
    for i, e in enumerate(edges):
        if e is None:
            continue
        if e not in first_seen:
            first_seen[e] = i

    is_dup = []
    for i, e in enumerate(edges):
        if e is None:
            is_dup.append(True)
        else:
            is_dup.append(first_seen.get(e, i) < i)

    cut = len(polar)
    for i in range(len(edges) - 1, -1, -1):
        if is_dup[i]:
            cut = i + 1
            continue
        ws = max(0, i - window_size + 1)
        win_len = i - ws + 1
        if win_len >= window_size:
            dup_count = sum(1 for j in range(ws, i + 1) if is_dup[j])
            if dup_count >= min_dups_in_window:
                cut = i + 1
                continue
        break

    return polar[:max(2, cut)]


def build_entry_path(polar_path: np.ndarray,
                     ball_at_center: bool) -> np.ndarray | None:
    if polar_path is None or len(polar_path) < 2:
        return None

    rhos = polar_path[:, 1]
    indices = np.arange(len(polar_path), dtype=float)

    if ball_at_center:
        costs = rhos * APPROACH_WEIGHT + indices
        start_rho = 0.0
    else:
        costs = (1.0 - rhos) * APPROACH_WEIGHT + indices
        start_rho = 1.0

    nearest_idx = int(np.argmin(costs))
    nearest_theta = float(polar_path[nearest_idx, 0])
    nearest_rho = float(polar_path[nearest_idx, 1])

    n_approach = max(5, int(abs(nearest_rho - start_rho) * APPROACH_WEIGHT))
    approach = np.column_stack([
        np.full(n_approach, nearest_theta),
        np.linspace(start_rho, nearest_rho, n_approach),
    ])

    if nearest_idx == 0:
        return approach

    trace = polar_path[nearest_idx::-1]
    return np.vstack([approach, trace[1:]])


def build_final_thr(
    result: dict,
    *,
    mirror: bool = False,
    ball_start: str = "center",
    table_orientation: bool = True,
) -> np.ndarray:
    polar = np.asarray(result.get("polar"))
    n_ol = int(result.get("n_outline", 0) or 0)
    navigate = result.get("navigate")

    if polar is None or len(polar) < 2:
        raise ValueError("Converted path is empty")

    orient = _orient_for_table if table_orientation else _orient_for_display

    n_ol = max(0, min(n_ol, len(polar)))
    polar_oriented = orient(polar.copy(), mirror)

    parts = []
    entry_l = ball_start.lower()
    if "none" not in entry_l:
        if navigate is not None and len(navigate) > 1:
            nav_oriented = orient(np.asarray(navigate).copy(), mirror)
            parts.append(nav_oriented)
        else:
            nav = build_entry_path(polar_oriented, ball_at_center="center" in entry_l)
            if nav is not None and len(nav) > 1:
                parts.append(nav)

    if n_ol > 0:
        parts.append(polar_oriented[:n_ol])
        full_tail = polar_oriented[n_ol:]
    else:
        full_tail = polar_oriented

    if len(full_tail) > 1:
        parts.append(full_tail)

    full = np.vstack(parts) if len(parts) > 1 else parts[0]
    if table_orientation and "center" in entry_l and len(full) > 0:
        anchor = np.array([CENTER_START_THETA, CENTER_START_RHO], dtype=full.dtype)
        if float(full[0, 1]) <= (CENTER_START_RHO * 4.0):
            full[0] = anchor
        else:
            full = np.vstack([anchor, full])
    return compress_polar_path(full)


def polar_to_cartesian_points(polar: np.ndarray) -> list[tuple[float, float]]:
    pts = []
    for theta, rho in polar:
        x = float(rho * math.cos(theta))
        y = float(rho * math.sin(theta))
        pts.append((x, y))
    return pts


def polar_to_svg_points(polar: np.ndarray, size: int = 400) -> list[list[float]]:
    r = size / 2 - 8
    cx = cy = size / 2
    pts: list[list[float]] = []
    for theta, rho in polar:
        x = cx + float(rho * math.cos(theta)) * r
        y = cy - float(rho * math.sin(theta)) * r
        pts.append([round(x, 2), round(y, 2)])
    return pts


def _orient_for_table(polar: np.ndarray, mirror: bool) -> np.ndarray:
    """Rotate into Oasis/Sisyphus table coordinates (+ optional mirror)."""
    out = np.asarray(polar).copy()
    if mirror:
        out[:, 0] = math.pi - out[:, 0]
    out[:, 0] -= math.pi / 2
    return out


def _orient_for_display(polar: np.ndarray, mirror: bool) -> np.ndarray:
    """Keep image-aligned orientation for on-screen preview."""
    out = np.asarray(polar).copy()
    if mirror:
        out[:, 0] = math.pi - out[:, 0]
    return out


# Back-compat alias
_orient_polar = _orient_for_table


def build_trace_segments(
    result: dict,
    *,
    mirror: bool = False,
    ball_start: str = "center",
    size: int = 400,
) -> list[dict]:
    """Playback segments in draw order with table orientation applied."""
    polar = np.asarray(result.get("polar"))
    n_ol = int(result.get("n_outline", 0) or 0)
    navigate = result.get("navigate")

    if polar is None or len(polar) < 2:
        return []

    n_ol = max(0, min(n_ol, len(polar)))
    polar_display = _orient_for_display(polar, mirror)
    segments: list[dict] = []
    entry_l = ball_start.lower()

    if "none" not in entry_l:
        if navigate is not None and len(navigate) > 1:
            nav = _orient_for_display(np.asarray(navigate), mirror)
            segments.append({"kind": "navigate", "points": polar_to_svg_points(nav, size)})
        else:
            nav = build_entry_path(polar_display, ball_at_center="center" in entry_l)
            if nav is not None and len(nav) > 1:
                segments.append({"kind": "navigate", "points": polar_to_svg_points(nav, size)})

    if n_ol > 0:
        outline = np.vstack([polar_display[:n_ol], polar_display[0:1]])
        segments.append({"kind": "outline", "points": polar_to_svg_points(outline, size)})

    full_tail = polar_display[n_ol:] if n_ol > 0 else polar_display
    if len(full_tail) > 1:
        segments.append({"kind": "draw", "points": polar_to_svg_points(full_tail, size)})

    if not segments:
        segments.append({"kind": "draw", "points": polar_to_svg_points(polar_display, size)})

    if "center" in entry_l and segments:
        first = segments[0]
        if first["kind"] == "navigate" and first["points"]:
            cx = cy = size / 2
            first["points"][0] = [round(cx, 2), round(cy, 2)]

    return [s for s in segments if len(s.get("points", [])) >= 2]


def polar_to_svg_path(polar: np.ndarray, size: int = 400) -> str:
    r = size / 2 - 8
    cx = cy = size / 2
    parts = []
    for i, (theta, rho) in enumerate(polar):
        x = cx + rho * math.cos(theta) * r
        y = cy - rho * math.sin(theta) * r
        parts.append(f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}")
    return " ".join(parts)


def thr_to_string(polar_points: np.ndarray, comment: str = "") -> str:
    """Serialize a polar path to .thr text (no file write)."""
    lines: list[str] = []
    if comment:
        lines.extend(f"# {line}" for line in comment.splitlines())
    for theta, rho in polar_points:
        lines.append(f"{theta:.6f} {rho:.6f}")
    return "\n".join(lines) + "\n"


def svg_to_string(
    polar_points: np.ndarray,
    size: int = 800,
    stroke_width: float = 1.0,
    comment: str = "",
) -> str:
    """Serialize a polar path to SVG text (no file write)."""
    r = size / 2 - 4
    cx = cy = size / 2
    parts = []
    for i, (theta, rho) in enumerate(polar_points):
        x = cx + rho * math.cos(theta) * r
        y = cy - rho * math.sin(theta) * r
        parts.append(f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}")
    out = []
    if comment:
        out.append(f"<!-- {comment} -->")
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    )
    out.append(
        f'  <circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="none" stroke="#b09070" stroke-width="0.5"/>'
    )
    out.append(
        f'  <path d="{" ".join(parts)}" fill="none" stroke="#5c3317" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def contours_to_svg_paths(contours: list, w: int, h: int, mirror: bool = False) -> list[str]:
    """Return SVG path data in image pixel coordinates (optionally mirrored horizontally)."""
    paths = []
    for contour in contours:
        pts = contour.astype(np.float32)
        if len(pts) < 2:
            continue
        parts = []
        for i, (x, y) in enumerate(pts):
            px = (w - x) if mirror else x
            parts.append(f"{'M' if i == 0 else 'L'}{px:.1f},{y:.1f}")
        paths.append(" ".join(parts))
    return paths
