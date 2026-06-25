"""
SandArt.py — core path-planning engine
======================================
Standalone island-splice path planner for the Oasis / Sisyphus kinetic
sand table. It has no dependencies on the rest of the app and is imported
by ``server.py`` / ``sandart_export.py``.

Three independently-computed phases stitched into the final output:

  Phase 1  NAVIGATE  (red in GUI)
      Straight line from ball start -> nearest point on ANY already-drawn
      line (outline or full path).  Then BFS along those drawn lines to
      reach outline[0].  The ONLY fresh-sand crossing is that first straight
      segment.

  Phase 2  OUTLINE   (blue in GUI)
      The actual outermost contour arcs from the image, joined with minimal
      straight-line bridges where gaps exist.  These are real drawing lines,
      not an artificial exterior boundary.  Rotated so outline[0] matches
      full_path[0].

  Phase 3  FULL PATH (black in GUI)
      Island-splice drawing: outer ring used as trunk, every remaining
      contour spliced in via minimal fresh-sand bridges.

Stitching order in saved file:
    navigate  ends at outline[0]
    outline   starts and ends at outline[0]  (closed loop)
    full path starts at outline[0]
"""

from __future__ import annotations

import math
import heapq
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_CLOSED_THRESHOLD_PX = 15.0
_BRIDGE_SAMPLE_STEP_PX = 2.0
_BRIDGE_OFF_BOUNDARY_WEIGHT = 3.0
_BRIDGE_INWARD_WEIGHT = 4.0
_BRIDGE_INWARD_GRACE_PX = 6.0


# ===========================================================================
#  IMAGE PROCESSING
# ===========================================================================

def load_and_preprocess(image_path: str, max_dim: int = 800) -> np.ndarray:
    """Load image as grayscale <= max_dim. Transparent PNGs get a white background."""
    gray = None
    try:
        from PIL import Image as _PIL
        pil = _PIL.open(image_path).convert("RGBA")
        bg  = _PIL.new("RGBA", pil.size, (255, 255, 255, 255))
        bg.paste(pil, mask=pil.split()[3])
        gray = cv2.cvtColor(np.array(bg.convert("RGB")), cv2.COLOR_RGB2GRAY)
    except Exception:
        pass
    if gray is None:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    h, w  = gray.shape
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return gray


def detect_edges(gray: np.ndarray, blur: int = 3,
                 low: int = 50, high: int = 150) -> np.ndarray:
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)
    return cv2.Canny(gray, low, high)


def thin_edges(binary: np.ndarray) -> np.ndarray:
    """Collapse Canny double-edges to single centre lines."""
    if binary is None or binary.max() == 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.dilate(binary, kernel, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=1)
    try:
        return cv2.ximgproc.thinning(closed)
    except AttributeError:
        pass
    skeleton = np.zeros_like(closed)
    element  = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = closed.copy()
    while True:
        eroded   = cv2.erode(img, element)
        opened   = cv2.dilate(eroded, element)
        temp     = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img      = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def threshold_image(gray: np.ndarray, threshold: int = 128,
                    invert: bool = True) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold, 255, mode)
    return binary


def build_silhouette_binary(gray: np.ndarray) -> np.ndarray:
    """
    Collapse a busy photo into a solid subject mask using Otsu + morphology.
    Tries both polarities and keeps the stronger single-subject result.
    """
    h, w = gray.shape
    blur_k = max(3, int(min(h, w) * 0.008) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    total = float(h * w)
    close_k = max(5, int(min(h, w) * 0.022) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    open_k = max(3, (close_k // 2) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    best_mask = None
    best_score = -1.0
    for use_inv in (True, False):
        flag = cv2.THRESH_BINARY_INV if use_inv else cv2.THRESH_BINARY
        _, bw = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)
        raw, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not raw:
            continue
        areas = sorted([float(cv2.contourArea(c)) for c in raw], reverse=True)
        main_frac = areas[0] / total
        if main_frac < 0.025 or main_frac > 0.90:
            continue
        n_sig = sum(1 for a in areas if a > total * 0.004)
        score = main_frac * (1.25 if n_sig == 1 else 1.0 / max(n_sig, 1))
        if score > best_score:
            best_score = score
            best_mask = closed

    if best_mask is None:
        _, best_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return best_mask


def extract_silhouette_contours(binary: np.ndarray, img_w: int, img_h: int,
                                min_hole_frac: float = 0.035
                                ) -> tuple[list, np.ndarray | None]:
    """Return the outer silhouette boundary plus large interior holes only."""
    total = float(img_w * img_h)
    raw, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if not raw:
        return [], None
    hier = hierarchy[0].astype(np.int32) if hierarchy is not None else None

    outers: list[tuple[float, int]] = []
    for i, c in enumerate(raw):
        if hier is not None and int(hier[i][3]) >= 0:
            continue
        area = float(cv2.contourArea(c))
        if area >= total * 0.025:
            outers.append((area, i))
    if not outers:
        return [], None

    outers.sort(reverse=True)
    outer_area, outer_idx = outers[0]
    result: list[np.ndarray] = []
    keep: list[int] = [outer_idx]

    outer_pts = raw[outer_idx].reshape(-1, 2).astype(np.float32)
    if len(outer_pts) >= 3:
        result.append(outer_pts)

    if hier is not None:
        child = int(hier[outer_idx][2])
        while child >= 0:
            area = float(cv2.contourArea(raw[child]))
            if area >= outer_area * min_hole_frac:
                pts = raw[child].reshape(-1, 2).astype(np.float32)
                if len(pts) >= 3:
                    result.append(pts)
                    keep.append(child)
            child = int(hier[child][0])

    return result, _remap_hierarchy(hier, keep)


def extract_contours(binary: np.ndarray, min_area: float = 5.0) -> list:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_TC89_KCOS)
    result = []
    for c in contours:
        if cv2.contourArea(c) >= min_area or len(c) >= 2:
            pts = c.squeeze()
            if pts.ndim == 1:
                pts = pts[np.newaxis, :]
            result.append(pts.astype(np.float32))
    return result


def _remap_hierarchy(hierarchy: np.ndarray | None,
                     keep_indices: list[int]) -> np.ndarray | None:
    if hierarchy is None:
        return None
    if not keep_indices:
        return np.zeros((0, 4), dtype=np.int32)
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}
    out = np.full((len(keep_indices), 4), -1, dtype=np.int32)
    for new_i, old_i in enumerate(keep_indices):
        refs = hierarchy[old_i]
        for k in range(4):
            old_ref = int(refs[k])
            if old_ref >= 0 and old_ref in old_to_new:
                out[new_i, k] = old_to_new[old_ref]
    return out


def extract_contours_with_hierarchy(binary: np.ndarray, min_area: float = 5.0,
                                    retrieval_mode: int = cv2.RETR_TREE) -> tuple[list, np.ndarray | None]:
    contours, hierarchy = cv2.findContours(binary, retrieval_mode,
                                           cv2.CHAIN_APPROX_TC89_KCOS)
    hierarchy_flat = hierarchy[0].astype(np.int32) if hierarchy is not None else None
    result = []
    keep_indices: list[int] = []
    for i, c in enumerate(contours):
        if cv2.contourArea(c) >= min_area or len(c) >= 2:
            pts = c.squeeze()
            if pts.ndim == 1:
                pts = pts[np.newaxis, :]
            result.append(pts.astype(np.float32))
            keep_indices.append(i)
    return result, _remap_hierarchy(hierarchy_flat, keep_indices)


def filter_short_contours(contours: list, min_length: float = 0.0) -> list:
    if min_length <= 0:
        return contours
    result = []
    for pts in contours:
        if len(pts) < 2:
            continue
        perimeter = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if perimeter >= min_length:
            result.append(pts)
    return result


def filter_short_contours_with_hierarchy(contours: list,
                                         hierarchy: np.ndarray | None,
                                         min_length: float = 0.0) -> tuple[list, np.ndarray | None]:
    if min_length <= 0:
        return contours, hierarchy
    result = []
    keep_indices: list[int] = []
    for i, pts in enumerate(contours):
        if len(pts) < 2:
            continue
        perimeter = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if perimeter >= min_length:
            result.append(pts)
            keep_indices.append(i)
    return result, _remap_hierarchy(hierarchy, keep_indices)


def _chaikin_subdivide(pts: np.ndarray, iterations: int = 2,
                        closed: bool = False) -> np.ndarray:
    result = pts.copy()
    for _ in range(iterations):
        if len(result) < 2:
            break
        new_pts  = []
        n        = len(result)
        loop_end = n if closed else n - 1
        for i in range(loop_end):
            p0 = result[i]
            p1 = result[(i + 1) % n]
            new_pts.append(0.75 * p0 + 0.25 * p1)
            new_pts.append(0.25 * p0 + 0.75 * p1)
        if not closed:
            new_pts[0]  = pts[0].copy()
            new_pts[-1] = pts[-1].copy()
        result = np.array(new_pts, dtype=np.float32)
    return result


def _map_simp_to_orig_indices(orig: np.ndarray, simp: np.ndarray) -> list:
    """
    Map each simplified point to its nearest index in orig, enforcing
    monotone ordering so the mapping respects the contour traversal direction.
    approxPolyDP returns a subset of orig, so matches are typically exact.
    """
    indices      = []
    search_start = 0
    n_orig       = len(orig)
    for sp in simp:
        remaining = orig[search_start:]
        if len(remaining) == 0:
            indices.append(n_orig - 1)
            continue
        best_local  = int(np.argmin(np.linalg.norm(remaining - sp, axis=1)))
        matched_idx = search_start + best_local
        indices.append(matched_idx)
        search_start = matched_idx
    return indices


def _restore_smoothing_gaps(orig: np.ndarray, simp: np.ndarray,
                             closed: bool, gap_threshold: float) -> np.ndarray:
    """
    Re-insert original contour points wherever approxPolyDP created a gap
    larger than gap_threshold.  This prevents the simplification from
    producing a straight-line shortcut across a visually important curve.
    """
    n_orig = len(orig)
    if n_orig < 2 or len(simp) < 2:
        return simp

    orig_idx = _map_simp_to_orig_indices(orig, simp)

    parts   = []
    n_simp  = len(simp)
    for i in range(n_simp - 1):
        parts.append(simp[i:i + 1])
        if float(np.linalg.norm(simp[i + 1] - simp[i])) > gap_threshold:
            i0, i1 = orig_idx[i], orig_idx[i + 1]
            if i1 > i0 + 1:
                parts.append(orig[i0 + 1:i1])
    parts.append(simp[-1:])

    # Handle the wrap-around gap for closed contours
    if closed and n_simp >= 2:
        if float(np.linalg.norm(simp[0] - simp[-1])) > gap_threshold:
            i0, i1 = orig_idx[-1], orig_idx[0]
            wrap   = []
            if i0 + 1 < n_orig:
                wrap.append(orig[i0 + 1:])
            if i1 > 0:
                wrap.append(orig[:i1])
            if wrap:
                parts.append(np.vstack(wrap))

    return np.vstack(parts).astype(np.float32)


def smooth_contours(contours: list, strength: int = 5) -> list:
    if strength <= 0:
        return contours
    epsilon       = float(strength)
    chaikin_iters = 1 if strength <= 2 else (2 if strength <= 8 else 3)
    smoothed      = []
    for pts in contours:
        if len(pts) < 3:
            smoothed.append(pts)
            continue
        pts_f   = pts.astype(np.float32)
        closed  = float(np.linalg.norm(pts_f[0] - pts_f[-1])) < _CLOSED_THRESHOLD_PX
        pts_cv  = pts_f.reshape(-1, 1, 2)
        simp    = cv2.approxPolyDP(pts_cv, epsilon, closed=closed)
        simp    = simp.reshape(-1, 2).astype(np.float32)

        # Don't let approxPolyDP create gaps larger than those in the original.
        # Re-insert original points across any such gap before Chaikin smoothing.
        if len(pts_f) > 1:
            orig_gaps     = np.linalg.norm(np.diff(pts_f, axis=0), axis=1)
            orig_max_gap  = float(orig_gaps.max())
        else:
            orig_max_gap  = 0.0
        gap_threshold = max(orig_max_gap * 1.5, 25.0)
        simp = _restore_smoothing_gaps(pts_f, simp, closed, gap_threshold)

        if len(simp) >= 3:
            simp = _chaikin_subdivide(simp, iterations=chaikin_iters, closed=closed)
        if closed and len(simp) > 1:
            simp = np.vstack([simp, simp[0:1]])
        smoothed.append(simp)
    return smoothed


def straighten_contours(contours: list, tolerance: float = 0.90) -> list:
    if tolerance >= 1.0:
        return contours
    result = []
    for pts in contours:
        if len(pts) < 3:
            result.append(pts)
            continue
        if float(np.linalg.norm(pts[0] - pts[-1])) < 15.0 and len(pts) > 6:
            result.append(pts)
            continue
        arc_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if arc_len < 1.0:
            result.append(pts)
            continue
        if float(np.linalg.norm(pts[-1] - pts[0])) / arc_len >= tolerance:
            result.append(np.array([pts[0], pts[-1]], dtype=np.float32))
        else:
            result.append(pts)
    return result


# ===========================================================================
#  COORDINATE HELPERS
# ===========================================================================

def cartesian_to_polar(points_xy: np.ndarray) -> np.ndarray:
    x, y  = points_xy[:, 0], points_xy[:, 1]
    rho   = np.clip(np.sqrt(x ** 2 + y ** 2), 0.0, 1.0)
    theta = np.arctan2(y, x)
    return np.column_stack([theta, rho])


def unwrap_theta(polar: np.ndarray) -> np.ndarray:
    p       = polar.copy()
    p[:, 0] = np.unwrap(p[:, 0])
    return p


def normalise_points(points_px: np.ndarray, img_w: int, img_h: int,
                     scale_half: float = 0.0,
                     center: tuple = None) -> np.ndarray:
    cx, cy = center if center is not None else (img_w / 2.0, img_h / 2.0)
    if scale_half <= 0:
        scale_half = math.sqrt(img_w ** 2 + img_h ** 2) / 2.0
    x = (points_px[:, 0] - cx) / scale_half
    y = -(points_px[:, 1] - cy) / scale_half
    return np.column_stack([x, y])


def _filter_border_points(contours: list, img_w: int, img_h: int,
                           margin: int = 3) -> np.ndarray:
    all_pts  = np.vstack(contours)
    mask     = ((all_pts[:, 0] > margin) & (all_pts[:, 0] < img_w - margin) &
                (all_pts[:, 1] > margin) & (all_pts[:, 1] < img_h - margin))
    interior = all_pts[mask]
    return interior if len(interior) >= 4 else all_pts


def compute_contour_center(contours: list, img_w: int = 0,
                            img_h: int = 0) -> tuple:
    pts = (_filter_border_points(contours, img_w, img_h)
           if img_w > 0 and img_h > 0 else np.vstack(contours))
    (cx, cy), _ = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2).astype(np.float32))
    return float(cx), float(cy)


def compute_contour_scale(contours: list, img_w: int, img_h: int,
                           fill: float = 1.0, center: tuple = None) -> float:
    if center is None:
        center = compute_contour_center(contours, img_w, img_h)
    cx, cy = center
    pts    = _filter_border_points(contours, img_w, img_h)
    max_r  = float(np.max(np.sqrt((pts[:, 0] - cx) ** 2 +
                                   (pts[:, 1] - cy) ** 2)))
    if max_r < 1.0:
        max_r = max(img_w, img_h) / 2.0
    return max_r / fill


def contour_centroid_rho(contour_px: np.ndarray, img_w: int, img_h: int,
                          scale_half: float = 0.0) -> float:
    pt   = np.array([[float(np.mean(contour_px[:, 0])),
                      float(np.mean(contour_px[:, 1]))]])
    norm = normalise_points(pt, img_w, img_h, scale_half=scale_half)
    return float(np.sqrt(norm[0, 0] ** 2 + norm[0, 1] ** 2))


def compute_outer_silhouette(contours: list, img_w: int, img_h: int,
                              smooth_strength: int = 0) -> list:
    """Legacy helper kept for GUI compatibility."""
    if not contours:
        return []
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for c in contours:
        pts = c.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    result = []
    for c in ext:
        pts = c.squeeze().astype(np.float32)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        if len(pts) >= 4:
            result.append(pts)
    result.sort(
        key=lambda c: cv2.contourArea(c.reshape(-1, 1, 2).astype(np.int32)),
        reverse=True)
    if smooth_strength > 0:
        result = smooth_contours(result, strength=smooth_strength)
    return result


# ===========================================================================
#  PATH SIMPLIFICATION
# ===========================================================================

def simplify_path(points: np.ndarray, max_points: int) -> np.ndarray:
    n = len(points)
    if n <= max_points:
        return points
    pts_cv = points.astype(np.float32).reshape(-1, 1, 2)
    lo, hi = 0.0, 1.0
    best   = points
    for _ in range(30):
        mid        = (lo + hi) / 2.0
        simplified = cv2.approxPolyDP(pts_cv, mid, closed=False)
        if len(simplified) <= max_points:
            best = simplified.reshape(-1, 2)
            hi   = mid
        else:
            lo = mid
    if len(best) < max_points * 0.5 and n > max_points:
        idx  = np.round(np.linspace(0, n - 1, max_points)).astype(int)
        best = points[idx]
    return best


def simplify_segmented_path(segments: list, max_points: int) -> np.ndarray:
    cleaned = [s for s in segments if s is not None and len(s) > 0]
    if not cleaned:
        return np.zeros((0, 2), dtype=np.float32)
    total = sum(len(s) for s in cleaned)
    if total <= max_points:
        return np.vstack(cleaned).astype(np.float32)
    mins  = [1 if len(s) == 1 else 2 for s in cleaned]
    min_r = sum(mins)
    if max_points <= min_r:
        return np.vstack(
            [s[:1] if m == 1 else np.vstack([s[0:1], s[-1:]])
             for s, m in zip(cleaned, mins)]
        ).astype(np.float32)
    caps      = [max(0, len(s) - m) for s, m in zip(cleaned, mins)]
    extra     = max_points - min_r
    total_cap = sum(caps)
    alloc     = mins.copy()
    if total_cap > 0 and extra > 0:
        raw    = [extra * (c / total_cap) for c in caps]
        floors = [int(math.floor(v)) for v in raw]
        for i, f in enumerate(floors):
            alloc[i] += min(f, caps[i])
        used  = sum(min(f, caps[i]) for i, f in enumerate(floors))
        rem   = extra - used
        if rem > 0:
            order = sorted(range(len(cleaned)),
                           key=lambda i: raw[i] - floors[i], reverse=True)
            for i in order:
                if rem <= 0:
                    break
                if alloc[i] < len(cleaned[i]):
                    alloc[i] += 1
                    rem       -= 1
    parts = []
    for seg, keep in zip(cleaned, alloc):
        if keep >= len(seg):
            parts.append(seg)
        elif keep <= 1:
            parts.append(seg[:1])
        elif keep == 2:
            parts.append(np.vstack([seg[0:1], seg[-1:]]))
        else:
            parts.append(simplify_path(seg, keep))
    return np.vstack(parts).astype(np.float32)


# ===========================================================================
#  FILE I/O
# ===========================================================================

def write_thr(polar_points: np.ndarray, output_path: str,
              comment: str = "") -> None:
    with open(output_path, "w", newline="\n") as f:
        if comment:
            for line in comment.splitlines():
                f.write(f"# {line}\n")
        for theta, rho in polar_points:
            f.write(f"{theta:.6f} {rho:.6f}\n")
    print(f"Wrote {len(polar_points)} waypoints -> {output_path}")


def write_svg(polar_points: np.ndarray, output_path: str,
              size: int = 800, stroke_width: float = 1.0,
              comment: str = "") -> None:
    r  = size / 2 - 4
    cx = cy = size / 2
    parts = []
    for i, (theta, rho) in enumerate(polar_points):
        x = cx + rho * math.cos(theta) * r
        y = cy - rho * math.sin(theta) * r
        parts.append(f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}")
    with open(output_path, "w", newline="\n") as f:
        if comment:
            f.write(f"<!-- {comment} -->\n")
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">\n')
        f.write(f'  <circle cx="{cx}" cy="{cy}" r="{r}" '
                f'fill="none" stroke="#b09070" stroke-width="0.5"/>\n')
        f.write(f'  <path d="{" ".join(parts)}" fill="none" stroke="#5c3317" '
                f'stroke-width="{stroke_width}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>\n')
        f.write('</svg>\n')
    print(f"Wrote SVG ({len(polar_points)} points) -> {output_path}")


# ===========================================================================
#  ISLAND-SPLICE GEOMETRY  (full path -- unchanged from SandArt3)
# ===========================================================================

_NAV_GRAPH_MAX_POINTS = 2500


def _decimate_path(pts: np.ndarray, max_pts: int) -> np.ndarray:
    """Uniformly subsample a path, preserving endpoints."""
    pts = pts.astype(np.float32)
    if len(pts) <= max_pts:
        return pts
    step = max(1, len(pts) // max_pts)
    dec = pts[::step]
    if not np.allclose(dec[0], pts[0]):
        dec = np.vstack([pts[0:1], dec])
    if not np.allclose(dec[-1], pts[-1]):
        dec = np.vstack([dec, pts[-1:]])
    return dec


def _contour_bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _bbox_min_dist(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


def _arc_length(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(
        np.diff(pts.astype(np.float32), axis=0), axis=1)))


def _is_closed(pts: np.ndarray) -> bool:
    return (len(pts) > 4 and
            float(np.linalg.norm(pts[0].astype(np.float32) -
                                  pts[-1].astype(np.float32))) < _CLOSED_THRESHOLD_PX)


def _closest_pts_on_polyline_batch(polyline: np.ndarray, queries: np.ndarray):
    polyline = polyline.astype(np.float32)
    queries  = queries.astype(np.float32)
    N = len(queries)
    M = len(polyline) - 1
    if M <= 0:
        p0    = polyline[0:1]
        dists = np.linalg.norm(queries - p0, axis=1)
        return (np.zeros(N, int), np.zeros(N, float),
                np.broadcast_to(p0, (N, 2)).copy(), dists.astype(float))
    p1    = polyline[:-1].astype(np.float64)
    p2    = polyline[1:].astype(np.float64)
    d     = p2 - p1
    denom = np.einsum('ij,ij->i', d, d)
    q64   = queries.astype(np.float64)
    diff  = q64[:, np.newaxis, :] - p1[np.newaxis, :, :]
    numer = np.einsum('nmi,mi->nm', diff, d)
    safe  = np.where(denom > 1e-12, denom, 1.0)
    t     = np.where(denom[np.newaxis, :] > 1e-12,
                     numer / safe[np.newaxis, :], 0.0)
    t     = np.clip(t, 0.0, 1.0)
    closest = (p1[np.newaxis, :, :] +
               t[:, :, np.newaxis] * d[np.newaxis, :, :])
    delta   = q64[:, np.newaxis, :] - closest
    dists2  = np.einsum('nmi,nmi->nm', delta, delta)
    best    = np.argmin(dists2, axis=1)
    n_idx   = np.arange(N)
    return (best.astype(int),
            t[n_idx, best].astype(float),
            closest[n_idx, best].astype(np.float32),
            np.sqrt(dists2[n_idx, best]).astype(float))


def _find_closest_pair(path: np.ndarray, island: np.ndarray):
    path   = path.astype(np.float32)
    island = island.astype(np.float32)
    closed = _is_closed(island)
    if closed:
        core  = (island[:-1]
                 if float(np.linalg.norm(island[0] - island[-1])) < _CLOSED_THRESHOLD_PX
                 else island)
        candidates    = core
        cand_orig_idx = np.arange(len(core))
    else:
        candidates    = island[[0, -1]]
        cand_orig_idx = np.array([0, len(island) - 1])
    seg_idxs, ts, Ps, dists = _closest_pts_on_polyline_batch(path, candidates)
    best_c = int(np.argmin(dists))
    min_d  = float(dists[best_c])
    ties   = np.where(np.abs(dists - min_d) < 1e-6)[0]
    if len(ties) > 1:
        best_c = int(ties[np.argmax(candidates[ties, 1])])
    return (int(seg_idxs[best_c]), float(ts[best_c]),
            Ps[best_c].copy(), int(cand_orig_idx[best_c]),
            candidates[best_c].copy(), float(dists[best_c]))


def _build_island_traversal(island: np.ndarray, q_idx: int) -> np.ndarray:
    island = island.astype(np.float32)
    closed = _is_closed(island)
    if closed:
        core = (island[:-1]
                if float(np.linalg.norm(island[0] - island[-1])) < _CLOSED_THRESHOLD_PX
                else island.copy())
        q   = q_idx % len(core)
        rot = np.vstack([core[q:], core[:q]])
        return np.vstack([rot, rot[[0]]])
    forward = (island.copy()
               if (q_idx == 0 or q_idx < len(island) // 2)
               else island[::-1].copy())
    return np.vstack([forward, forward[-2::-1]])


def _splice_island(path_pts: list, island: np.ndarray) -> list:
    path_arr = np.array(path_pts, dtype=np.float32)
    island   = island.astype(np.float32)
    path_seg, path_t, P, q_idx, _Q, _dist = _find_closest_pair(path_arr, island)
    traversal = _build_island_traversal(island, q_idx)
    EPS = 1e-6
    n   = len(path_arr)
    if path_t > EPS and path_t < 1.0 - EPS:
        prefix = list(path_arr[:path_seg + 1]) + [P]
        suffix = list(path_arr[path_seg + 1:])
    elif path_t <= EPS:
        P      = path_arr[path_seg].copy()
        prefix = list(path_arr[:path_seg + 1])
        suffix = list(path_arr[path_seg + 1:])
    else:
        nxt    = min(path_seg + 1, n - 1)
        P      = path_arr[nxt].copy()
        prefix = list(path_arr[:nxt + 1])
        suffix = list(path_arr[nxt + 1:])
    return prefix + list(traversal) + [P.copy()] + suffix


# ===========================================================================
#  OUTER-RING IDENTIFICATION  (used for both outline and trunk)
# ===========================================================================

def _find_outer_ring_contours(contours: list, img_w: int, img_h: int,
                               angle_bins: int = 72,
                               coverage_target: float = 0.70,
                               hierarchy: np.ndarray | None = None) -> tuple:
    """
    Angular-coverage sweep: identify contours that collectively cover the
    outer boundary of the drawing.

    A contour's "angular coverage" is how many 5-degree bins around the image
    centre its points span.  Contours covering the most distinct angular bins
    are selected greedily until coverage_target of all bins is filled.

    Returns (outer_list, inner_list).
    """
    if not contours:
        return [], []
    candidate_indices = list(range(len(contours)))
    if hierarchy is not None and len(hierarchy) == len(contours):
        # Prefer root-level contours so interior/detail children are de-prioritized.
        root_indices = [i for i in range(len(contours)) if int(hierarchy[i][3]) < 0]
        if root_indices:
            candidate_indices = root_indices

    # Filter out image-boundary artifacts: contours where the majority of points
    # hug the image edge (within edge_margin px) are frame-detection noise, not
    # real drawing content.  This happens when artwork bleeds to the PNG border.
    _EDGE_MARGIN   = max(15, int(min(img_w, img_h) * 0.025))
    _EDGE_FRAC_MAX = 0.50   # exclude if >50% of points are near the image edge
    def _is_edge_artifact(c):
        near = np.sum(
            (c[:, 0] < _EDGE_MARGIN) | (c[:, 0] > img_w - _EDGE_MARGIN) |
            (c[:, 1] < _EDGE_MARGIN) | (c[:, 1] > img_h - _EDGE_MARGIN)
        )
        return (near / len(c)) > _EDGE_FRAC_MAX

    candidate_indices = [i for i in candidate_indices
                         if not _is_edge_artifact(contours[i])]
    # Fall back to all non-artifact contours if the filtered list is empty.
    if not candidate_indices:
        candidate_indices = [i for i in range(len(contours))
                             if not _is_edge_artifact(contours[i])]

    cx, cy   = img_w / 2.0, img_h / 2.0
    bin_sets = []
    max_rhos = []
    # Interpolate along segments so heavily-straightened contours still get
    # credit for the full angular span of each long edge, not just vertices.
    _INTERP_STEP = 8.0  # px between interpolated samples
    for i in candidate_indices:
        c = contours[i]
        bins: set = set()
        for pt in c:
            a = math.atan2(float(pt[1]) - cy, float(pt[0]) - cx)
            bins.add(int((a + math.pi) / (2.0 * math.pi) * angle_bins) % angle_bins)
        for j in range(len(c) - 1):
            seg_len = float(np.linalg.norm(c[j + 1] - c[j]))
            if seg_len > _INTERP_STEP:
                n = max(2, int(seg_len / _INTERP_STEP))
                p0, p1 = c[j].astype(np.float64), c[j + 1].astype(np.float64)
                for k in range(1, n):
                    pt = p0 + (k / n) * (p1 - p0)
                    a  = math.atan2(float(pt[1]) - cy, float(pt[0]) - cx)
                    bins.add(int((a + math.pi) / (2.0 * math.pi) * angle_bins) % angle_bins)
        bin_sets.append(bins)
        max_rhos.append(float(np.max(np.sqrt((c[:, 0] - cx) ** 2 +
                                              (c[:, 1] - cy) ** 2))))
    # Sort by radius first so outer-edge contours always win over inner contours
    # that happen to have more vertices (e.g. after straighten_contours collapses them).
    order_local = sorted(range(len(candidate_indices)),
                   key=lambda i: (max_rhos[i], len(bin_sets[i])), reverse=True)
    target_bins   = max(1, int(coverage_target * angle_bins))
    outer_indices = set()
    covered       = set()
    for local_idx in order_local:
        idx = candidate_indices[local_idx]
        if len(covered) >= target_bins:
            break
        new = bin_sets[local_idx] - covered
        if new or not outer_indices:
            covered      |= bin_sets[local_idx]
            outer_indices.add(idx)
    outer = [contours[i].astype(np.float32) for i in outer_indices]
    inner = [contours[i].astype(np.float32)
             for i in range(len(contours)) if i not in outer_indices]
    return outer, inner


# ===========================================================================
#  OUTLINE BUILDER  (outermost actual contour arcs + straight-line bridges)
# ===========================================================================

def _join_silhouette_to_loop(pieces: list, center=None,
                             boundary_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Join a list of outer-ring contour arcs into a single closed loop.

    The pieces ARE the real drawn lines from the image.  Connections between
    them use straight-line bridges only -- the shortest possible gap-spanning
    mark across untouched sand.

    Algorithm:
    1. Build the loop greedily from the current tail, evaluating all unused
       pieces in both orientations.
    2. Score each bridge by distance plus penalties for going away from known
       boundary pixels and diving inward toward the image center.
    3. Use the minimum-cost bridge each step, then close the final gap.
    """
    if not pieces:
        return np.zeros((0, 2), dtype=np.float32)
    if len(pieces) == 1:
        loop = pieces[0].astype(np.float32)
        if float(np.linalg.norm(loop[-1] - loop[0])) > 1.0:
            loop = np.vstack([loop, loop[[0]]])
        return loop

    if center is not None:
        cx, cy = float(center[0]), float(center[1])
    else:
        all_pts = np.vstack([p.astype(np.float32) for p in pieces])
        cx      = float(np.mean(all_pts[:, 0]))
        cy      = float(np.mean(all_pts[:, 1]))

    boundary_band = None
    if boundary_mask is not None and boundary_mask.size > 0:
        boundary_u8 = (boundary_mask > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boundary_band = cv2.dilate(boundary_u8, k, iterations=1)

    pieces_f = [p.astype(np.float32) for p in pieces]

    def _sample_line_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist = float(np.linalg.norm(b - a))
        n = max(2, int(dist / _BRIDGE_SAMPLE_STEP_PX) + 1)
        return np.column_stack([
            np.linspace(float(a[0]), float(b[0]), n),
            np.linspace(float(a[1]), float(b[1]), n),
        ]).astype(np.float32)

    def _bridge_cost(a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(b - a))
        if dist <= 1e-6:
            return 0.0
        pts = _sample_line_points(a, b)
        cost = dist

        # Penalize bridges that leave the observed boundary band.
        if boundary_band is not None:
            h, w = boundary_band.shape[:2]
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            on_boundary = (boundary_band[yi, xi] > 0)
            off_ratio = 1.0 - float(np.mean(on_boundary.astype(np.float32)))
            cost += off_ratio * dist * _BRIDGE_OFF_BOUNDARY_WEIGHT

        # Penalize inward dives so the planner avoids large interior chords.
        radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        endpoint_r = min(float(np.linalg.norm(a - np.array([cx, cy], dtype=np.float32))),
                         float(np.linalg.norm(b - np.array([cx, cy], dtype=np.float32))))
        inward_depth = max(0.0, endpoint_r - float(np.min(radii)) - _BRIDGE_INWARD_GRACE_PX)
        if inward_depth > 0.0:
            cost += inward_depth * _BRIDGE_INWARD_WEIGHT
        return cost

    def _build_from(start_idx: int, start_reversed: bool) -> tuple[np.ndarray, float]:
        start_piece = pieces_f[start_idx][::-1].copy() if start_reversed else pieces_f[start_idx].copy()
        loop = start_piece
        used = {start_idx}
        total_cost = 0.0
        while len(used) < len(pieces_f):
            tail = loop[-1]
            best_idx = -1
            best_rev = False
            best_cost = float("inf")
            for i, piece in enumerate(pieces_f):
                if i in used:
                    continue
                c_fwd = _bridge_cost(tail, piece[0])
                if c_fwd < best_cost:
                    best_cost = c_fwd
                    best_idx = i
                    best_rev = False
                c_rev = _bridge_cost(tail, piece[-1])
                if c_rev < best_cost:
                    best_cost = c_rev
                    best_idx = i
                    best_rev = True
            piece = pieces_f[best_idx]
            if best_rev:
                piece = piece[::-1].copy()
            loop = np.vstack([loop, piece])
            total_cost += best_cost
            used.add(best_idx)
        total_cost += _bridge_cost(loop[-1], loop[0])
        if float(np.linalg.norm(loop[-1] - loop[0])) > 1.0:
            loop = np.vstack([loop, loop[[0]]])
        return loop, total_cost

    loop_a, cost_a = _build_from(0, False)
    loop_b, cost_b = _build_from(0, True)
    loop = loop_a if cost_a <= cost_b else loop_b

    # Close the loop: straight bridge from last point back to first
    if float(np.linalg.norm(loop[-1] - loop[0])) > 1.0:
        loop = np.vstack([loop, loop[[0]]])
    return loop


def _close_polyline(points: np.ndarray) -> np.ndarray:
    if points is None or len(points) < 2:
        return points
    if float(np.linalg.norm(points[-1] - points[0])) > 1.0:
        return np.vstack([points, points[0:1]])
    return points


def _extract_external_outline_from_mask(boundary_mask: np.ndarray | None) -> np.ndarray | None:
    if boundary_mask is None or boundary_mask.size == 0:
        return None
    mask_u8 = (boundary_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    ext, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ext:
        return None
    c = max(ext, key=lambda x: cv2.contourArea(x))
    pts = c.squeeze().astype(np.float32)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    return _close_polyline(pts)


def _build_outline_px(contours_px: list, img_w: int, img_h: int,
                      boundary_mask: np.ndarray | None = None,
                      contour_hierarchy: np.ndarray | None = None) -> np.ndarray | None:
    """Build blue outline from the external contour of the original mask."""
    del contours_px, img_w, img_h
    del contour_hierarchy
    return _extract_external_outline_from_mask(boundary_mask)


def _align_outline_to_start(outline_px: np.ndarray,
                              reference_px: np.ndarray) -> np.ndarray:
    """
    Rotate the closed outline loop so it starts at the vertex nearest to
    reference_px (= full_path_px[0]).

    This makes the outline -> full-path junction seamless: both phases start
    from the exact same pixel location.
    """
    if (len(outline_px) > 1 and
            float(np.linalg.norm(outline_px[0] - outline_px[-1])) < 2.0):
        core = outline_px[:-1]
    else:
        core = outline_px.copy()
    if len(core) == 0:
        return outline_px
    dists   = np.linalg.norm(core - reference_px.astype(np.float32), axis=1)
    best_i  = int(np.argmin(dists))
    rotated = np.vstack([core[best_i:], core[:best_i]])
    return np.vstack([rotated, rotated[[0]]])   # re-close


# ===========================================================================
#  DRAWN NETWORK  (BFS along drawn lines -- used by navigate path)
# ===========================================================================

class _DrawnNetwork:
    """
    Coarse grid of drawn pixels with 8-directional BFS pathfinding.
    Used to navigate along already-drawn lines without crossing fresh sand.
    """

    def __init__(self, img_w: int, img_h: int, complexity_pct: int = 70):
        self.cell_size = max(2, int(18 - complexity_pct * 0.16))
        self.gw        = (img_w  + self.cell_size - 1) // self.cell_size
        self.gh        = (img_h  + self.cell_size - 1) // self.cell_size
        self.grid      = np.zeros((self.gh, self.gw), dtype=bool)
        self.img_w     = img_w
        self.img_h     = img_h
        self.max_bfs   = int(5_000 + (complexity_pct / 100.0) ** 2 * 2_000_000)
        self._pad      = max(2, 5 // self.cell_size + 1)
        self._drawn_points: list = []

    def _mark_cell(self, gx: int, gy: int):
        for dx in range(-self._pad, self._pad + 1):
            for dy in range(-self._pad, self._pad + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.gw and 0 <= ny < self.gh:
                    self.grid[ny, nx] = True

    def _mark_line_dda(self, p1: np.ndarray, p2: np.ndarray):
        cs = self.cell_size
        gx1, gy1 = int(p1[0]) // cs, int(p1[1]) // cs
        gx2, gy2 = int(p2[0]) // cs, int(p2[1]) // cs
        steps    = max(abs(gx2 - gx1), abs(gy2 - gy1), 1)
        for s in range(steps + 1):
            t  = s / steps
            gx = max(0, min(int(round(gx1 + t * (gx2 - gx1))), self.gw - 1))
            gy = max(0, min(int(round(gy1 + t * (gy2 - gy1))), self.gh - 1))
            self._mark_cell(gx, gy)

    def mark_segment(self, points_px: np.ndarray):
        cs = self.cell_size
        for pt in points_px:
            gx = max(0, min(int(pt[0]) // cs, self.gw - 1))
            gy = max(0, min(int(pt[1]) // cs, self.gh - 1))
            self._mark_cell(gx, gy)
        for i in range(len(points_px) - 1):
            self._mark_line_dda(points_px[i], points_px[i + 1])
        self._drawn_points.append(points_px.copy())

    def nearest_drawn_point(self, target_px: np.ndarray) -> np.ndarray | None:
        if not self._drawn_points:
            return None
        all_pts = np.vstack(self._drawn_points)
        dists   = np.linalg.norm(all_pts - target_px, axis=1)
        return all_pts[int(np.argmin(dists))].copy()

    def find_path(self, start_px: np.ndarray,
                  end_px: np.ndarray) -> np.ndarray | None:
        """
        BFS on the drawn-cell grid from start_px to end_px.
        Returns Nx2 float32 pixel path, or None if no path found.
        """
        cs  = self.cell_size
        sgx = max(0, min(int(start_px[0]) // cs, self.gw - 1))
        sgy = max(0, min(int(start_px[1]) // cs, self.gh - 1))
        egx = max(0, min(int(end_px[0])   // cs, self.gw - 1))
        egy = max(0, min(int(end_px[1])   // cs, self.gh - 1))
        if sgx == egx and sgy == egy:
            return np.array([start_px, end_px], dtype=np.float32)
        self._mark_cell(sgx, sgy)
        self._mark_cell(egx, egy)
        visited = {(sgx, sgy): None}
        queue   = deque([(sgx, sgy)])
        found   = False
        nodes   = 0
        DIRS    = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1)]
        while queue and nodes < self.max_bfs:
            x, y   = queue.popleft()
            nodes += 1
            if x == egx and y == egy:
                found = True
                break
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.gw and 0 <= ny < self.gh
                        and (nx, ny) not in visited
                        and self.grid[ny, nx]):
                    visited[(nx, ny)] = (x, y)
                    queue.append((nx, ny))
        if not found:
            return None
        path_g = []
        cur    = (egx, egy)
        while cur is not None:
            path_g.append(cur)
            cur = visited[cur]
        path_g.reverse()
        result   = [start_px.copy()]
        interior = path_g[1:-1]
        if interior and self._drawn_points:
            all_drawn  = np.vstack(self._drawn_points)
            snap_limit = cs * 2.0
            for gx, gy in interior:
                cell_c = np.array([(gx + .5) * cs, (gy + .5) * cs],
                                   dtype=np.float32)
                dists  = np.linalg.norm(all_drawn - cell_c, axis=1)
                min_i  = int(np.argmin(dists))
                result.append(all_drawn[min_i].copy()
                               if dists[min_i] <= snap_limit else cell_c)
        else:
            for gx, gy in interior:
                result.append(np.array([(gx + .5) * cs, (gy + .5) * cs],
                                        dtype=np.float32))
        result.append(end_px.copy())
        cleaned = [result[0]]
        for pt in result[1:]:
            if not np.array_equal(pt, cleaned[-1]):
                cleaned.append(pt)
        if len(cleaned) < 2:
            cleaned = [result[0], result[-1]]
        return np.array(cleaned, dtype=np.float32)


# ===========================================================================
#  NAVIGATE PATH  (Phase 1 -- BFS from ball start to outline[0])
# ===========================================================================

def _nearest_point_index(points: np.ndarray, query: np.ndarray) -> int:
    d = np.linalg.norm(points - query.reshape(1, 2).astype(np.float32), axis=1)
    return int(np.argmin(d))


def _build_drawn_polyline_graph(polylines: list[np.ndarray],
                                merge_tol_px: float = 1.5) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    """Build an undirected weighted graph from polyline vertices and segments."""
    nodes_list: list[np.ndarray] = []
    adjacency: list[list[tuple[int, float]]] = []

    def _add_node(pt: np.ndarray) -> int:
        nodes_list.append(pt.astype(np.float32).copy())
        adjacency.append([])
        return len(nodes_list) - 1

    def _add_edge(i: int, j: int):
        if i == j:
            return
        w = float(np.linalg.norm(nodes[i] - nodes[j]))
        if w <= 1e-6:
            return
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))

    poly_node_ids: list[list[int]] = []
    for poly in polylines:
        if poly is None or len(poly) < 2:
            continue
        ids = [_add_node(pt) for pt in poly.astype(np.float32)]
        poly_node_ids.append(ids)

    if not nodes_list:
        return np.zeros((0, 2), dtype=np.float32), []

    nodes = np.vstack(nodes_list).astype(np.float32)

    # Add native polyline edges.
    for ids in poly_node_ids:
        for a, b in zip(ids[:-1], ids[1:]):
            _add_edge(a, b)

    # Merge/connect near-identical vertices where different polylines meet.
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, pt in enumerate(nodes):
        key = (int(round(float(pt[0]) / merge_tol_px)),
               int(round(float(pt[1]) / merge_tol_px)))
        buckets.setdefault(key, []).append(i)

    for key, idxs in buckets.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nkey = (key[0] + dx, key[1] + dy)
                if nkey not in buckets:
                    continue
                for i in idxs:
                    for j in buckets[nkey]:
                        if j <= i:
                            continue
                        if float(np.linalg.norm(nodes[i] - nodes[j])) <= merge_tol_px:
                            _add_edge(i, j)

    return nodes, adjacency


def _dijkstra_path(nodes: np.ndarray,
                   adjacency: list[list[tuple[int, float]]],
                   start_idx: int,
                   goal_idx: int) -> np.ndarray | None:
    if start_idx == goal_idx:
        return nodes[[start_idx]].astype(np.float32)
    n = len(nodes)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start_idx))]

    while heap:
        cur_d, u = heapq.heappop(heap)
        if cur_d > dist[u]:
            continue
        if u == goal_idx:
            break
        for v, w in adjacency[u]:
            nd = cur_d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, int(v)))

    if not np.isfinite(dist[goal_idx]):
        return None

    chain = [int(goal_idx)]
    cur = int(goal_idx)
    while prev[cur] >= 0:
        cur = int(prev[cur])
        chain.append(cur)
    chain.reverse()
    return nodes[np.array(chain, dtype=np.int32)].astype(np.float32)

def _build_navigate_path_px(full_path_px, outline_px, ball_start_px,
                             img_w, img_h, complexity_pct=70):
    target = (outline_px[0].astype(np.float32)
              if outline_px is not None and len(outline_px) > 0
              else full_path_px[0].astype(np.float32))
    ball = ball_start_px.astype(np.float32)

    if full_path_px is None or len(full_path_px) < 2:
        n = max(5, int(float(np.linalg.norm(target - ball)) / 2) + 1)
        return np.column_stack([
            np.linspace(ball[0], target[0], n),
            np.linspace(ball[1], target[1], n),
        ]).astype(np.float32)

    path = full_path_px.astype(np.float32)
    if len(path) > _NAV_GRAPH_MAX_POINTS:
        path = _decimate_path(path, _NAV_GRAPH_MAX_POINTS)
    N    = len(path)

    # ── Build graph from merged_px only ──────────────────────────────────
    # merged_px is one continuous path that already contains every bridge
    # as sequential edges.  The splice pattern (P→island→P) causes P to
    # appear twice — proximity edges between these duplicates create the
    # shortcuts that let Dijkstra find genuinely short routes.
    MERGE_TOL = 4.0   # pixels — splice return points are exact or near-exact

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(N)]

    # Sequential edges (the drawn path itself — bridges included)
    for i in range(N - 1):
        w = float(np.linalg.norm(path[i] - path[i + 1]))
        if w > 1e-6:
            adjacency[i].append((i + 1, w))
            adjacency[i + 1].append((i, w))

    # Proximity shortcut edges — captures splice revisit points
    # Use spatial bucketing so this stays O(N) instead of O(N²)
    cell_sz = MERGE_TOL
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, pt in enumerate(path):
        key = (int(pt[0] / cell_sz), int(pt[1] / cell_sz))
        buckets.setdefault(key, []).append(i)

    for i, pt in enumerate(path):
        bx = int(pt[0] / cell_sz)
        by = int(pt[1] / cell_sz)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((bx + dx, by + dy), []):
                    if j <= i:
                        continue
                    w = float(np.linalg.norm(path[i] - path[j]))
                    if 1e-6 < w <= MERGE_TOL:
                        adjacency[i].append((j, w))
                        adjacency[j].append((i, w))

    # ── Find landing point and target index ───────────────────────────────
    approach_dists = np.linalg.norm(path - ball, axis=1)
    landing_idx    = int(np.argmin(approach_dists))
    landing        = path[landing_idx]

    target_dists = np.linalg.norm(path - target, axis=1)
    target_idx   = int(np.argmin(target_dists))

    # Straight approach — only fresh-sand crossing
    n_app    = max(5, int(float(np.linalg.norm(landing - ball)) / 2) + 1)
    approach = np.column_stack([
        np.linspace(ball[0],    landing[0], n_app),
        np.linspace(ball[1],    landing[1], n_app),
    ]).astype(np.float32)

    if landing_idx == target_idx:
        return approach

    # ── Dijkstra on the graph ─────────────────────────────────────────────
    dist_arr         = np.full(N, np.inf, dtype=np.float64)
    prev_arr         = np.full(N, -1,    dtype=np.int32)
    dist_arr[landing_idx] = 0.0
    heap = [(0.0, int(landing_idx))]

    while heap:
        cur_d, u = heapq.heappop(heap)
        if cur_d > dist_arr[u] + 1e-9:
            continue
        if u == target_idx:
            break
        for v, w in adjacency[u]:
            nd = cur_d + w
            if nd < dist_arr[v]:
                dist_arr[v] = nd
                prev_arr[v] = u
                heapq.heappush(heap, (nd, int(v)))

    if not np.isfinite(dist_arr[target_idx]):
        # Fallback: simple retrace backwards along merged_px
        retrace = path[:landing_idx + 1][::-1]
        return np.vstack([approach, retrace[1:]])

    # Reconstruct the Dijkstra path
    chain = []
    cur   = int(target_idx)
    while cur >= 0:
        chain.append(cur)
        cur = int(prev_arr[cur])
    chain.reverse()

    graph_path = path[np.array(chain, dtype=np.int32)]
    return np.vstack([approach, graph_path[1:]])


def get_ball_start_px(mode: str, center: tuple, scale_half: float,
                       reference_px: np.ndarray) -> np.ndarray:
    """
    Compute the ball's pixel-space starting position.

    mode         : 'center' -> ball at the drawing centre (rho=0)
                   'edge'   -> ball at the outer rail, aimed toward reference_px
    center       : (cx, cy) contour centre in pixels
    scale_half   : pixels-per-unit (from compute_contour_scale)
    reference_px : full_path_px[0] -- used to aim the edge start
    """
    cx, cy = float(center[0]), float(center[1])
    if mode == 'center':
        return np.array([cx, cy], dtype=np.float32)
    dx, dy = float(reference_px[0]) - cx, float(reference_px[1]) - cy
    dist   = math.sqrt(dx * dx + dy * dy)
    if dist < 1.0:
        return np.array([cx + scale_half, cy], dtype=np.float32)
    nx, ny = dx / dist, dy / dist
    return np.array([cx + nx * scale_half, cy + ny * scale_half],
                    dtype=np.float32)


# ===========================================================================
#  HOME RETURN  (path-following return to centre after drawing)
# ===========================================================================

def _add_home_return(merged_px: np.ndarray,
                     center_px: np.ndarray) -> np.ndarray:
    """Append a path-following return to the drawing centre."""
    cp   = center_px.astype(np.float32).ravel()[:2]
    path = merged_px.astype(np.float32)
    if len(path) == 0:
        return path
    nearest_idx = int(np.argmin(np.linalg.norm(path - cp, axis=1)))
    retrace     = path[nearest_idx:][::-1]
    return np.vstack([path, retrace[1:], cp.reshape(1, 2)])


# ===========================================================================
#  FULL ISLAND-SPLICE PATH  (Phase 3)
# ===========================================================================

def build_merged_path(contours: list, img_w: int, img_h: int,
                      progress_cb=None,
                      contour_hierarchy: np.ndarray | None = None,
                      boundary_mask: np.ndarray | None = None,
                      max_bridge_px: float | None = None,
                      max_islands: int = 24,
                      debug_splices: bool = False) -> np.ndarray:
    """
    Merge all contours into one continuous pixel-space path via island-splicing.
    Outer ring = trunk.  Every remaining contour spliced in greedy-nearest-first.

    max_bridge_px : if set, any island whose closest trunk distance exceeds this
                    value (in pixels) is silently skipped.  Use this to suppress
                    stray lines for islands only reachable via a very long bridge.
                    A reasonable starting value is ~20% of min(img_w, img_h).

    debug_splices : if True, prints one line per splice showing the island index,
                    bridge distance (px), arc length (px), and insertion position
                    as a fraction of trunk length (0=start, 1=end).  Use this to
                    identify which island is causing a problematic long bridge.
    """
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    if len(contours) == 1:
        c = contours[0].astype(np.float32)
        if _is_closed(c) and float(np.linalg.norm(c[0] - c[-1])) > 1.0:
            c = np.vstack([c, c[[0]]])
        return c

    outer_ring, remaining = _find_outer_ring_contours(
        contours, img_w, img_h, hierarchy=contour_hierarchy)
    if outer_ring:
        outer_ring.sort(key=_arc_length, reverse=True)
        # Only use the single largest piece as the silhouette trunk.
        # Additional outer-ring pieces are already closed loops that are
        # physically separate (e.g. two wings); bridging between their
        # endpoints would cross empty sand.  Island-splicing them (below)
        # routes the bridge through the nearest actual drawn points instead.
        if len(outer_ring) > 1:
            remaining = outer_ring[1:] + remaining
        outer = _join_silhouette_to_loop([outer_ring[0]],
                                          center=(img_w / 2.0, img_h / 2.0),
                                          boundary_mask=boundary_mask)
    else:
        arc_lens  = [_arc_length(c) for c in contours]
        outer_idx = int(np.argmax(arc_lens))
        outer     = contours[outer_idx].astype(np.float32)
        remaining = [c.astype(np.float32)
                     for i, c in enumerate(contours) if i != outer_idx]

    if float(np.linalg.norm(outer[0] - outer[-1])) > 1.0:
        outer = np.vstack([outer, outer[[0]]])

    # Drop edge-boundary artifacts from the island list using the same filter
    # as _find_outer_ring_contours.  These contours (image-frame noise, tiny
    # fragments along the PNG border) were excluded from the outer ring but
    # would otherwise be spliced in as islands, causing the ball to trace the
    # image boundary as a visible stray line.
    _EDGE_MARGIN   = max(15, int(min(img_w, img_h) * 0.025))
    _EDGE_FRAC_MAX = 0.50
    def _is_edge_artifact_island(c):
        near = np.sum(
            (c[:, 0] < _EDGE_MARGIN) | (c[:, 0] > img_w - _EDGE_MARGIN) |
            (c[:, 1] < _EDGE_MARGIN) | (c[:, 1] > img_h - _EDGE_MARGIN)
        )
        return (near / len(c)) > _EDGE_FRAC_MAX

    remaining = [c.astype(np.float32) for c in remaining if not _is_edge_artifact_island(c)]
    if max_islands and len(remaining) > max_islands:
        remaining.sort(key=_arc_length, reverse=True)
        remaining = remaining[:max_islands]

    island_bboxes = [_contour_bbox(c) for c in remaining]

    trunk = list(outer)
    trunk_arr = np.array(trunk, dtype=np.float32)
    trunk_bbox = _contour_bbox(trunk_arr)
    total = len(remaining)
    skipped = []
    for step in range(total):
        best_ri, best_dist = 0, float('inf')
        best_seg           = 0
        for ri, island in enumerate(remaining):
            if _bbox_min_dist(trunk_bbox, island_bboxes[ri]) >= best_dist:
                continue
            seg, _, _, _, _, dist = _find_closest_pair(trunk_arr, island)
            if dist < best_dist:
                best_dist, best_ri, best_seg = dist, ri, seg

        island   = remaining[best_ri]
        arc      = _arc_length(island)
        ins_frac = best_seg / max(len(trunk) - 1, 1)   # 0=start, 1=end of trunk

        if debug_splices:
            print(f"  splice {step+1:3d}/{total}: island_ri={best_ri:3d}  "
                  f"bridge={best_dist:6.1f}px  arc={arc:6.1f}px  "
                  f"insertion={ins_frac:.2f}  "
                  f"({'SKIPPED - over max_bridge_px' if max_bridge_px and best_dist > max_bridge_px else 'spliced'})")

        if max_bridge_px is not None and best_dist > max_bridge_px:
            skipped.append(remaining.pop(best_ri))
            island_bboxes.pop(best_ri)
        else:
            remaining.pop(best_ri)
            island_bboxes.pop(best_ri)
            trunk = _splice_island(trunk, island)
            trunk_arr = np.array(trunk, dtype=np.float32)
            trunk_bbox = _contour_bbox(trunk_arr)

        if progress_cb is not None:
            progress_cb(step + 1, total)

    if debug_splices and skipped:
        print(f"  {len(skipped)} island(s) skipped (bridge > {max_bridge_px:.0f}px)")

    return np.array(trunk, dtype=np.float32)


# ===========================================================================
#  GUI PREVIEW HELPERS
# ===========================================================================

def extract_preview_data(image_path: str,
                         mode: str = "edges",
                         blur: int = 1,
                         canny_low: int = 30,
                         canny_high: int = 100,
                         threshold: int = 128,
                         invert: bool = True,
                         min_area: float = 10.0,
                         min_length: float = 20.0,
                         smooth: int = 2,
                         thin: bool = True,
                         straighten: float = 0.90,
                         max_dim: int = 800):
    gray = load_and_preprocess(image_path, max_dim)
    h, w = gray.shape
    if mode == "silhouette":
        binary = build_silhouette_binary(gray)
        raw_contours, _ = extract_silhouette_contours(binary, w, h)
    elif mode == "edges":
        binary = detect_edges(gray, blur=blur, low=canny_low, high=canny_high)
        if thin:
            binary = thin_edges(binary)
        raw_contours = extract_contours(binary, min_area=min_area)
        raw_contours = filter_short_contours(raw_contours, min_length=min_length)
    else:
        binary = threshold_image(gray, threshold=threshold, invert=invert)
        raw_contours = extract_contours(binary, min_area=min_area)
        raw_contours = filter_short_contours(raw_contours, min_length=min_length)
    contours     = smooth_contours(raw_contours, strength=smooth)
    contours     = straighten_contours(contours, tolerance=straighten)
    return gray, binary, raw_contours, contours, w, h


def _score_contour_set(contours: list, img_w: int, img_h: int,
                        edges: np.ndarray | None = None,
                        grad_mag: np.ndarray | None = None,
                        mean_all_grad: float = 0.0) -> float:
    """
    Quality score for a candidate set of contours for sand-art conversion.
    Returns a value in [0, 1]; higher is better.

    Six components:
      count_score    – targets 20–100 contours (too few = missing detail,
                       too many = noise)
      dom_score      – penalises when one contour holds >65% of total arc
                       (outer ring swamps interior detail)
      frag_score     – penalises a very short median arc (lots of noise fragments)
      density_score  – penalises over-dense edge maps; low Canny fires on every
                       soft gradient and floods the image with edge pixels
      select_score   – rewards high gradient selectivity: detected edges should
                       sit at pixels with much stronger gradients than average,
                       not at soft texture or JPEG noise
      ring_score     – penalises multiple disconnected outer-ring pieces;
                       a single clean outer ring scores 1.0, two pieces 0.6,
                       more pieces decrease further
    """
    if not contours:
        return 0.0

    n         = len(contours)
    arcs      = [_arc_length(c) for c in contours]
    total_arc = sum(arcs)
    max_arc   = max(arcs)
    diagonal  = math.sqrt(img_w ** 2 + img_h ** 2)

    # --- Count ---
    if n < 10:
        count_score = n / 10.0
    elif n <= 100:
        count_score = 1.0
    elif n <= 200:
        count_score = 1.0 - 0.5 * (n - 100) / 100.0
    else:
        count_score = max(0.1, 100.0 / n)

    # --- Dominance ---
    dom       = (max_arc / total_arc) if total_arc > 0 else 1.0
    dom_score = max(0.1, 1.0 - max(0.0, dom - 0.65) * 3.0)

    # --- Fragment quality ---
    median_arc = sorted(arcs)[n // 2]
    meaningful = max(30.0, diagonal * 0.02)
    frag_score = min(1.0, median_arc / meaningful)

    # --- Edge pixel density ---
    # Too-low Canny thresholds fire on every soft gradient and flood the image;
    # the ideal density for clean line-art is roughly 1–8% of pixels.
    if edges is not None:
        density = float(np.sum(edges > 0)) / max(img_w * img_h, 1)
        if density < 0.003:
            density_score = density / 0.003
        elif density <= 0.06:
            density_score = 1.0
        elif density <= 0.18:
            density_score = 1.0 - (density - 0.06) / 0.12
        else:
            density_score = max(0.0, 0.06 / density * 0.5)
    else:
        density_score = 0.5   # neutral when not provided

    # --- Gradient selectivity ---
    # Edge pixels should be at locations with gradients well above the image
    # mean.  selectivity = mean_gradient_at_edges / mean_gradient_everywhere.
    # Good clean art edges: 4–8×.  Noise / soft texture: 1.5–2.5×.
    if grad_mag is not None and edges is not None and mean_all_grad > 0:
        edge_mask = edges > 0
        if np.any(edge_mask):
            mean_edge_grad = float(np.mean(grad_mag[edge_mask]))
            selectivity    = mean_edge_grad / mean_all_grad
        else:
            selectivity = 0.0
        # Rises from 0 at 1.5× to 1 at 5.5× and above
        select_score = min(1.0, max(0.0, (selectivity - 1.5) / 4.0))
    else:
        select_score = 0.5

    # --- Outer ring piece count ---
    # Ideally the outer boundary is a single closed loop. Multiple pieces mean
    # the algorithm bridges disconnected closed loops, which can introduce stray
    # lines. Score 1.0 for one piece, 0.6 for two, decreasing beyond that.
    try:
        outer_ring, _ = _find_outer_ring_contours(contours, img_w, img_h)
        n_ring = len(outer_ring)
        if n_ring <= 1:
            ring_score = 1.0
        elif n_ring == 2:
            ring_score = 0.6
        else:
            ring_score = max(0.1, 0.6 - 0.2 * (n_ring - 2))
    except Exception:
        ring_score = 0.5

    return (count_score   * 0.20 +
            dom_score     * 0.15 +
            frag_score    * 0.15 +
            density_score * 0.25 +
            select_score  * 0.15 +
            ring_score    * 0.10)


def _score_silhouette_fit(contours: list, img_w: int, img_h: int) -> float:
    """
    How suitable is this image for silhouette mode?
    High score = a few clean closed blobs (photos, solid subjects).
    Low score  = fragmented mask or unusable polarity.
    """
    if not contours:
        return 0.0

    total = float(img_w * img_h)
    areas = [float(cv2.contourArea(c)) for c in contours if len(c) >= 3]
    if not areas:
        return 0.0

    n         = len(contours)
    main_frac = max(areas) / total

    if main_frac < 0.04 or main_frac > 0.88:
        return 0.15

    if n == 1:
        count_score = 1.0
    elif n <= 3:
        count_score = 0.85
    elif n <= 5:
        count_score = 0.55
    else:
        count_score = max(0.1, 0.55 - (n - 5) * 0.12)

    if 0.10 <= main_frac <= 0.70:
        size_score = 1.0
    elif 0.70 < main_frac <= 0.88:
        size_score = 0.65
    else:
        size_score = 0.45

    return count_score * 0.55 + size_score * 0.45


def _edges_clearly_better(n_edges: int, edge_score: float) -> bool:
    """True when edge detection should win over silhouette."""
    if n_edges <= 8:
        return edge_score >= 0.30
    if n_edges <= 18:
        return edge_score >= 0.38
    if n_edges <= 35:
        return edge_score >= 0.45
    return edge_score >= 0.52


def _edges_clearly_failing(n_edges: int, edge_score: float) -> bool:
    """True when edge mode will produce unusable spaghetti."""
    if n_edges > 80:
        return True
    if n_edges > 45 and edge_score < 0.45:
        return True
    if n_edges > 28 and edge_score < 0.35:
        return True
    if n_edges > 18 and edge_score < 0.28:
        return True
    return False


def compute_auto_settings(image_path: str, max_dim: int = 1200) -> dict:
    """
    Analyse the image and return a full settings dict with auto-selected mode.
    Defaults to edge detection; silhouette only when edges fail badly and the
    subject collapses cleanly into a solid mask.
    """
    gray     = load_and_preprocess(image_path, max_dim)
    h, w     = gray.shape
    diagonal = math.sqrt(w ** 2 + h ** 2)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if   lap_var > 300: blur = 1
    elif lap_var > 100: blur = 3
    elif lap_var >  30: blur = 5
    else:               blur = 7

    blurred = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)

    min_length = round(max(8.0,  min(60.0, diagonal * 0.012)), 1)
    min_area   = round(max(5.0,  min(30.0, (w * h) * 0.00020)), 1)

    sobelx        = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely        = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag      = np.sqrt(sobelx ** 2 + sobely ** 2)
    mean_all_grad = float(np.mean(grad_mag)) + 1e-6

    otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_high   = int(np.clip(otsu_val * 0.85, 40, 200))

    raw_candidates = [
        otsu_high * 0.50, otsu_high * 0.80, otsu_high * 1.10,
        otsu_high * 1.50, 50, 120,
    ]
    candidates = sorted({int(np.clip(v, 20, 220)) for v in raw_candidates})

    best_score = -1.0
    best_ch    = otsu_high
    best_cl    = max(10, int(otsu_high * 0.33))
    best_contours: list = []

    for ch in candidates:
        cl    = max(10, min(ch - 5, int(ch * 0.33)))
        edges = cv2.Canny(blurred, cl, ch)
        raw, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [
            c.reshape(-1, 2).astype(np.float32) for c in raw
            if len(c) >= 3
            and cv2.contourArea(c) >= min_area
            and _arc_length(c.reshape(-1, 2).astype(np.float32)) >= min_length
        ]
        score = _score_contour_set(contours, w, h,
                                   edges=edges,
                                   grad_mag=grad_mag,
                                   mean_all_grad=mean_all_grad)
        if score > best_score:
            best_score = score
            best_ch, best_cl = ch, cl
            best_contours = contours

    n_edges = len(best_contours)
    edge_score = best_score

    sil_binary = build_silhouette_binary(gray)
    sil_contours, _ = extract_silhouette_contours(sil_binary, w, h)
    n_sil = len(sil_contours)
    sil_score = _score_silhouette_fit(sil_contours, w, h)

    _, thresh_bw = cv2.threshold(blurred, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_raw = extract_contours(thresh_bw, min_area=min_area * 2)
    n_thresh = len(filter_short_contours(thresh_raw, min_length=min_length * 2))

    silhouette_wins = (
        n_sil >= 1
        and sil_score >= 0.55
        and _edges_clearly_failing(n_edges, edge_score)
        and not _edges_clearly_better(n_edges, edge_score)
    )

    base = {
        'auto_detected': True,
        'blur': blur,
        'canny_low': best_cl,
        'canny_high': best_ch,
        'threshold': int(np.clip(otsu_val, 40, 200)),
        'invert': True,
        'straighten': 0.90,
        'max_dim': 800,
        'mirror': False,
        'add_home': False,
    }

    if silhouette_wins:
        preset = 'silhouette' if n_sil == 1 else 'silhouette_holes'
        label  = 'Silhouette' if n_sil == 1 else 'Silhouette + holes'
        return {
            **base,
            'mode': 'silhouette',
            'preset': preset,
            'preset_label': label,
            'min_area': round(max(min_area, (w * h) * 0.0005), 1),
            'min_length': round(max(min_length, diagonal * 0.04), 1),
            'smooth': 3,
            'thin': False,
            'max_points': 12000,
            'fill': 0.95,
            'ball_start': 'none',
        }

    if _edges_clearly_better(n_edges, edge_score) or n_edges >= 1:
        if n_edges <= 6:
            preset, label = 'line_art', 'Line art'
        elif n_edges <= 14:
            preset, label = 'detailed_line', 'Detailed line art'
        else:
            preset, label = 'edges', 'Edge detection'
        mode = 'edges'
    elif 1 <= n_thresh <= 8:
        mode = 'threshold'
        preset, label = 'solid_shape', 'Solid shape'
    elif n_sil >= 1 and sil_score >= 0.55:
        mode = 'silhouette'
        preset = 'silhouette' if n_sil == 1 else 'silhouette_holes'
        label  = 'Silhouette' if n_sil == 1 else 'Silhouette + holes'
    else:
        mode = 'edges'
        preset, label = 'edges', 'Edge detection'

    if n_edges > 20 and mode == 'edges':
        min_length = round(max(min_length, diagonal * 0.022), 1)
        min_area   = round(max(min_area, (w * h) * 0.00035), 1)
    elif n_edges > 12 and mode == 'edges':
        min_length = round(max(min_length, diagonal * 0.016), 1)

    if mode == 'silhouette':
        min_area = round(max(min_area, (w * h) * 0.0005), 1)
        min_length = round(max(min_length, diagonal * 0.04), 1)

    return {
        **base,
        'mode': mode,
        'preset': preset,
        'preset_label': label,
        'min_area': min_area,
        'min_length': min_length,
        'smooth': 2 if mode == 'edges' else 3,
        'thin': mode == 'edges',
        'max_points': 15000 if mode == 'edges' else 12000,
        'fill': 1.0 if mode == 'edges' else 0.95,
        'ball_start': 'center' if mode == 'edges' else 'none',
    }


def compute_mode_settings(image_path: str, max_dim: int = 800) -> dict:
    """
    Return tuned settings for each detection mode so the user can compare
    silhouette, edge, and threshold previews before choosing one.
    """
    auto     = compute_auto_settings(image_path, max_dim)
    gray     = load_and_preprocess(image_path, max_dim)
    h, w     = gray.shape
    diagonal = math.sqrt(w ** 2 + h ** 2)
    min_length = auto.get('min_length', 20.0)
    min_area   = auto.get('min_area', 10.0)

    shared = {
        'blur': auto['blur'],
        'canny_low': auto['canny_low'],
        'canny_high': auto['canny_high'],
        'threshold': auto['threshold'],
        'invert': auto['invert'],
        'straighten': auto['straighten'],
        'max_dim': auto['max_dim'],
        'mirror': False,
        'add_home': False,
    }

    edge_min_length = auto.get('min_length', min_length)
    edge_min_area   = auto.get('min_area', min_area)

    modes = {
        'silhouette': {
            **shared,
            'mode': 'silhouette',
            'preset': 'silhouette',
            'preset_label': 'Silhouette',
            'min_area': round(max(min_area, (w * h) * 0.0005), 1),
            'min_length': round(max(min_length, diagonal * 0.04), 1),
            'smooth': 3,
            'thin': False,
            'max_points': 12000,
            'fill': 0.95,
            'ball_start': 'none',
        },
        'edges': {
            **shared,
            'mode': 'edges',
            'preset': 'edges',
            'preset_label': 'Edge detection',
            'min_area': edge_min_area,
            'min_length': edge_min_length,
            'smooth': 2,
            'thin': True,
            'max_points': 15000,
            'fill': 1.0,
            'ball_start': 'center',
        },
        'threshold': {
            **shared,
            'mode': 'threshold',
            'preset': 'threshold',
            'preset_label': 'Threshold',
            'min_area': round(max(min_area, (w * h) * 0.0003), 1),
            'min_length': round(max(min_length, diagonal * 0.02), 1),
            'smooth': 3,
            'thin': False,
            'max_points': 12000,
            'fill': 0.95,
            'ball_start': 'none',
        },
    }

    return {
        'suggested_mode': auto.get('mode', 'edges'),
        'modes': modes,
    }


def plan_preview_stitch_segments(contours, img_w, img_h, outside_in=True,
                                  complexity_pct=100, greedy_pct=100,
                                  strict_mode=False, progress_cb=None):
    """GUI stitch-preview compatibility shim."""
    merged    = build_merged_path(contours, img_w, img_h, progress_cb=progress_cb)
    draw_only = list(contours)
    kinds     = ['draw'] * len(draw_only)
    return draw_only, [merged], kinds


def preview_thr(thr_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping preview.")
        return
    thetas, rhos = [], []
    with open(thr_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                thetas.append(float(parts[0]))
                rhos.append(float(parts[1]))
    x = [r * math.cos(t) for t, r in zip(thetas, rhos)]
    y = [r * math.sin(t) for t, r in zip(thetas, rhos)]
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(x, y, linewidth=0.5, color="saddlebrown")
    ax1.set_aspect("equal")
    ax1.set_title("Sand path (Cartesian)")
    ax1.add_patch(plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--"))
    ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1)
    ax1.set_facecolor("#f5e6c8")
    ax2 = fig.add_subplot(122, polar=True)
    ax2.plot(thetas, rhos, linewidth=0.5, color="saddlebrown")
    ax2.set_title("Sand path (polar)", pad=15)
    ax2.set_facecolor("#f5e6c8")
    plt.suptitle(f"{Path(thr_path).name}  --  {len(thetas)} waypoints",
                 fontsize=10)
    plt.tight_layout()
    plt.show()


# ===========================================================================
#  MAIN PIPELINE
# ===========================================================================

def build_thr_path(image_path: str,
                   mode: str = 'edges',
                   blur: int = 1,
                   canny_low: int = 30,
                   canny_high: int = 100,
                   threshold: int = 128,
                   invert: bool = True,
                   min_area: float = 10.0,
                   min_length: float = 20.0,
                   smooth: int = 2,
                   thin: bool = True,
                   straighten: float = 0.90,
                   max_dim: int = 800,
                   max_points: int = 15_000,
                   outside_in: bool = True,
                   add_home: bool = False,
                   complexity_pct: int = 100,
                   strict_mode: bool = False,
                   max_retrace_points: int = 0,
                   greedy_pct: int = 100,
                   fill: float = 1.0,
                   ball_start: str = 'center',
                   progress_cb=None) -> dict:
    """
    Full pipeline: image -> dict with keys:

        'polar'     : Nx2 (theta, rho) -- outline prepended to full drawing
        'outline'   : Mx2 (theta, rho) -- outline loop (GUI tri-colour preview)
        'navigate'  : Kx2 (theta, rho) -- navigate path from ball to outline[0]
        'n_outline' : int              -- leading polar points that are the outline

    ball_start : 'center' | 'edge' | 'none'
        Where the ball is parked before this track starts.
        'none' skips navigate computation entirely.

    Unused params (complexity_pct, strict_mode, greedy_pct) accepted silently
    for drop-in GUI compatibility with SandArt2/SandArt3.
    """
    gray = load_and_preprocess(image_path, max_dim)
    h, w = gray.shape

    if mode == 'silhouette':
        binary = build_silhouette_binary(gray)
        contours_px, contour_hierarchy = extract_silhouette_contours(binary, w, h)
        if not contours_px:
            return {'polar': np.zeros((1, 2)), 'outline': None,
                    'navigate': None, 'n_outline': 0}
        contours_px = smooth_contours(contours_px, strength=smooth)
        contours_px = straighten_contours(contours_px, tolerance=straighten)
    elif mode == 'edges':
        binary = detect_edges(gray, blur=blur, low=canny_low, high=canny_high)
        if thin:
            binary = thin_edges(binary)
        contours_px, contour_hierarchy = extract_contours_with_hierarchy(
            binary, min_area=min_area, retrieval_mode=cv2.RETR_TREE)
        contours_px, contour_hierarchy = filter_short_contours_with_hierarchy(
            contours_px, contour_hierarchy, min_length=min_length)
        if not contours_px:
            return {'polar': np.zeros((1, 2)), 'outline': None,
                    'navigate': None, 'n_outline': 0}
        contours_px = smooth_contours(contours_px, strength=smooth)
        contours_px = straighten_contours(contours_px, tolerance=straighten)
    else:
        binary = threshold_image(gray, threshold=threshold, invert=invert)
        contours_px, contour_hierarchy = extract_contours_with_hierarchy(
            binary, min_area=min_area, retrieval_mode=cv2.RETR_TREE)
        contours_px, contour_hierarchy = filter_short_contours_with_hierarchy(
            contours_px, contour_hierarchy, min_length=min_length)
        if not contours_px:
            return {'polar': np.zeros((1, 2)), 'outline': None,
                    'navigate': None, 'n_outline': 0}
        contours_px = smooth_contours(contours_px, strength=smooth)
        contours_px = straighten_contours(contours_px, tolerance=straighten)

    center     = compute_contour_center(contours_px, w, h)
    scale_half = compute_contour_scale(contours_px, w, h, fill=fill,
                                       center=center)

    if mode == 'silhouette':
        bridge_limit = max(40.0, float(min(w, h)) * 0.20)
        max_islands  = min(8, max(0, len(contours_px) - 1))
    else:
        bridge_limit = max(25.0, float(min(w, h)) * 0.14)
        max_islands  = 24

    merged_px = build_merged_path(
        contours_px, w, h, progress_cb=progress_cb,
        contour_hierarchy=contour_hierarchy, boundary_mask=binary,
        max_bridge_px=bridge_limit, max_islands=max_islands)
    if len(merged_px) == 0:
        return {'polar': np.zeros((1, 2)), 'outline': None,
                'navigate': None, 'n_outline': 0}

    if add_home:
        merged_px = _add_home_return(
            merged_px, np.array(center, dtype=np.float32))

    # ---- Phase 2: outline (outermost actual arcs + straight bridges) -----
    outline_px = _build_outline_px(
        contours_px, w, h,
        boundary_mask=binary,
        contour_hierarchy=contour_hierarchy)
    if outline_px is not None and len(outline_px) > 1:
        # Rotate so outline[0] == nearest point to merged_px[0].
        # Both phases then start from the same pixel -> seamless junction.
        outline_px = _align_outline_to_start(outline_px, merged_px[0])

    # ---- Guarantee no drawn point exceeds rho=1.0 after normalisation ------
    # compute_contour_scale uses _filter_border_points to avoid border
    # artifacts inflating the scale, but those same pixels still appear in
    # merged_px / outline_px.  Find the true furthest point and grow
    # scale_half so all drawn points land at or inside the rim.
    _cx, _cy = float(center[0]), float(center[1])
    _pts_to_check = [merged_px]
    if outline_px is not None and len(outline_px) > 1:
        _pts_to_check.append(outline_px)
    _all_draw_pts = np.vstack(_pts_to_check)
    _actual_max_r = float(np.max(
        np.sqrt((_all_draw_pts[:, 0] - _cx) ** 2 +
                (_all_draw_pts[:, 1] - _cy) ** 2)))
    scale_half = max(scale_half, _actual_max_r / fill)

    # ---- Phase 1: navigate (BFS from ball start to outline[0]) ----------
    navigate_px = None
    if ball_start.lower() != 'none' and outline_px is not None:
        nav_mode = 'center' if 'center' in ball_start.lower() else 'edge'
        ball_px  = get_ball_start_px(
            nav_mode, center, scale_half, merged_px[0])
        navigate_px = _build_navigate_path_px(
            merged_px, outline_px, ball_px, w, h,
            complexity_pct=complexity_pct)

    # ---- Normalise all phases to [-1, 1] Cartesian ----------------------
    if outline_px is not None and len(outline_px) > 1:
        outline_cart = normalise_points(outline_px, w, h,
                                         scale_half=scale_half, center=center)
        merged_cart  = normalise_points(merged_px,  w, h,
                                         scale_half=scale_half, center=center)

        n_raw_ol  = len(outline_cart)
        n_raw_me  = len(merged_cart)
        n_raw_tot = n_raw_ol + n_raw_me
        ol_budget = max(10, int(max_points * n_raw_ol / max(n_raw_tot, 1)))
        me_budget = max(10, max_points - ol_budget)

        ol_simple = simplify_path(outline_cart, ol_budget)
        me_simple = simplify_path(merged_cart,  me_budget)

        # approxPolyDP (used inside simplify_path with closed=False) drops the
        # closing duplicate when outline_cart[0] == outline_cart[-1], so the last
        # simplified outline point may be some mid-silhouette vertex instead of
        # merged_cart[0].  Re-append the drawing start point explicitly so the
        # outline->drawing junction is always gap-free.
        if float(np.linalg.norm(ol_simple[-1] - me_simple[0])) > 1e-6:
            ol_simple = np.vstack([ol_simple, me_simple[[0]]])

        n_outline = len(ol_simple)

        combined_cart = np.vstack([ol_simple, me_simple])
        polar         = cartesian_to_polar(combined_cart)
        polar         = unwrap_theta(polar)

        ol_polar = cartesian_to_polar(outline_cart)
        ol_polar = unwrap_theta(ol_polar)
    else:
        ol_polar  = None
        n_outline = 0
        merged_cart = normalise_points(merged_px, w, h,
                                        scale_half=scale_half, center=center)
        me_simple   = simplify_path(merged_cart, max_points)
        polar       = cartesian_to_polar(me_simple)
        polar       = unwrap_theta(polar)

    # Navigate: convert pixel-space path to polar
    nav_polar = None
    if navigate_px is not None and len(navigate_px) > 1:
        nav_cart  = normalise_points(navigate_px, w, h,
                                      scale_half=scale_half, center=center)
        nav_polar = cartesian_to_polar(nav_cart)
        nav_polar = unwrap_theta(nav_polar)

    return {
        'polar':     polar,
        'outline':   ol_polar,
        'navigate':  nav_polar,
        'n_outline': n_outline,
    }


def image_to_thr(image_path: str, output_path: str, **kwargs) -> None:
    """Convenience file-to-file wrapper."""
    result  = build_thr_path(image_path, **kwargs)
    polar   = result['polar'] if isinstance(result, dict) else result
    comment = (f'Generated by SandTrace\n'
               f'Source: {Path(image_path).name}\n'
               f'Points: {len(polar)}')
    write_thr(polar, output_path, comment=comment)


# Late exports so GUI can call core.join_silhouette_to_loop() etc.
join_silhouette_to_loop  = _join_silhouette_to_loop
find_outer_ring_contours = _find_outer_ring_contours


if __name__ == '__main__':
    raise SystemExit(
        'SandArt.py is an importable module. '
        'Run server.py or call build_thr_path() / image_to_thr() directly.'
    )