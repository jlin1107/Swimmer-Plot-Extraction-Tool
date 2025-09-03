import io
import os
import math
import csv
import cv2
import numpy as np
import pandas as pd
import streamlit as st

def seg_len(p1, p2):
    return float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))

def seg_angle_deg(p1, p2):
    ang = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    ang = abs(ang)
    if ang > 180:
        ang -= 180
    return ang


def load_cv2_image(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


# 1) X-axis detection with LSD (no OCR; vision-only)

def detect_x_axis_with_bounds(image,
                              search_lo_frac=0.40,
                              search_hi_frac=0.92,
                              horiz_tol_deg=8.0,
                              min_axis_len_frac=0.55,
                              use_clahe=True):
  
    img = image.copy()
    if img is None:
        raise FileNotFoundError("Image is None")


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dbg = img.copy()


    # light normalization for stylized plots
    gray_n = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray) if use_clahe else gray


    # debug maps
    _, bw_debug = cv2.threshold(gray_n, 200, 255, cv2.THRESH_BINARY_INV)
    horiz_debug = cv2.morphologyEx(
        gray_n, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    )


    # LSD segments (requires opencv-contrib-python)
    lsd = cv2.createLineSegmentDetector()
    out = lsd.detect(gray_n)[0]
    if out is None:
        raise RuntimeError("LSD returned no line segments; cannot detect x-axis.")


    segments = [tuple(map(float, l[0])) for l in out]
    y_lo = int(h * search_lo_frac)
    y_hi = int(h * search_hi_frac)
    min_axis_len = w * min_axis_len_frac


    best = None
    for (x1f, y1f, x2f, y2f) in segments:
        L = seg_len((x1f, y1f), (x2f, y2f))
        if L < min_axis_len:
            continue
        ang = seg_angle_deg((x1f, y1f), (x2f, y2f))
        if ang <= horiz_tol_deg or abs(ang - 180.0) <= horiz_tol_deg:
            ym = 0.5 * (y1f + y2f)
            if y_lo <= ym <= y_hi:
                if (best is None) or (L > best["L"]):
                    best = dict(x1=x1f, y1=y1f, x2=x2f, y2=y2f, L=L, y_mid=ym)


    if best is None:
        raise RuntimeError("No suitable horizontal axis segment found in the search band.")


    axis_y = int(round(best["y_mid"]))
    x0 = int(round(min(best["x1"], best["x2"])))
    x1 = int(round(max(best["x1"], best["x2"])))


    # draw axis
    cv2.line(dbg, (x0, axis_y), (x1, axis_y), (0, 255, 0), 2)
    cv2.putText(dbg, f"axis_y={axis_y}", (10, max(20, axis_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    return axis_y, (x0, x1), bw_debug, horiz_debug, dbg


# 2) Gridline (tick) detection via morphology around the axis band

def detect_x_gridlines_with_morph(bw,
                                  axis_y, x0, x1,
                                  tick_half_height=12,
                                  vert_kernel_len=17,
                                  min_tick_width=1,
                                  max_tick_width=5,
                                  dedup_px=6):
    """
    Detect vertical tick marks intersecting the x-axis using a narrow band
    around axis_y and a vertical opening kernel (1 x vert_kernel_len).
    """
    h, _ = bw.shape
    band = bw[max(0, axis_y - tick_half_height):min(h, axis_y + tick_half_height), x0:x1]


    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    opened = cv2.morphologyEx(band, cv2.MORPH_OPEN, vker)


    n, _, stats, cents = cv2.connectedComponentsWithStats(opened)
    xs = []
    for i in range(1, n):
        x, y, wcc, hcc, area = stats[i]
        if min_tick_width <= wcc <= max_tick_width and hcc >= vert_kernel_len // 2:
            xs.append(int(cents[i][0]) + x0)


    xs.sort()
    ticks = []
    for x in xs:
        if not ticks or abs(x - ticks[-1]) > dedup_px:
            ticks.append(x)
    return ticks


# 3) Patient bars via morphology (color + structure cues)

def detect_patient_bars_by_morphology(img,
                                      axis_y, x0, x1,
                                      min_height_px=10,
                                      max_height_px=70,
                                      min_width_frac=0.035,
                                      horiz_kernel_frac=0.08):
    """
    Detect horizontal patient bars above the x-axis and within [x0, x1].
    """
    roi = img[0:max(0, axis_y), x0:x1]
    if roi.size == 0:
        return []


    # color mask (saturated colors)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (0, 50, 50), (179, 255, 255))


    # structure mask (dark strokes, horizontal opening)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    axis_span = max(1, x1 - x0)
    k_w = max(15, int(round(axis_span * horiz_kernel_frac)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    struct_mask = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, horiz_kernel)


    # combine, light close
    combined = cv2.bitwise_or(color_mask, struct_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))


    # components
    n, _, stats, _ = cv2.connectedComponentsWithStats(combined)
    bars = []
    min_width_px = max(15, int(round(axis_span * min_width_frac)))


    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_width_px:
            continue
        if not (min_height_px <= h <= max_height_px):
            continue
        bars.append((x0 + x, y, w, h))  # y already absolute (ROI starts at 0)


    return sorted(bars, key=lambda b: b[1])

# 4) Durable responder detection (black-hat squares left of each bar)

def detect_durable_responders_by_blackhat(img, bars, axis_y, x1,
                                          kernel_size=None, bh_thresh=30, ar_tol=0.35):
    """
    Detect small, dark square markers indicating durable responders.
    Search above axis and left of x1 (avoid legend).
    """
    if not bars:
        return []


    if kernel_size is None:
        med_h = int(np.median([bh for (_, _, _, bh) in bars]))
        kernel_size = int(np.clip(round(0.7 * max(12, med_h)), 12, 28))


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, bh_mask = cv2.threshold(bh, bh_thresh, 255, cv2.THRESH_BINARY)


    roi_mask = np.zeros_like(bh_mask)
    roi_mask[:axis_y, :x1] = bh_mask[:axis_y, :x1]


    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flags = [False] * len(bars)


    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        if not (0.6 * kernel_size <= w <= 1.8 * kernel_size and
                0.6 * kernel_size <= h <= 1.8 * kernel_size):
            continue
        if abs((w / float(h)) - 1.0) > ar_tol:
            continue
        if area / float(w * h) < 0.45:
            continue


        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) not in (4, 5):
            continue


        cy = y + h / 2.0
        for i, (bx, by, bw_i, bh_i) in enumerate(bars):
            if by <= cy <= (by + bh_i) and (x + w) < bx:
                flags[i] = True


    return flags


# 5) Legend & stage-square detection (below axis)  ✅ fixed key=cv2.contourArea

def detect_legend_and_stage_squares_single(img, axis_y,
                                           sq_min_area=200, sq_max_area=2000,
                                           sat_thresh=50, val_thresh=50):
    h, w = img.shape[:2]
    y0 = min(h - 1, axis_y + 10)
    roi = img[y0:, :].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0, 0, 0, 0), []


    best = max(cnts, key=cv2.contourArea)   # <-- fix
    x, y0p, w0, h0 = cv2.boundingRect(best)
    legend_box = (x, y0p + y0, x + w0, y0p + h0 + y0)


    lx0, ly0, lx1, ly1 = map(int, legend_box)
    roi2 = img[ly0:ly1, lx0:lx1]
    hsv = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, sat_thresh, val_thresh), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))


    squares = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x2, y2, w2, h2 = cv2.boundingRect(approx)
        area, ar = w2 * h2, w2 / float(h2)
        if sq_min_area < area < sq_max_area and 0.8 < ar < 1.25:
            squares.append((lx0 + x2, ly0 + y2, w2, h2))
    return legend_box, squares


# 6) Continued-response detection near right end (Hough-based)

def detect_continued_responses(img, bars, width_frac=0.12,
                               angle_range=(22, 75), line_thresh=12, min_gap=2):
    if not bars:
        return []
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = []
    for (bx, by, bw_i, bh_i) in bars:
        w_crop = max(4, int(bw_i * width_frac))
        x0 = max(0, bx + bw_i - w_crop)
        x1 = min(W, bx + bw_i + w_crop)
        y0 = max(0, by)
        y1 = min(H, by + bh_i)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            flags.append(False)
            continue
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        min_len = max(6, int(0.45 * bh_i))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, line_thresh, min_len, min_gap)
        if lines is None:
            flags.append(False)
            continue
        pos, neg = False, False
        a0, a1 = angle_range
        for x1l, y1l, x2l, y2l in lines[:, 0]:
            dx, dy = (x2l - x1l), (y2l - y1l)
            if dx == 0:
                continue
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if a0 <= ang <= a1:
                slope = dy / float(dx)
                if slope > 0:
                    pos = True
                elif slope < 0:
                    neg = True
            if pos and neg:
                break
        flags.append(pos and neg)
    return flags


# 7) Edge-projection line detection within bars

def detect_horizontal_bar_lines(img, bars,
                                band_half_pix=4,
                                min_len_frac=0.05,
                                merge_tol=5,
                                canny_thresh=(50,150)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bar_lines = []
    for bx, by, bw_i, bh_i in bars:
        roi = gray[by:by+bh_i, bx:bx+bw_i]
        cy = bh_i // 2
        y1 = max(0, cy - band_half_pix)
        y2 = min(bh_i, cy + band_half_pix)
        band = roi[y1:y2, :]
        edges = cv2.Canny(band, canny_thresh[0], canny_thresh[1])
        proj = np.sum(edges > 0, axis=0)


        runs, in_run = [], False
        for x, val in enumerate(proj):
            if val and not in_run:
                in_run = True
                start = x
            elif not val and in_run:
                in_run = False
                runs.append((start, x - 1))
        if in_run:
            runs.append((start, bw_i - 1))


        merged = []
        for st, en in runs:
            if not merged:
                merged.append([st, en])
            else:
                pst, pen = merged[-1]
                if st <= pen + merge_tol:
                    merged[-1][1] = max(pen, en)
                else:
                    merged.append([st, en])


        min_len_px = int(bw_i * min_len_frac)
        segs = []
        for st, en in merged:
            if (en - st + 1) >= min_len_px:
                y_coord = by + cy
                segs.append(((bx + st, y_coord), (bx + en, y_coord)))
        bar_lines.append(segs)
    return bar_lines


# 8) Classify each detected line as red or blue (Complete/Partial)

def classify_bar_line_colors(img, bar_lines,
                             blue_range=((100,80,80),(140,255,255)),
                             red_range1=((0,80,80),(10,255,255)),
                             red_range2=((160,80,80),(179,255,255))):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colors_per_bar = []
    for segs in bar_lines:
        bar_colors = []
        for (x1, y1), (x2, y2) in segs:
            xm = (x1 + x2) // 2
            ym = (y1 + y2) // 2
            h, s, v = hsv[ym, xm]
            (bh0, bs0, bv0), (bh1, bs1, bv1) = blue_range
            is_blue = (bh0 <= h <= bh1 and bs0 <= s <= bs1 and bv0 <= v <= bv1)
            (r10, rs10, rv10), (r11, rs11, rv11) = red_range1
            (r20, rs20, rv20), (r21, rs21, rv21) = red_range2
            is_red = ((r10 <= h <= r11 and rs10 <= s <= rs11 and rv10 <= v <= rv11)
                      or (r20 <= h <= r21 and rs20 <= s <= rs21 and rv20 <= v <= rv21))
            if is_blue and not is_red:
                bar_colors.append('Partial')
            elif is_red and not is_blue:
                bar_colors.append('Complete')
            else:
                bar_colors.append('unknown')
        colors_per_bar.append(bar_colors)
    return colors_per_bar


# 9) Compute start/end "months" for each line segment

def compute_line_months(bar_lines, bar_line_colors, usable, tick_vals):
    max_idx = len(usable) - 1
    def xp_to_month(xp):
        lo = [t for t in usable if t <= xp]
        hi = [t for t in usable if t >= xp]
        if lo and hi:
            x_lo, x_hi = max(lo), min(hi)
            i_lo, i_hi = tick_vals[x_lo], tick_vals[x_hi]
            if x_lo == x_hi:
                return round(float(i_lo), 1)
            frac = (xp - x_lo) / float(x_hi - x_lo)
            return round(i_lo + frac, 1)
        return round(0.0 if (usable and xp < usable[0]) else float(max_idx), 1)


    all_info = []
    for segs, cols in zip(bar_lines, bar_line_colors):
        info = []
        for ((x1,_),(x2,_)), color in zip(segs, cols):
            info.append((color, xp_to_month(x1), xp_to_month(x2)))
        all_info.append(info)
    return all_info


# 10) Convert results to DataFrame (for display & download)

def results_to_dataframe(line_months_info, patient_stages, patient_treat, continued):
    rows = []
    for i, segments in enumerate(line_months_info, 1):
        stg = patient_stages[i-1] if i-1 < len(patient_stages) else ''
        tr_start, tr_end = patient_treat[i-1] if i-1 < len(patient_treat) else ('','')
        cont_flag = int(continued[i-1]) if i-1 < len(continued) else 0


        if segments:
            for color, start, end in segments:
                rows.append([i, stg, tr_start, tr_end, cont_flag, color, start, end])
        else:
            rows.append([i, stg, tr_start, tr_end, cont_flag, '', '', ''])


    df = pd.DataFrame(rows, columns=[
        'Patient', 'Stage', 'Treatment Start', 'Treatment End',
        'Continued', 'Line Color', 'Response Start', 'Response End'
    ])
    return df


# 11) If continued, extend the last response segment to treatment end

def adjust_last_response_end(line_months_info, patient_treat, continued):
    adjusted = []
    for i, segments in enumerate(line_months_info):
        segs = [list(s) for s in segments]
        if i < len(continued) and i < len(patient_treat) and continued[i] and segs:
            ends = [s[2] for s in segs]
            idx_last = int(np.argmax(ends))
            segs[idx_last][2] = patient_treat[i][1]
        adjusted.append([tuple(s) for s in segs])
    return adjusted


# Streamlit UI

st.set_page_config(page_title="Swimmer Plot Extractor", layout="wide")
st.title("🏊‍♀️ Swimmer Plot Data Extractor")


with st.sidebar:
    st.markdown("**Upload an image** of a swimmer plot and process it.")
    uploaded = st.file_uploader("Image file", type=["png", "jpg", "jpeg"])
    # Removed all sliders; using sensible defaults in code.
    run_btn = st.button("Run Extraction", type="primary", disabled=uploaded is None)


if uploaded and run_btn:
    try:
        img = load_cv2_image(uploaded)
        dbg = img.copy()


        # ---- Defaults for detection (sliders removed) ----
        _search_lo_frac = 0.40
        _search_hi_frac = 0.92
        _horiz_tol_deg = 8.0
        _min_axis_len_frac = 0.55


        # 1) Axis
        axis_y, (x0, x1), bw_dbg, horiz_dbg, dbg = detect_x_axis_with_bounds(
            img,
            search_lo_frac=_search_lo_frac,
            search_hi_frac=_search_hi_frac,
            horiz_tol_deg=_horiz_tol_deg,
            min_axis_len_frac=_min_axis_len_frac,
            use_clahe=True
        )


        # 2) Gridlines/ticks
        ticks = detect_x_gridlines_with_morph(bw_dbg, axis_y, x0, x1)
        usable = ticks[1:-1] if len(ticks) > 2 else ticks
        tick_vals = {tx: i for i, tx in enumerate(usable)}


        for tx in ticks:
            cv2.line(dbg, (tx, axis_y - 10), (tx, axis_y + 10), (255, 0, 0), 2)
            if tx in tick_vals:
                cv2.putText(dbg, str(tick_vals[tx]), (tx - 5, axis_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


        # 3) Bars
        bars = detect_patient_bars_by_morphology(
            img, axis_y, x0, x1,
            min_height_px=10, max_height_px=70,
            min_width_frac=0.035, horiz_kernel_frac=0.08
        )
        for i, (bx, by, bw_i, bh_i) in enumerate(bars, 1):
            cv2.rectangle(dbg, (bx, by), (bx + bw_i, by + bh_i), (0, 255, 255), 2)
            cv2.putText(dbg, str(i), (bx, max(12, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


        # 4) Durable responders
        durable = detect_durable_responders_by_blackhat(img, bars, axis_y, x1)
        for (bx, by, bw_i, bh_i), flag in zip(bars, durable):
            if flag:
                px = max(0, bx - 10)
                py = int(by + bh_i * 0.5) - 6
                cv2.rectangle(dbg, (px, py), (px + 8, py + 8), (255, 0, 255), -1)
                cv2.putText(dbg, "D", (px - 14, py + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


        # 5) Legend & stage squares
        legend_box, squares = detect_legend_and_stage_squares_single(img, axis_y)
        squares = sorted(squares, key=lambda s: s[0])
        stage_colors = []
        for sx, sy, sw, sh in squares:
            seg = max(2, int(sw * 0.1))
            mb, mg, mr = cv2.mean(img[sy:sy + sh, sx:sx + seg])[:3]
            stage_colors.append((mb, mg, mr))


        # patient -> stage by color proximity
        patient_stages = []
        for bx, by, bw_i, bh_i in bars:
            seg = max(2, int(bw_i * 0.1))
            mb, mg, mr = cv2.mean(img[by:by + bh_i, bx:bx + seg])[:3]
            dists = [np.linalg.norm(np.array((mb, mg, mr)) - np.array(sc)) for sc in stage_colors]
            si = int(np.argmin(dists)) + 1 if dists else 0
            patient_stages.append(si)
        for (bx, by, bw_i, bh_i), si in zip(bars, patient_stages):
            cv2.putText(dbg, f"S{si}", (bx, by + bh_i + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # 6) Treatment start/end from bar bounds and ticks
        patient_treat = []
        max_idx = len(usable) - 1 if usable else 0
        for bx, by, bw_i, bh_i in bars:
            # start
            lo = [t for t in usable if t <= bx]
            hi = [t for t in usable if t >= bx]
            if lo and hi:
                x_lo, x_hi = max(lo), min(hi)
                i_lo, i_hi = tick_vals[x_lo], tick_vals[x_hi]
                frac = 0 if x_lo == x_hi else (bx - x_lo) / float(x_hi - x_lo)
                start = round(i_lo + frac, 1)
            else:
                start = 0.0 if (usable and bx < usable[0]) else float(max_idx)
            # end
            ex = bx + bw_i
            lo = [t for t in usable if t <= ex]
            hi = [t for t in usable if t >= ex]
            if lo and hi:
                x_lo, x_hi = max(lo), min(hi)
                i_lo, i_hi = tick_vals[x_lo], tick_vals[x_hi]
                frac = 0 if x_lo == x_hi else (ex - x_lo) / float(x_hi - x_lo)
                end = round(i_lo + frac, 1)
            else:
                end = 0.0 if (usable and ex < usable[0]) else float(max_idx)


            patient_treat.append((start, end))
            cv2.putText(dbg, f"{start:.1f}-{end:.1f}", (bx, by + bh_i + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # 7) Continued responses
        continued = detect_continued_responses(img, bars)
        for (bx, by, bw_i, bh_i), flag in zip(bars, continued):
            if flag:
                cv2.putText(dbg, "C", (bx + bw_i - 12, by + bh_i // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # 8) Line detection & color classification within bars
        bar_lines = detect_horizontal_bar_lines(img, bars)
        bar_line_colors = classify_bar_line_colors(img, bar_lines)


        # 9) Compute per-line month spans, adjust with continued
        line_months_info = compute_line_months(bar_lines, bar_line_colors, usable, tick_vals)
        line_months_info = adjust_last_response_end(line_months_info, patient_treat, continued)


        # -------------------- Outputs --------------------


        # Results table
        df = results_to_dataframe(line_months_info, patient_stages, patient_treat, continued)
        st.subheader("Extracted Data")
        st.dataframe(df, use_container_width=True)


        # Download: CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="swimmer_results.csv",
            mime="text/csv"
        )


        # Quick summary
        st.subheader("Summary")
        st.write(f"- Axis y = **{axis_y}**, ticks detected: **{len(ticks)}**")
        st.write(f"- Bars detected: **{len(bars)}**, durable responders: **{int(np.sum(durable))}**, continued: **{int(np.sum(continued))}**")


        # Annotated image (RGB for Streamlit display) — moved to bottom and smaller
        dbg_rgb = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
        st.subheader("Annotated Image")
        st.image(dbg_rgb, width=600)  # smaller, fixed width for easier viewing


        # Download: annotated PNG (encode from BGR to preserve colors)
        ok, png_bytes = cv2.imencode(".png", dbg)
        if ok:
            st.download_button(
                "Download Annotated Image",
                data=png_bytes.tobytes(),
                file_name="swimmer_annotated.png",
                mime="image/png"
            )


    except Exception as e:
        st.error(f"Processing failed: {e}")









