# StintCalculator_Streamlit_v2.7.py
# Per-car green-flag stints; UX + pace compare + per-row falloff models for strategies
# v2.7: Added manual coefficient adjustment for falloff curves

import io
import json
from typing import Dict, List, Tuple, Optional
from string import Template

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

APP_TITLE = "TRK – Stint Calculator"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="TH_FullLogo_White.png",
    layout="wide"
)

# ---- compact top spacing ----
st.markdown(
    """
<style>
.block-container { padding-top: .6rem; }
h1 { margin-bottom: .25rem; }
[data-testid="stHeader"] { height: 2rem; background: transparent; }
div[role="tablist"] { margin-top: .25rem; }
div[data-testid="stDecoration"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_CAR_COLORS: Dict[str, str] = {
    "1": "#02E3FF",
    "5": "#8A2BE2",
    "9": "#E53935",
    "3": "#4B5563",
    "8": "#BDBDBD",
    "24": "#F59E0B",
    "97": "#FFD700",
    "88": "#003FFF",
    "48": "#2E7D32",
}
st.markdown("""
<style>
:root {
  --primary-color: #10B981 !important;
  --primary-color-text: #ffffff !important;
  --secondary-background-color: #f5f5f5 !important;
}
/* Reduce spacing between expanders */
.streamlit-expanderHeader {
  margin-bottom: 0.25rem !important;
}
details[open] {
  margin-bottom: 0.5rem !important;
}
/* Compact icon buttons in strategy table */
[data-testid="column"] button[kind="secondary"] {
  padding: 0.15rem 0.4rem !important;
  min-height: 1.5rem !important;
  font-size: 0.8rem !important;
  line-height: 1 !important;
}
/* Center Color column in data_editor */
[data-testid="stDataEditor"] td:nth-child(2) {
  text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ========================= math =========================
def falloff_equation(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * (x ** -0.5) + c * ((x ** -0.5) * np.log(x))

def piecewise_linear_equation(x: np.ndarray, transitions: List[float], slopes: List[float], intercepts: List[float]) -> np.ndarray:
    """Evaluate piecewise linear function at x values.

    With N transitions, we have N segments:
    - Segment 0: x <= T1
    - Segment 1: T1 < x <= T2
    - ...
    - Segment N-1: T(N-1) < x (extends to T_N and beyond)

    The last transition (T4) marks the end of the fitted region but
    the last segment continues with the same slope for extrapolation.
    """
    result = np.zeros_like(x, dtype=float)
    transitions_sorted = sorted(transitions)

    # N transitions = N segments (last segment extends beyond last transition)
    n_segments = len(transitions_sorted)
    max_segment = min(n_segments, len(slopes), len(intercepts)) - 1

    for i, x_val in enumerate(x):
        # Find which segment this x belongs to
        # Use all transitions except the last one as boundaries
        segment = 0
        for t in transitions_sorted[:-1]:  # Don't use last transition as boundary
            if x_val > t:
                segment += 1
            else:
                break

        # Clamp segment to valid range
        segment = min(segment, max_segment)
        result[i] = slopes[segment] * x_val + intercepts[segment]

    return result

def detect_transition_points(x: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """Detect transition points where slope changes significantly and fit piecewise linear model.

    Now uses 4 transitions for 5 segments, with T4 extending beyond data for extrapolation.
    """
    if len(x) < 4:
        # Not enough data, use simple linear fit
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
        intercept = y[0] - slope * x[0]
        return [x[-1]], [slope], [intercept]

    # 4 transitions create 5 segments
    # T1 ~lap 2, T2/T3 spread across data, T4 at last lap (for extrapolation control)
    n_laps = len(x)
    transitions = [
        2.0,                                    # T1: early transition
        float(max(3, int(n_laps * 0.33))),      # T2: ~1/3 through
        float(max(4, int(n_laps * 0.66))),      # T3: ~2/3 through
        float(x[-1])                            # T4: last lap (can be extended by user)
    ]

    # Fit linear segments using refit function
    slopes, intercepts = refit_linear_segments(x, y, transitions)

    return transitions, slopes, intercepts

def refit_linear_segments(x: np.ndarray, y: np.ndarray, transitions: List[float]) -> Tuple[List[float], List[float]]:
    """Refit linear segments by drawing straight lines between transition points.

    With N transitions, we have N+1 segments. Each segment connects consecutive key points.
    Key points are: start (lap 1), T1, T2, ..., TN.

    For transitions beyond the data (extrapolation), uses the last known slope.
    """
    transitions_sorted = sorted(transitions)
    last_data_x = float(x[-1])

    # Key points: start + all transitions
    key_x = [float(x[0])] + transitions_sorted

    # Get Y values at each key point (use closest data point, or extrapolate)
    key_y = []
    last_known_slope = None
    last_known_y = None
    last_known_x = None

    for i, kx in enumerate(key_x):
        if kx <= last_data_x:
            # Within data range - use actual data
            idx = np.abs(x - kx).argmin()
            key_y.append(float(y[idx]))
            last_known_x = kx
            last_known_y = float(y[idx])
        else:
            # Beyond data - extrapolate using the previous segment's slope
            if last_known_slope is not None and last_known_y is not None:
                extrapolated_y = last_known_y + last_known_slope * (kx - last_known_x)
                key_y.append(extrapolated_y)
            else:
                # Fallback: use last data point
                key_y.append(float(y[-1]))

        # Calculate slope for next iteration's extrapolation
        if i > 0 and key_x[i] != key_x[i-1]:
            last_known_slope = (key_y[i] - key_y[i-1]) / (key_x[i] - key_x[i-1])
            if key_x[i] <= last_data_x:
                last_known_x = key_x[i]
                last_known_y = key_y[i]

    # Calculate slope and intercept for each segment
    # N transitions = N+1 segments, but we have N+1 key points (start + N transitions)
    # So we get N slopes from consecutive pairs, which is correct for N transitions
    slopes = []
    intercepts = []

    for i in range(len(key_x) - 1):
        x1, y1 = key_x[i], key_y[i]
        x2, y2 = key_x[i + 1], key_y[i + 1]

        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = 0
        intercept = y1 - slope * x1

        slopes.append(slope)
        intercepts.append(intercept)

    return slopes, intercepts

def fit_falloff_linear_ls(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = x.astype(float)
    y = y.astype(float)
    X = np.vstack(
        [np.ones_like(x), (x ** -0.5), ((x ** -0.5) * np.log(x))]
    ).T
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b), float(c)

def compute_effective_lap_time(
    stint_lap: int,
    a: float,
    b: float,
    c: float,
    base: float,
    use_model_base: bool = False,
    model_name: Optional[str] = None,
    use_linear: bool = False,
    linear_params: Optional[dict] = None
) -> float:
    # Check if we should use linear mode
    if use_linear and linear_params:
        transitions = linear_params.get("transitions", [])
        slopes = linear_params.get("slopes", [])
        intercepts = linear_params.get("intercepts", [])
        raw = piecewise_linear_equation(np.array([float(stint_lap)]), transitions, slopes, intercepts)[0]

        if use_model_base:
            return float(raw)
        else:
            # Normalize using lap 3 as reference
            base0 = piecewise_linear_equation(np.array([3.0]), transitions, slopes, intercepts)[0]
            return float(base + (raw - base0))
    else:
        # Use falloff equation
        raw = falloff_equation(np.array([stint_lap], float), a, b, c)[0]
        if use_model_base:
            # Use the model's fitted base time directly
            return float(raw)
        else:
            # Normalize to the specified base lap time using lap 3 as reference
            # Lap 3 is more stable than lap 1 (avoids singularity issues at lap 1)
            base0 = falloff_equation(np.array([3.0], float), a, b, c)[0]
            return float(base + (raw - base0))

def compute_total_time_and_laps_dual_model(
    pit_stops: List[int],
    start_lap: int,
    end_lap: int,
    pit_time: float,
    a1: float,
    b1: float,
    c1: float,
    a2: float,
    b2: float,
    c2: float,
    base_lap_time: float,
    use_model_base: bool = False,
    model1_name: Optional[str] = None,
    model2_name: Optional[str] = None,
    use_linear1: bool = False,
    linear_params1: Optional[dict] = None,
    use_linear2: bool = False,
    linear_params2: Optional[dict] = None,
) -> Tuple[float, List[float]]:
    if end_lap < start_lap:
        return 0.0, []
    pit_stops_sorted = sorted(set(pit_stops))
    first_stop = next(
        (ps for ps in pit_stops_sorted if start_lap <= ps < end_lap), None
    )

    cum = 0.0
    out: List[float] = []
    stint_start = start_lap
    using_first = True
    for lap in range(start_lap, end_lap + 1):
        if first_stop is not None and lap > first_stop:
            using_first = False
        stint_lap = lap - stint_start + 1
        if using_first:
            lt = compute_effective_lap_time(
                stint_lap, a1, b1, c1, base_lap_time, use_model_base,
                model1_name, use_linear1, linear_params1
            )
        else:
            lt = compute_effective_lap_time(
                stint_lap, a2, b2, c2, base_lap_time, use_model_base,
                model2_name, use_linear2, linear_params2
            )
        cum += lt
        out.append(cum)
        if lap in pit_stops_sorted and lap < end_lap:
            cum += pit_time
            stint_start = lap + 1
    return out[-1] if out else 0.0, out

def find_optimal_single_stop(s, e, pit, a1, b1, c1, a2, b2, c2, base, use_model_base=False):
    if e <= s + 1:
        return None, None
    best = None
    where = None
    for p in range(s + 1, e):
        tt, _ = compute_total_time_and_laps_dual_model(
            [p], s, e, pit, a1, b1, c1, a2, b2, c2, base, use_model_base
        )
        if best is None or tt < best:
            best, where = tt, p
    return best, where

def find_optimal_two_stop(s, e, pit, a1, b1, c1, a2, b2, c2, base, use_model_base=False):
    if e <= s + 2:
        return None, None
    best = None
    pair = None
    for p1 in range(s + 1, e - 1):
        for p2 in range(p1 + 1, e):
            tt, _ = compute_total_time_and_laps_dual_model(
                [p1, p2], s, e, pit, a1, b1, c1, a2, b2, c2, base, use_model_base
            )
            if best is None or tt < best:
                best, pair = tt, (p1, p2)
    return best, pair

# ========================= session defaults =========================
SS_DEFAULTS = {
    "raw_df": pd.DataFrame(),
    "car_visible": set(),
    "car_colors": {},
    "car_ranges": {},  # {car:(start,end)} inclusive 1-based
    "slow_thresh": 1.0,
    "aligned_avgs": [],
    "_upload_token": None,
    # Model tab
    "model_table": pd.DataFrame(),
    "model_params": {},
    "model_params_original": {},  # Store original fitted values for reset
    "model_colors": {},
    "model_linear_mode": {},  # {model_name: bool} - True if using linear mode
    "model_linear_params": {},  # {model_name: {"transitions": [x1, x2, x3], "slopes": [s1, s2, s3, s4], "intercepts": [...]}}
    "model_visible": {},  # {model_name: bool} - True if model should be plotted

    # Global strategy settings (shared across all tabs)
    "global_pit_time": None,  # Requires user input
    # Strategies tabs (per-row pre/post models now)
    "strategies_tabs": [
        {
            "tab_name": "Stage 1",
            "strategies": [],
            "start_lap": 1,
            "end_lap": 50,
            "pit_time": 31.0,
        },
        {
            "tab_name": "Stage 2",
            "strategies": [],
            "start_lap": 1,
            "end_lap": 50,
            "pit_time": 31.0,
        },
        {
            "tab_name": "Stage 3",
            "strategies": [],
            "start_lap": 1,
            "end_lap": 50,
            "pit_time": 31.0,
        },
        {
            "tab_name": "Stage 4",
            "strategies": [],
            "start_lap": 1,
            "end_lap": 50,
            "pit_time": 31.0,
        },
        {
            "tab_name": "Crossover",
            "strategies": [],
            "start_lap": 1,
            "end_lap": 50,
            "pit_time": 31.0,
        },
    ],
    # per-car green-flag settings/state
    "gf_slow_plus": 3.0,        # slow if lap > median + gf_slow_plus (seconds)
    "gf_min_len": 3,            # drop short green segments
    "gf_min_slow": 1,           # min consecutive slow laps to form caution block
    "gf_windows_by_car": {},    # {car:[(s,e),...]}
}

def ss_init():
    for k, v in SS_DEFAULTS.items():
        if k not in st.session_state:
            if isinstance(v, pd.DataFrame):
                st.session_state[k] = v.copy()
            elif isinstance(v, dict):
                st.session_state[k] = v.copy()
            elif isinstance(v, list):
                st.session_state[k] = list(v)
            elif isinstance(v, set):
                st.session_state[k] = set(v)
            else:
                st.session_state[k] = v
    if not isinstance(st.session_state.get("car_visible", set()), set):
        st.session_state.car_visible = set()
    if not isinstance(st.session_state.get("car_ranges", {}), dict):
        st.session_state.car_ranges = {}

ss_init()

# ========================= helpers =========================
def normalize_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
    return df

def _numeric_then_alpha(s: str):
    return (0, int(s)) if str(s).isdigit() else (1, str(s).lower())

# ========================= UI widgets =========================
def render_car_button_grid(all_cars: List[str], columns_per_row=5):
    all_cars = [str(c).strip() for c in all_cars]
    cars_sorted = sorted(all_cars, key=_numeric_then_alpha)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select all", width="stretch", key="btn_select_all"):
            st.session_state.car_visible = set(cars_sorted)
            for car in cars_sorted:
                st.session_state.car_colors.setdefault(
                    car, DEFAULT_CAR_COLORS.get(car, "#1f77b4")
                )
            st.rerun()
    with c2:
        if st.button("Clear all", width="stretch", key="btn_clear_all"):
            st.session_state.car_visible = set()
            st.rerun()

    rows = (len(cars_sorted) + columns_per_row - 1) // columns_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(columns_per_row, gap="small")
        for col in cols:
            if idx >= len(cars_sorted):
                col.empty()
                continue
            car = cars_sorted[idx]
            selected = car in st.session_state.car_visible
            if col.button(
                f"{'✅' if selected else '➕'} {car}",
                key=f"car_btn_{car}",
                width="stretch",
            ):
                if selected:
                    st.session_state.car_visible.discard(car)
                else:
                    st.session_state.car_visible.add(car)
                    st.session_state.car_colors.setdefault(
                        car, DEFAULT_CAR_COLORS.get(car, "#1f77b4")
                    )
                st.rerun()
            idx += 1

    used = [c for c in cars_sorted if c in st.session_state.car_visible]
    if used:
        st.markdown("**Per-car colors**")
        cols_per_row = max(3, min(8, len(used)))
        rows = (len(used) + cols_per_row - 1) // cols_per_row
        i = 0
        for _ in range(rows):
            rcols = st.columns(cols_per_row, gap="small")
            for col in rcols:
                if i >= len(used):
                    col.empty()
                    continue
                car = used[i]
                col.caption(car)
                st.session_state.car_colors[car] = col.color_picker(
                    " ",
                    value=st.session_state.car_colors.get(
                        car, DEFAULT_CAR_COLORS.get(car, "#1f77b4")
                    ),
                    key=f"col_{car}",
                    label_visibility="collapsed",
                )
                i += 1

# ========================= per-car green-flag stints =========================
def _merge_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Return 1-based contiguous runs where mask==1, filtered by min_len."""
    runs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i] == 1:
            j = i
            while j + 1 < n and mask[j + 1] == 1:
                j += 1
            if (j - i + 1) >= min_len:
                runs.append((i + 1, j + 1))
            i = j + 1
        else:
            i += 1
    return runs

def detect_green_flag_stints_per_car(
    df: pd.DataFrame,
    car: str,
    slow_plus_seconds: float,
    min_slow_len: int,
    min_green_len: int,
) -> List[Tuple[int, int]]:
    """
    For one car:
      - slow = lap_time > (median(car) + slow_plus_seconds)
      - Any BLANK / NaN / non-finite lap time is treated as caution (end green stints).
      - Merge contiguous caution (slow OR blank) runs >= min_slow_len -> caution blocks
      - Stints are green segments between/around caution blocks, drop short ones (< min_green_len)
    """
    ser = pd.to_numeric(df[car], errors="coerce").astype(float)
    n = len(ser)
    if n == 0:
        return []

    med = float(ser.dropna().median()) if not ser.dropna().empty else 0.0
    slow_numeric = (ser > (med + float(slow_plus_seconds)))
    is_blank = ~np.isfinite(ser.values)  # NaN/inf

    caution_mask = np.where(is_blank | slow_numeric.fillna(False).values, 1, 0)
    caution_blocks = _merge_runs(caution_mask, max(1, int(min_slow_len)))

    stints: List[Tuple[int, int]] = []
    cur = 1
    for (cs, ce) in caution_blocks:
        if cur <= cs - 1:
            stints.append((cur, cs - 1))
        cur = ce + 1
    if cur <= n:
        stints.append((cur, n))

    stints = [(s, e) for (s, e) in stints if (e - s + 1) >= max(1, int(min_green_len))]
    return stints

# ========================= Top Tabs =========================
st.markdown(
    '<p style="font-size:1.1rem;font-weight:600;color:#555;margin:0 0 0.5rem 0;">TRK – Stint Calculator</p>',
    unsafe_allow_html=True
)
tab1, tab2, tab3 = st.tabs(["Laps & Averages", "Data & Fitting", "Strategies"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Load Data & Select Cars")
    left, right = st.columns([1, 2], gap="large")

    # ======================= LEFT: upload + car grid =======================
    with left:
        up = st.file_uploader("Excel or CSV", type=["xlsx", "xls", "csv"])
        with st.expander("ℹ️ Data Format Requirements", expanded=False):
            st.markdown("""
            **Your file should have:**
            - **First row**: Car numbers (e.g., 1, 5, 9, 24, 88)
            - **Each row below**: Lap times in seconds for each car

            **Example:**
            ```
            1      5      9      88     97
            28.5   28.7   29.1   28.3   29.0
            28.6   28.9   29.2   28.5   29.1
            28.8   29.1   29.3   28.7   29.3
                   34.8   29.5   28.9   29.5
            29.0   29.2   29.6   29.0   29.6
            ```
            """)

        if up is not None:
            token = (up.name, getattr(up, "size", None))
            if st.session_state._upload_token != token:
                try:
                    df_loaded = (
                        pd.read_csv(up)
                        if up.name.lower().endswith(".csv")
                        else pd.read_excel(up)
                    )
                    df_loaded = normalize_columns_to_str(df_loaded)
                    st.session_state.raw_df = df_loaded
                    st.session_state._upload_token = token
                    st.session_state.car_visible = set()
                    st.session_state.car_ranges = {}
                    st.success(f"Data loaded: {up.name}")
                except Exception as e:
                    st.error(f"Load error: {e}")

        df = st.session_state.raw_df.copy()
        if not df.empty:
            df = normalize_columns_to_str(df)
            st.session_state.raw_df = df
            # car grid: only numeric column names by default
            numeric_cols = [c for c in df.columns if str(c).strip().isdigit()]
            render_car_button_grid(numeric_cols, columns_per_row=5)

    # ================= RIGHT: auto-detect • stints • UI =================
    with right:
        # Smart auto-detection based on data statistics
        def calculate_smart_thresholds(df_data, visible_cars_list):
            """Calculate smart thresholds based on lap time distribution"""
            if not visible_cars_list or df_data.empty:
                return 4.0, 3.0, 1, 3

            all_times = []
            for car in visible_cars_list:
                if car in df_data.columns:
                    times = pd.to_numeric(df_data[car], errors="coerce").dropna()
                    if len(times) > 0:
                        all_times.extend(times.tolist())

            if not all_times:
                return 4.0, 3.0, 1, 3

            times_series = pd.Series(all_times)
            std_dev = float(times_series.std())

            # Slow lap threshold: use 2 standard deviations
            slow_thresh = max(2.0, min(6.0, 2.0 * std_dev))

            # Green flag slow threshold: use 1.5 standard deviations
            gf_slow_plus = max(1.5, min(4.0, 1.5 * std_dev))

            # Min slow laps for caution: always 1
            gf_min_slow = 1

            # Min green stint length: scale with data size
            total_laps = len(df_data)
            if total_laps < 50:
                gf_min_len = 3
            elif total_laps < 150:
                gf_min_len = 5
            else:
                gf_min_len = 7

            return slow_thresh, gf_slow_plus, gf_min_slow, gf_min_len

        # Calculate smart thresholds
        df_r = normalize_columns_to_str(st.session_state.raw_df.copy())
        visible_cars = [str(c) for c in df_r.columns if str(c) in st.session_state.car_visible]

        slow_thresh, gf_slow_plus, gf_min_slow, gf_min_len = calculate_smart_thresholds(df_r, visible_cars)

        st.session_state.slow_thresh = slow_thresh
        st.session_state.gf_slow_plus = gf_slow_plus
        st.session_state.gf_min_slow = gf_min_slow
        st.session_state.gf_min_len = gf_min_len

        # --------- silent auto-detect whenever visible cars change ---------
        if "_last_visible_cars_fp" not in st.session_state:
            st.session_state._last_visible_cars_fp = None
        if "detected_stints" not in st.session_state:
            st.session_state.detected_stints = {}

        def _cars_fp(cars: List[str]) -> tuple:
            return tuple(sorted(map(str, cars)))

        df_r = normalize_columns_to_str(st.session_state.raw_df.copy())
        visible_cars = [str(c) for c in df_r.columns if str(c) in st.session_state.car_visible]
        cur_fp = _cars_fp(visible_cars)

        if not df_r.empty and cur_fp != st.session_state._last_visible_cars_fp:
            det = {}
            for car in visible_cars:
                stints = detect_green_flag_stints_per_car(
                    df=df_r,
                    car=car,
                    slow_plus_seconds=float(st.session_state.get("gf_slow_plus", 3.0)),
                    min_slow_len=int(st.session_state.get("gf_min_slow", 1)),
                    min_green_len=int(st.session_state.get("gf_min_len", 3)),
                )
                det[car] = stints
                if stints:
                    lengths = [e - s + 1 for (s, e) in stints]
                    best_idx = int(np.argmax(lengths))
                    st.session_state.car_ranges[car] = tuple(map(int, stints[best_idx]))
            st.session_state.detected_stints = det
            st.session_state._last_visible_cars_fp = cur_fp

        # --------- per-car stint picker (dropdowns) - default to FIRST stint ---------
        det_map = st.session_state.get("detected_stints", {})
        if visible_cars:
            st.markdown("**Choose a green-flag stint per car** *(select 'Custom' to enter manual lap range)*")
            cols_per_row = 4
            cars_sorted = sorted(visible_cars, key=_numeric_then_alpha)
            rows = (len(cars_sorted) + cols_per_row - 1) // cols_per_row
            idx = 0
            for _ in range(rows):
                row_cols = st.columns(cols_per_row)
                for col in row_cols:
                    if idx >= len(cars_sorted):
                        col.empty(); continue
                    car = cars_sorted[idx]; idx += 1

                    stints = det_map.get(car, [])
                    labels = [f"L{s}-L{e}" for (s, e) in stints]
                    opts = ["— Select —", "Custom"] + labels

                    # Determine default index based on stored selection
                    stint_key = f"stint_selection_{car}"
                    if stint_key not in st.session_state:
                        # Default to first stint if available
                        st.session_state[stint_key] = labels[0] if labels else "— Select —"

                    stored_selection = st.session_state[stint_key]
                    if stored_selection in opts:
                        default_index = opts.index(stored_selection)
                    elif stints:
                        default_index = 2  # First stint
                    else:
                        default_index = 0

                    choice = col.selectbox(f"Car {car}", opts, index=default_index, key=f"gf_pick_{car}")

                    # Update stored selection and car_ranges
                    if choice != stored_selection:
                        st.session_state[stint_key] = choice

                    if choice not in ["— Select —", "Custom"]:
                        # User selected a detected stint - set car_ranges
                        j = opts.index(choice) - 2  # -2 for "— Select —" and "Custom"
                        s_lap, e_lap = stints[j]
                        st.session_state.car_ranges[car] = (int(s_lap), int(e_lap))
                    elif choice == "— Select —":
                        st.session_state.car_ranges.pop(car, None)
                    # "Custom" - car_ranges will be set by manual lap range inputs
        else:
            st.info("Select at least one car to prepare stints.")

        # --------- Lap Times (selected cars) ---------
        with st.expander("Lap Times (selected cars)", expanded=False):
            cars_for_table = [str(c) for c in df_r.columns if str(c) in st.session_state.car_visible]
            if not df_r.empty and cars_for_table:
                cars_for_table = sorted(cars_for_table, key=_numeric_then_alpha)
                table = pd.DataFrame({"Lap": np.arange(1, df_r.shape[0] + 1, dtype=int)})
                for car in cars_for_table:
                    table[car] = pd.to_numeric(df_r[car], errors="coerce")

                slow_thr = float(st.session_state.get("slow_thresh", 4.0))
                med_by_car = {
                    car: float(pd.to_numeric(df_r[car], errors="coerce").median())
                    for car in cars_for_table
                }

                def _style_slow(dfX: pd.DataFrame):
                    sty = pd.DataFrame("", index=dfX.index, columns=dfX.columns)
                    for car in cars_for_table:
                        if car in dfX.columns:
                            slow_mask = dfX[car] > (med_by_car[car] + slow_thr)
                            nan_mask = dfX[car].isna()
                            sty.loc[slow_mask, car] = "background-color: #ffd6d6"
                            sty.loc[nan_mask, car] = "background-color: #f2f2f2; color: #888"
                    if "Lap" in dfX.columns:
                        sty["Lap"] = ""
                    return sty

                approx_rows = min(20, int(table.shape[0]))
                height = 40 + 28 * approx_rows
                st.dataframe(
                    table.style.apply(_style_slow, axis=None).format(precision=3),
                    width="stretch",
                    height=height,
                    hide_index=True,
                )
            else:
                st.info("Select at least one car to view lap times.")

        # --------- Manual lap ranges ---------
        with st.expander("Manual lap ranges", expanded=False):
            if not df_r.empty and visible_cars:
                st.caption("Select **'Custom'** in the stint dropdown above to enable manual lap range entry.")
                max_laps = int(df_r.shape[0])
                cars_sorted = sorted(visible_cars, key=_numeric_then_alpha)

                cols_per_row = 3
                rows = (len(cars_sorted) + cols_per_row - 1) // cols_per_row
                idx = 0
                for _ in range(rows):
                    row_cols = st.columns(cols_per_row)
                    for col in row_cols:
                        if idx >= len(cars_sorted):
                            col.empty(); continue
                        car = cars_sorted[idx]; idx += 1

                        # Check if Custom is selected for this car
                        stint_key = f"stint_selection_{car}"
                        is_custom = st.session_state.get(stint_key) == "Custom"

                        # Initialize session state keys if not present
                        if f"manual_start_{car}" not in st.session_state:
                            cur_s, cur_e = st.session_state.car_ranges.get(car, (1, max_laps))
                            cur_s = int(max(1, min(cur_s, max_laps)))
                            cur_e = int(max(1, min(cur_e, max_laps)))
                            st.session_state[f"manual_start_{car}"] = cur_s
                            st.session_state[f"manual_end_{car}"] = cur_e

                        # Show current range or allow editing if Custom
                        if is_custom:
                            s_val = col.number_input(
                                f"Car {car} — Start Lap",
                                min_value=1,
                                max_value=max_laps,
                                step=1,
                                key=f"manual_start_{car}",
                            )
                            e_val = col.number_input(
                                f"Car {car} — End Lap",
                                min_value=1,
                                max_value=max_laps,
                                step=1,
                                key=f"manual_end_{car}",
                            )
                            if e_val < s_val:
                                e_val = s_val
                                st.session_state[f"manual_end_{car}"] = s_val

                            st.session_state.car_ranges[car] = (int(s_val), int(e_val))
                        else:
                            # Disabled - show current values from stint selection
                            cur_s, cur_e = st.session_state.car_ranges.get(car, (1, max_laps))
                            col.number_input(
                                f"Car {car} — Start Lap",
                                value=int(cur_s),
                                disabled=True,
                                key=f"manual_start_disabled_{car}",
                            )
                            col.number_input(
                                f"Car {car} — End Lap",
                                value=int(cur_e),
                                disabled=True,
                                key=f"manual_end_disabled_{car}",
                            )
            else:
                st.info("Select at least one car to choose laps.")

        # --------- Pit Penalty Detection ---------
        with st.expander("Pit Penalty Analysis", expanded=False):
            if not df_r.empty and visible_cars:
                st.caption("Detects pit stops (exactly 2 consecutive slow laps surrounded by fast laps) and calculates pit time penalty.")

                def detect_pit_penalties_full_race(df_data: pd.DataFrame, car: str, stints: List[Tuple[int, int]]) -> List[dict]:
                    """Detect pit stops: exactly 2 consecutive slow laps surrounded by fast laps."""
                    if not stints:
                        return []

                    ser = pd.to_numeric(df_data[car], errors="coerce")
                    n = len(ser)

                    # Calculate median and threshold from green-flag data only
                    green_vals = []
                    for (s, e) in stints:
                        stint_data = ser.iloc[s-1:e].dropna()
                        green_vals.extend(stint_data.tolist())

                    if not green_vals:
                        return []

                    med = float(np.median(green_vals))
                    threshold = med + float(st.session_state.get("gf_slow_plus", 3.0))

                    penalties = []

                    # Search across the entire race for pit patterns
                    i = 0
                    while i < n - 3:  # Need at least lap before, 2 slow, 1 after
                        lap_before = ser.iloc[i]
                        lap1 = ser.iloc[i + 1]
                        lap2 = ser.iloc[i + 2]
                        lap_after = ser.iloc[i + 3] if i + 3 < n else None

                        # Check for pit pattern: fast, slow, slow, fast
                        # All laps must be valid and finite
                        if (pd.notna(lap_before) and np.isfinite(lap_before) and
                            pd.notna(lap1) and np.isfinite(lap1) and
                            pd.notna(lap2) and np.isfinite(lap2) and
                            pd.notna(lap_after) and np.isfinite(lap_after)):

                            # Check pattern: fast -> slow -> slow -> fast
                            is_pit_pattern = (
                                lap_before <= threshold and  # Fast before
                                lap1 > threshold and         # Slow (pit in)
                                lap2 > threshold and         # Slow (pit out)
                                lap_after <= threshold       # Fast after
                            )

                            if is_pit_pattern:
                                # Calculate penalty
                                slow_total = float(lap1) + float(lap2)
                                fast_total = float(lap_before) + float(lap_after)
                                penalty = slow_total - fast_total

                                penalties.append({
                                    "Pit Laps": f"{i + 2}-{i + 3}",  # 1-indexed (lap1, lap2)
                                    "Final Stint Lap (s)": round(float(lap_before), 2),
                                    "Pit In (s)": round(float(lap1), 2),
                                    "Pit Out (s)": round(float(lap2), 2),
                                    "First Stint Lap (s)": round(float(lap_after), 2),
                                    "Penalty (s)": round(penalty, 2)
                                })

                                # Skip past this pit stop
                                i += 3
                            else:
                                i += 1
                        else:
                            i += 1

                    return penalties

                # Get detected stints from session state
                det_map = st.session_state.get("detected_stints", {})

                # Detect penalties for all visible cars
                all_penalties = {}
                for car in sorted(visible_cars, key=_numeric_then_alpha):
                    car_stints = det_map.get(car, [])
                    if car_stints:
                        penalties = detect_pit_penalties_full_race(df_r, car, car_stints)
                        if penalties:
                            all_penalties[car] = penalties

                if all_penalties:
                    # Calculate overall average across all selected cars
                    all_penalty_values = []
                    for penalties in all_penalties.values():
                        all_penalty_values.extend([p["Penalty (s)"] for p in penalties])

                    if all_penalty_values:
                        overall_avg = sum(all_penalty_values) / len(all_penalty_values)
                        st.markdown(f"**Average Pit Penalty (All Selected Cars): {overall_avg:.2f}s** across {len(all_penalty_values)} total stop(s)")
                        st.markdown("<div style='margin: 0.25rem 0; border-top: 1px solid #e0e0e0;'></div>", unsafe_allow_html=True)

                    # Create scrollable container for per-car results with tighter spacing
                    container_html = "<div style='max-height: 400px; overflow-y: auto; padding-right: 10px;'>"
                    st.markdown(container_html, unsafe_allow_html=True)

                    # Display results per car
                    for car, penalties in all_penalties.items():
                        st.markdown(f"<p style='margin-bottom: 0.25rem; font-weight: bold;'>Car {car}</p>", unsafe_allow_html=True)
                        pen_df = pd.DataFrame(penalties)
                        st.dataframe(pen_df, hide_index=True, width="stretch")

                        # Show average penalty for this car
                        avg_penalty = sum(p["Penalty (s)"] for p in penalties) / len(penalties)
                        st.markdown(f"<p style='margin: 0.25rem 0 0.5rem 0; font-size: 0.875rem; color: #666;'>Average: {avg_penalty:.2f}s across {len(penalties)} stop(s)</p>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No pit stops detected. Pattern required: fast lap → 2 slow laps → fast lap.")
            else:
                st.info("Select at least one car to analyze pit penalties.")

        # ---------- Auto-recompute aligned averages when selections change ----------
        def _align_fp(visible_cars_: List[str], ranges: Dict[str, Tuple[int, int]], nrows: int) -> tuple:
            items = tuple((c, ranges.get(c, None)) for c in sorted(map(str, visible_cars_)))
            return (items, int(nrows))

        if "_last_align_fp" not in st.session_state:
            st.session_state._last_align_fp = None

        df_align = normalize_columns_to_str(st.session_state.raw_df.copy())
        visible_for_align = [str(c) for c in df_align.columns if str(c) in st.session_state.car_visible]
        cur_fp_align = _align_fp(visible_for_align, st.session_state.car_ranges, df_align.shape[0])

        if not df_align.empty and visible_for_align and cur_fp_align != st.session_state._last_align_fp:
            aligned_vals: List[Optional[float]] = []
            series_map: Dict[str, List[float]] = {}
            max_len = 0

            for c in visible_for_align:
                rng = st.session_state.car_ranges.get(c)
                if not rng:
                    continue
                s_l, e_l = rng
                ser = (
                    pd.to_numeric(df_align[c].iloc[int(s_l) - 1:int(e_l)], errors="coerce")
                    .dropna().astype(float).tolist()
                )
                if not ser:
                    continue
                series_map[c] = ser
                max_len = max(max_len, len(ser))

            for i in range(max_len):
                vals = [ser[i] for ser in series_map.values() if i < len(ser)]
                aligned_vals.append(float(np.mean(vals)) if vals else None)

            st.session_state.aligned_avgs = aligned_vals
            st.session_state._last_align_fp = cur_fp_align

        # Reduce vertical spacing before export section
        st.markdown("<div style='margin-top: -1rem;'></div>", unsafe_allow_html=True)

        # ============================ Export (Create New Model) ============================
        c_target, c_send = st.columns([1.6, 1], gap="small")

        with c_target:
            new_model_name = st.text_input(
                "Model name",
                value="",
                placeholder="Enter new model name (e.g., 'C1 Stage 1', 'P1R3')",
                key="new_model_name_export",
                label_visibility="collapsed"
            )

        with c_send:
            if st.button("Export to Data & Fitting", type="primary", key="btn_send_model_compact"):
                if not st.session_state.aligned_avgs:
                    st.warning("Nothing to export yet—pick stints/lap ranges to auto-compute.")
                elif not new_model_name.strip():
                    st.warning("Please enter a model name.")
                else:
                    target_name = new_model_name.strip()
                    # filter out None
                    clean = [v for v in st.session_state.aligned_avgs if v is not None]
                    if not clean:
                        st.warning("Aligned averages are empty after filtering.")
                    else:
                        # Check if model already exists
                        if target_name in st.session_state.model_table.columns:
                            # Model exists, trigger confirmation dialog
                            st.session_state._pending_export = {
                                "clean": clean,
                                "target": target_name
                            }
                        else:
                            # New model, create and export directly
                            needed = len(clean)
                            # Create new column
                            st.session_state.model_table[target_name] = pd.Series(dtype=float)
                            # Ensure enough rows
                            cur_rows = st.session_state.model_table.shape[0]
                            if needed > cur_rows:
                                extra = pd.DataFrame(
                                    [[None] * len(st.session_state.model_table.columns)]
                                    * (needed - cur_rows),
                                    columns=st.session_state.model_table.columns,
                                )
                                st.session_state.model_table = pd.concat(
                                    [st.session_state.model_table, extra], ignore_index=True
                                )
                            # Set the data
                            st.session_state.model_table.loc[:needed - 1, target_name] = [
                                round(float(v), 3) for v in clean
                            ]
                            # Assign color from palette (will be assigned by _ensure_color on first render)
                            st.success(f"Created and exported {needed} rows into new model '{target_name}'.")
                            st.rerun()

        # Confirmation dialog for overwriting existing model
        if st.session_state.get("_pending_export"):
            @st.dialog("Model Already Exists")
            def confirm_overwrite():
                st.warning(f"⚠️ The model **'{st.session_state._pending_export['target']}'** already exists.")
                st.write("Do you want to overwrite it with the new data?")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Overwrite", type="primary", width="stretch"):
                        clean = st.session_state._pending_export["clean"]
                        target_name = st.session_state._pending_export["target"]
                        needed = len(clean)
                        cur_rows = st.session_state.model_table.shape[0]
                        if needed > cur_rows:
                            extra = pd.DataFrame(
                                [[None] * len(st.session_state.model_table.columns)]
                                * (needed - cur_rows),
                                columns=st.session_state.model_table.columns,
                            )
                            st.session_state.model_table = pd.concat(
                                [st.session_state.model_table, extra], ignore_index=True
                            )
                        st.session_state.model_table.loc[:needed - 1, target_name] = [
                            round(float(v), 3) for v in clean
                        ]
                        st.session_state._pending_export = None
                        st.success(f"Overwritten model '{target_name}' with {needed} rows.")
                        st.rerun()
                with col2:
                    if st.button("Cancel", width="stretch"):
                        st.session_state._pending_export = None
                        st.rerun()

            confirm_overwrite()

        # ----------------- Aligned plot -----------------
        dfp = normalize_columns_to_str(st.session_state.raw_df.copy())
        visible_plot = [str(c) for c in dfp.columns if str(c) in st.session_state.car_visible]

        if st.session_state.aligned_avgs and not dfp.empty and visible_plot:
            rows_plot = []
            for c in sorted(visible_plot, key=_numeric_then_alpha):
                se = st.session_state.car_ranges.get(c)
                if not se:
                    continue
                s_l, e_l = se
                colser = (
                    pd.to_numeric(dfp[c].iloc[int(s_l) - 1:int(e_l)], errors="coerce")
                    .dropna().astype(float).tolist()
                )
                for j, y in enumerate(colser, start=1):
                    rows_plot.append({"Series": c, "Aligned Lap": j, "Lap Time": y})

            # add aligned average series (skip None)
            for j, y in enumerate(st.session_state.aligned_avgs, start=1):
                if y is None:
                    continue
                rows_plot.append({"Series": "Aligned Avg", "Aligned Lap": j, "Lap Time": y})

            if rows_plot:
                plot_df = pd.DataFrame(rows_plot)
                ordered = sorted(visible_plot, key=_numeric_then_alpha)
                domain = ordered + ["Aligned Avg"]
                range_colors = [
                    st.session_state.car_colors.get(c, DEFAULT_CAR_COLORS.get(c, "#1f77b4"))
                    for c in ordered
                ] + ["#000000"]

                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=True, clip=True)
                    .encode(
                        x=alt.X("Aligned Lap:Q",
                                axis=alt.Axis(title="Aligned Lap", tickMinStep=1, format="d"),
                                scale=alt.Scale(zero=False, nice=True)),
                        y=alt.Y("Lap Time:Q",
                                axis=alt.Axis(title="Lap Time"),
                                scale=alt.Scale(zero=False, nice=True)),
                        color=alt.Color("Series:N",
                                        scale=alt.Scale(domain=domain, range=range_colors),
                                        legend=alt.Legend(title="Series")),
                    )
                    .properties(height=450)
                )
                st.altair_chart(chart, width="stretch")
        else:
            st.caption("Pick stints or lap ranges to auto-compute and preview the curves.")

# ---------------- TAB 2 ----------------
with tab2:
    _hdr_c1, _hdr_c2 = st.columns([20, 1])
    with _hdr_c1:
        st.subheader("Data & Fitting")
    with _hdr_c2:
        st.button("↻", key="refresh_tab2", help="Refresh after loading workspace")

    if "model_table" not in st.session_state:
        st.session_state.model_table = pd.DataFrame()
    if "model_params" not in st.session_state:
        st.session_state.model_params = {}
    if "model_colors" not in st.session_state:
        st.session_state.model_colors = {}

    def _has_data(col: str) -> bool:
        s = pd.to_numeric(st.session_state.model_table[col], errors="coerce")
        return s.notna().any()

    active_cols = [c for c in st.session_state.model_table.columns if _has_data(c)]

    # Auto-fit any models that haven't been fitted yet
    for col in active_cols:
        if col not in st.session_state.model_params:
            y = pd.to_numeric(st.session_state.model_table[col], errors="coerce").dropna().values.astype(float)
            if len(y) >= 2:
                x = np.arange(1, len(y) + 1, dtype=float)
                try:
                    a, b, c = fit_falloff_linear_ls(x, y)
                    st.session_state.model_params[col] = (a, b, c)
                    # Store original fitted values for reset functionality
                    st.session_state.model_params_original[col] = (a, b, c)
                except Exception:
                    pass

    left, right = st.columns([2, 1], gap="large")

    # ========================= LEFT: Plot =========================
    with left:
        rows = []

        # Distinct color palette for models - visually distinct and appealing
        MODEL_PALETTE = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#17becf",  # cyan
            "#bcbd22",  # olive
            "#7f7f7f",  # gray
        ]

        def _ensure_color(name: str) -> str:
            if name not in st.session_state.model_colors:
                # Assign next color from palette based on how many models exist
                idx = len(st.session_state.model_colors) % len(MODEL_PALETTE)
                st.session_state.model_colors[name] = MODEL_PALETTE[idx]
            return st.session_state.model_colors[name]

        for model_name, params in st.session_state.model_params.items():
            if model_name not in st.session_state.model_table.columns:
                continue
            if model_name not in active_cols:
                continue
            # Skip if model is not visible
            if not st.session_state.model_visible.get(model_name, True):
                continue

            a, b, c_ = params
            series = pd.to_numeric(st.session_state.model_table[model_name], errors="coerce").dropna().values.astype(float)
            if len(series) < 2:
                continue
            x = np.arange(1, len(series) + 1, dtype=float)
            color = _ensure_color(model_name)

            for xi, yi in zip(x, series):
                rows.append({"Series": model_name, "Kind": "data", "x": float(xi), "y": float(yi), "color": color})

            # Check if this model is in linear mode
            use_linear = st.session_state.model_linear_mode.get(model_name, False)

            if use_linear and model_name in st.session_state.model_linear_params:
                # Use piecewise linear - extend to T4 if beyond data
                linear_params = st.session_state.model_linear_params[model_name]
                transitions = linear_params.get("transitions", [])
                slopes = linear_params.get("slopes", [])
                intercepts = linear_params.get("intercepts", [])

                # Check if slopes/intercepts match transitions (fix old workspace data)
                # With N transitions, we have N segments (T4 is end point, not boundary)
                expected_segments = len(transitions)
                if len(slopes) != expected_segments or len(intercepts) != expected_segments:
                    # Refit to fix mismatched data
                    slopes, intercepts = refit_linear_segments(x, series, transitions)
                    st.session_state.model_linear_params[model_name] = {
                        "transitions": transitions,
                        "slopes": slopes,
                        "intercepts": intercepts
                    }

                # Extend fit line to max of data length or last transition (T4)
                max_x = max(len(series), max(transitions) if transitions else len(series))
                fx = np.linspace(1, max_x, 300)
                fy = piecewise_linear_equation(fx, transitions, slopes, intercepts)

                # Add transition point markers
                for t in transitions:
                    t_y = piecewise_linear_equation(np.array([float(t)]), transitions, slopes, intercepts)[0]
                    rows.append({"Series": model_name, "Kind": "transition", "x": float(t), "y": float(t_y), "color": color})
            else:
                # Use falloff curve
                fx = np.linspace(1, len(series), 200)
                fy = falloff_equation(fx, a, b, c_)

            for xi, yi in zip(fx, fy):
                rows.append({"Series": model_name, "Kind": "fit", "x": float(xi), "y": float(yi), "color": color})

        if rows:
            chart_df = pd.DataFrame(rows)
            model_domain = sorted({r["Series"] for r in rows})
            color_map = {m: st.session_state.model_colors.get(m, "#1f77b4") for m in model_domain}
            model_range = [color_map[m] for m in model_domain]

            base = alt.Chart(chart_df).encode(
                x=alt.X("x:Q",
                        axis=alt.Axis(title="Lap #", tickMinStep=1, format="d"),
                        scale=alt.Scale(zero=False, nice=True)),
                y=alt.Y("y:Q",
                        axis=alt.Axis(title="Lap Time (s)"),
                        scale=alt.Scale(zero=False, nice=True)),
                color=alt.Color("Series:N",
                                legend=alt.Legend(title="Model"),
                                scale=alt.Scale(domain=model_domain, range=model_range)),
            )
            line = base.transform_filter(alt.datum.Kind == "fit").mark_line()
            pts  = base.transform_filter(alt.datum.Kind == "data").mark_point()
            # Transition point markers - larger diamonds
            trans_pts = base.transform_filter(alt.datum.Kind == "transition").mark_point(
                shape="diamond", size=150, filled=True, stroke="white", strokeWidth=1
            )
            st.altair_chart((line + pts + trans_pts).properties(height=540), width="stretch")
        else:
            st.info("Fit models to see a plot.")

    # ========================= RIGHT: Model Summary =========================
    with right:
        cols = active_cols
        if not cols:
            st.info("No models yet. Export data from Tab 1 to create models.")
        else:
            st.caption(f"**{len(cols)}** model(s) loaded")

            rename_ops = []
            delete_ops = []
            color_changed = False
            visibility_changed = False

            for name in cols:
                # Get model info
                series = pd.to_numeric(st.session_state.model_table[name], errors="coerce").dropna()
                n_laps = len(series)
                is_fitted = name in st.session_state.model_params
                color = st.session_state.model_colors.get(name, "#1f77b4")

                # Default visibility to True if not set
                if name not in st.session_state.model_visible:
                    st.session_state.model_visible[name] = True

                is_visible = st.session_state.model_visible.get(name, True)

                # Checkbox and expander on same row
                col_check, col_expand = st.columns([0.5, 9.5])
                with col_check:
                    new_visible = st.checkbox("Show", value=is_visible, key=f"visible_{name}", label_visibility="collapsed")
                    if new_visible != is_visible:
                        st.session_state.model_visible[name] = new_visible
                        visibility_changed = True

                with col_expand:
                    # Compact expandable model editor
                    with st.expander(name, expanded=False):
                        # Rename, color picker, and delete button on same row
                        col_name, col_color, col_delete = st.columns([5, 1, 1])
                        with col_name:
                            new_name = st.text_input("Model Name", value=name, key=f"rename_{name}", label_visibility="collapsed")
                            if new_name != name and new_name.strip() and new_name not in st.session_state.model_table.columns:
                                rename_ops.append((name, new_name.strip()))
                        with col_color:
                            new_color = st.color_picker("Color", value=color, key=f"color_{name}", label_visibility="collapsed")
                            if new_color != color:
                                st.session_state.model_colors[name] = new_color
                                color_changed = True
                        with col_delete:
                            if st.button("🗑️", key=f"delete_{name}", help="Delete model"):
                                delete_ops.append(name)

                        # Piecewise mode adjustment (if fitted)
                        if is_fitted:
                            # Piecewise mode toggle
                            use_linear = st.checkbox("Piecewise Mode",
                                                    value=st.session_state.model_linear_mode.get(name, False),
                                                    key=f"linear_mode_{name}",
                                                    help="Adjust degradation using piecewise linear segments")

                            if use_linear != st.session_state.model_linear_mode.get(name, False):
                                st.session_state.model_linear_mode[name] = use_linear
                                # When switching to piecewise mode, always fit fresh
                                if use_linear:
                                    series_data = pd.to_numeric(st.session_state.model_table[name], errors="coerce").dropna()
                                    x_data = np.arange(1, len(series_data) + 1, dtype=float)
                                    y_data = series_data.values.astype(float)
                                    transitions, slopes, intercepts = detect_transition_points(x_data, y_data)
                                    st.session_state.model_linear_params[name] = {
                                        "transitions": transitions,
                                        "slopes": slopes,
                                        "intercepts": intercepts
                                    }
                                st.rerun()

                            if use_linear:
                                # Linear mode controls
                                linear_params = st.session_state.model_linear_params.get(name, {})
                                if not linear_params:
                                    st.warning("No linear parameters detected yet.")
                                else:
                                    transitions = linear_params.get("transitions", [])
                                    slopes = linear_params.get("slopes", [])
                                    intercepts = linear_params.get("intercepts", [])

                                    # Get original data for refitting
                                    series_data = pd.to_numeric(st.session_state.model_table[name], errors="coerce").dropna()
                                    x_data = np.arange(1, len(series_data) + 1, dtype=float)
                                    y_data = series_data.values.astype(float)

                                    # Transition points in 2x2 grid with number inputs
                                    # T4 can extend beyond data for extrapolation (up to 150 laps)
                                    st.markdown("**Transition Points (Lap)** - *adjusts curve automatically*")
                                    new_transitions = []
                                    trans_rows = st.columns(2)
                                    for i, t in enumerate(transitions):
                                        with trans_rows[i % 2]:
                                            # T4 (last transition) can go beyond data for extrapolation
                                            max_val = 150 if i == len(transitions) - 1 else int(n_laps) - 1
                                            min_val = 2 if i == 0 else int(transitions[i-1]) + 1
                                            new_t = st.number_input(f"T{i+1}", value=int(t), min_value=min_val, max_value=max_val, step=1, key=f"trans_{name}_{i}")
                                            new_transitions.append(float(new_t))

                                    # Check if transitions changed - auto-refit if so
                                    transitions_changed = new_transitions != transitions
                                    if transitions_changed:
                                        # Auto-refit the linear segments to the data
                                        new_slopes, new_intercepts = refit_linear_segments(x_data, y_data, new_transitions)
                                        st.session_state.model_linear_params[name] = {
                                            "transitions": new_transitions,
                                            "slopes": new_slopes,
                                            "intercepts": new_intercepts
                                        }
                                        # Increment version to force slope widgets to reset
                                        ver_key = f"_slope_version_{name}"
                                        st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                                        st.rerun()

                                    # Slopes in 2x2 grid with number inputs (matching transition style)
                                    st.markdown("**Slopes (s/lap)** - *fine-tune if needed*")
                                    slope_rows = st.columns(2)
                                    new_slopes = []
                                    # Use version in key so widgets reset when transitions change
                                    slope_ver = st.session_state.get(f"_slope_version_{name}", 0)
                                    for i, s in enumerate(slopes):
                                        with slope_rows[i % 2]:
                                            new_s = st.number_input(f"Seg {i+1}", value=float(s), step=0.001, format="%.4f", key=f"slope_{name}_{i}_v{slope_ver}")
                                            new_slopes.append(new_s)

                                    # Check if slopes changed - recalculate intercepts for continuity
                                    if new_slopes != slopes:
                                        new_intercepts = [intercepts[0]]
                                        for j in range(1, len(new_slopes)):
                                            t = transitions[j-1]
                                            y_at_t = new_slopes[j-1] * t + new_intercepts[j-1]
                                            new_intercepts.append(y_at_t - new_slopes[j] * t)
                                        st.session_state.model_linear_params[name] = {
                                            "transitions": transitions,
                                            "slopes": new_slopes,
                                            "intercepts": new_intercepts
                                        }
                                        st.rerun()

            # Handle visibility changes
            if visibility_changed:
                st.rerun()

            # Handle color changes
            if color_changed:
                st.rerun()

            # Handle renames
            if rename_ops:
                for old, new in rename_ops:
                    if old in st.session_state.model_table.columns:
                        st.session_state.model_table.rename(columns={old: new}, inplace=True)
                    if old in st.session_state.model_colors:
                        st.session_state.model_colors[new] = st.session_state.model_colors.pop(old)
                    if old in st.session_state.model_params:
                        st.session_state.model_params[new] = st.session_state.model_params.pop(old)
                    if old in st.session_state.model_params_original:
                        st.session_state.model_params_original[new] = st.session_state.model_params_original.pop(old)
                    if old in st.session_state.model_linear_mode:
                        st.session_state.model_linear_mode[new] = st.session_state.model_linear_mode.pop(old)
                    if old in st.session_state.model_linear_params:
                        st.session_state.model_linear_params[new] = st.session_state.model_linear_params.pop(old)
                    if old in st.session_state.model_visible:
                        st.session_state.model_visible[new] = st.session_state.model_visible.pop(old)
                st.success("Model renamed.")
                st.rerun()

            # Handle deletes
            if delete_ops:
                for name in delete_ops:
                    if name in st.session_state.model_table.columns:
                        st.session_state.model_table.drop(columns=[name], inplace=True)
                    st.session_state.model_colors.pop(name, None)
                    st.session_state.model_params.pop(name, None)
                    st.session_state.model_params_original.pop(name, None)
                st.success(f"Deleted {len(delete_ops)} model(s).")
                st.rerun()

    # =============================== Lap Time Models (table) ===============================
    with st.expander("Lap Time Models (Data Table)", expanded=False):
        st.caption("View and edit lap times for each model. Models are automatically fitted when created.")
        st.session_state.model_table = st.data_editor(
            st.session_state.model_table,
            num_rows="dynamic",
            width="stretch",
        )

        # Refit button for manual edits
        if st.button("Refit All Models", key="btn_refit_all", help="Refit all models after manual edits"):
            st.session_state.model_params.clear()
            fitted = 0
            for col in active_cols:
                y = pd.to_numeric(st.session_state.model_table[col], errors="coerce").dropna().values.astype(float)
                if len(y) >= 2:
                    x = np.arange(1, len(y) + 1, dtype=float)
                    try:
                        a, b, c = fit_falloff_linear_ls(x, y)
                        st.session_state.model_params[col] = (a, b, c)
                        fitted += 1
                    except Exception:
                        pass
            st.success(f"Refitted {fitted} model(s).")
            st.rerun()

    # ----------------------------- Save / Load Workspace -----------------------------
    s1, s2 = st.columns(2)

    def _sanitize_ws_name(name: str) -> str:
        import re
        name = (name or "").strip()
        if not name:
            return "stint_workspace"
        safe = re.sub(r"[^A-Za-z0-9 _-]", "", name)
        safe = re.sub(r"\s+", "_", safe)
        return safe or "stint_workspace"

    with s1:
        st.markdown("#### Save workspace")
        default_ws = st.session_state.get("ws_name", "stint_workspace")
        ws_name_input = st.text_input("Workspace filename (no extension)", value=default_ws, key="ws_name_input")
        safe_name = _sanitize_ws_name(ws_name_input)
        st.session_state["ws_name"] = safe_name

        payload = {
            "tab1": {
                "car_visible": sorted(list(st.session_state.get("car_visible", set()))),
                "car_colors": st.session_state.get("car_colors", {}),
                "car_ranges": st.session_state.get("car_ranges", {}),
                "pit_laps": st.session_state.get("pit_laps", []),
                "slow_thresh": float(st.session_state.get("slow_thresh", 4.0)),
                "gf_slow_plus": float(st.session_state.get("gf_slow_plus", 3.0)),
                "gf_min_slow": int(st.session_state.get("gf_min_slow", 1)),
                "gf_min_len": int(st.session_state.get("gf_min_len", 3)),
            },
            "tab2": {
                "model_table": st.session_state.get("model_table", pd.DataFrame()).to_dict(orient="list"),
                "model_params": st.session_state.get("model_params", {}),
                "model_params_original": st.session_state.get("model_params_original", {}),
                "model_colors": st.session_state.get("model_colors", {}),
                "model_linear_mode": st.session_state.get("model_linear_mode", {}),
                "model_linear_params": st.session_state.get("model_linear_params", {}),
                "model_visible": st.session_state.get("model_visible", {}),
            },
            "tab3": {
                "strategies_tabs": st.session_state.get("strategies_tabs", []),
                "str_start_lap": int(st.session_state.get("str_start_lap", 1)),
                "str_end_lap": int(st.session_state.get("str_end_lap", 50)),
                "str_pit_time": float(st.session_state.get("str_pit_time", 31.0)),
            },
        }
        b = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
        st.download_button(
            label="Download JSON",
            data=b,
            file_name=f"{safe_name}.json",
            mime="application/json",
            width="stretch",
            key="download_ws_named",
        )

    with s2:
        st.markdown("#### Load workspace")
        up_ws = st.file_uploader("Choose a workspace (.json)", type=["json"], key="ws_up")
        if up_ws is not None:
            try:
                loaded = json.load(up_ws)
                # tab1
                t1 = loaded.get("tab1", {})
                st.session_state.car_visible = set(map(str, t1.get("car_visible", [])))
                st.session_state.car_colors = {str(k): v for k, v in t1.get("car_colors", {}).items()}
                st.session_state.car_ranges = {str(k): tuple(v) for k, v in t1.get("car_ranges", {}).items()}
                st.session_state.pit_laps = list(map(int, t1.get("pit_laps", [])))
                st.session_state.slow_thresh = float(t1.get("slow_thresh", 4.0))
                st.session_state.gf_slow_plus = float(t1.get("gf_slow_plus", 3.0))
                st.session_state.gf_min_slow = int(t1.get("gf_min_slow", 1))
                st.session_state.gf_min_len = int(t1.get("gf_min_len", 3))
                # tab2
                t2 = loaded.get("tab2", {})
                st.session_state.model_table = pd.DataFrame.from_dict(t2.get("model_table", {}))
                st.session_state.model_params = {str(k): tuple(v) for k, v in t2.get("model_params", {}).items()}
                st.session_state.model_params_original = {str(k): tuple(v) for k, v in t2.get("model_params_original", {}).items()}
                st.session_state.model_colors = {str(k): v for k, v in t2.get("model_colors", {}).items()}
                st.session_state.model_linear_mode = {str(k): v for k, v in t2.get("model_linear_mode", {}).items()}
                st.session_state.model_linear_params = {str(k): v for k, v in t2.get("model_linear_params", {}).items()}
                st.session_state.model_visible = {str(k): v for k, v in t2.get("model_visible", {}).items()}
                # tab3
                t3 = loaded.get("tab3", {})
                st.session_state.strategies_tabs = t3.get("strategies_tabs", st.session_state.strategies_tabs)
                st.session_state.str_start_lap = int(t3.get("str_start_lap", 1))
                st.session_state.str_end_lap = int(t3.get("str_end_lap", 50))
                st.session_state.str_pit_time = float(t3.get("str_pit_time", 31.0))

                import os
                base = os.path.splitext(getattr(up_ws, "name", "stint_workspace.json"))[0]
                st.session_state["ws_name"] = _sanitize_ws_name(base)

                st.success("Workspace loaded.")
            except Exception as e:
                st.error(f"Failed to load workspace: {e}")

# ---------------- TAB 3: Strategies + Crossover ----------------
with tab3:
    if "strategies_tabs" not in st.session_state or not st.session_state.strategies_tabs:
        st.session_state.strategies_tabs = [
            {"tab_name": "Stage 1", "strategies": []},
            {"tab_name": "Stage 2", "strategies": []},
            {"tab_name": "Stage 3", "strategies": []},
            {"tab_name": "Stage 4", "strategies": []},
            {"tab_name": "Crossover", "strategies": []},
        ]

    # ---------- utilities ----------
    def _tight_y(vals: np.ndarray) -> tuple[float, float]:
        if vals is None or len(vals) == 0 or np.all(np.isnan(vals)):
            return (-1.0, 1.0)
        s = pd.Series(vals).dropna()
        if s.empty:
            return (-1.0, 1.0)
        vmin, vmax = float(s.min()), float(s.max())
        if np.isclose(vmin, vmax):
            lo, hi = vmin - 1.0, vmax + 1.0
        else:
            pad = 0.05 * (vmax - vmin)
            lo, hi = vmin - pad, vmax + pad
        if 0 < lo < 0.5: lo = 0.0
        if -0.5 < hi < 0: hi = 0.0
        if lo >= hi: lo, hi = lo - 1.0, hi + 1.0
        return (lo, hi)

    def _robust_y_lims(vals: np.ndarray) -> tuple[float, float]:
        if len(vals) == 0 or np.all(np.isnan(vals)): return (-1.0, 1.0)
        s = pd.Series(vals).dropna()
        if s.empty: return (-1.0, 1.0)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 0:
            lo = float(q1 - 1.5 * iqr); hi = float(q3 + 1.5 * iqr)
        else:
            p5, p95 = s.quantile([0.05, 0.95])
            lo, hi = float(p5), float(p95)
            if hi - lo == 0:
                med = float(s.median()); lo, hi = med - 1.0, med + 1.0
        if lo > 0: lo = 0.0
        if hi < 0: hi = 0.0
        return (lo, hi)

    def _palette(n: int) -> list[str]:
        base = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]
        if n <= len(base): return base[:n]
        return (base * ((n // len(base)) + 1))[:n]

    def _per_lap_from_cum(cum: list[float]) -> np.ndarray:
        c = np.array(cum, dtype=float)
        return np.diff(np.insert(c, 0, 0.0))

    # ---------- STAGE TABS (1–4) ----------
    def render_stage_tab(tab_idx: int):
        models = list(st.session_state.model_params.keys())
        if not models:
            st.info("Fit at least one model in the Data & Fitting tab.")
            return

        tab_store = st.session_state.strategies_tabs[tab_idx]
        if "strategies" not in tab_store:
            tab_store["strategies"] = []
        strategies = tab_store["strategies"]

        # Always use model's fitted base time
        use_model_base = True
        base_default = 0.0  # Not used when use_model_base=True

        def _both_models_selected(pm, qm) -> bool:
            return (pm in st.session_state.model_params) and (qm in st.session_state.model_params)

        # Two-column layout: inputs on left, buttons on right
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            # Model selectors
            m1, m2 = st.columns(2)
            with m1:
                pre_model_default = st.selectbox("Pre-Stop Model", ["-- Select --"] + models, key=f"pre_model_{tab_idx}")
            with m2:
                post_model_default = st.selectbox("Post-Stop Model", ["-- Select --"] + models, key=f"post_model_{tab_idx}")

            # Lap range and pit time
            r1, r2, r3 = st.columns(3)
            with r1:
                st.session_state.str_start_lap = st.number_input(
                    "Start Lap", min_value=1, value=int(st.session_state.get("str_start_lap", 1)),
                    step=1, key=f"start_{tab_idx}"
                )
            with r2:
                st.session_state.str_end_lap = st.number_input(
                    "End Lap", min_value=1, value=int(st.session_state.get("str_end_lap", 100)),
                    step=1, key=f"end_{tab_idx}"
                )
            with r3:
                pit_time = st.number_input(
                    "Pit Time (s)", min_value=0.0,
                    value=st.session_state.global_pit_time if st.session_state.global_pit_time is not None else 0.0,
                    step=0.5, key=f"pit_time_{tab_idx}",
                    placeholder="Required"
                )
                if pit_time > 0:
                    st.session_state.global_pit_time = pit_time
                else:
                    st.session_state.global_pit_time = None

        s = int(st.session_state.str_start_lap)
        e = int(st.session_state.str_end_lap)
        pit_time_defined = st.session_state.global_pit_time is not None and st.session_state.global_pit_time > 0
        pit_time_default = float(st.session_state.global_pit_time) if pit_time_defined else 0.0

        with right_col:
            st.caption("Add Strategy")
            disabled = not pit_time_defined
            # Row 1: Even splits
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("No-Stop", key=f"add_ns_{tab_idx}", use_container_width=True, disabled=disabled):
                    if _both_models_selected(pre_model_default, post_model_default):
                        strategies.append({
                            "name": "No-Stop", "pit_stops": [],
                            "pre_model": pre_model_default, "post_model": post_model_default,
                            "total_time": 0.0, "cum_times": [],
                            "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                        })
                    else:
                        st.toast("Select valid Pre/Post models")
            with b2:
                if st.button("Even 1-Stop", key=f"add_even1_{tab_idx}", use_container_width=True, disabled=disabled):
                    if e > s and _both_models_selected(pre_model_default, post_model_default):
                        pit = s + (e - s) // 2
                        strategies.append({
                            "name": "Even 1-Stop", "pit_stops": [pit],
                            "pre_model": pre_model_default, "post_model": post_model_default,
                            "total_time": 0.0, "cum_times": [],
                            "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                        })
                    else:
                        st.toast("Select valid Pre/Post models")
            with b3:
                if st.button("Even 2-Stop", key=f"add_even2_{tab_idx}", use_container_width=True, disabled=disabled):
                    if e > s + 1 and _both_models_selected(pre_model_default, post_model_default):
                        L = e - s + 1
                        p1 = s + L // 3; p2 = s + (2 * L // 3)
                        if s < p1 < p2 < e:
                            strategies.append({
                                "name": "Even 2-Stop", "pit_stops": [p1, p2],
                                "pre_model": pre_model_default, "post_model": post_model_default,
                                "total_time": 0.0, "cum_times": [],
                                "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                            })
                        else:
                            st.toast("Window too short for 2 stops")
                    else:
                        st.toast("Select valid Pre/Post models")
            with b4:
                if st.button("Even 3-Stop", key=f"add_even3_{tab_idx}", use_container_width=True, disabled=disabled):
                    if e > s + 2 and _both_models_selected(pre_model_default, post_model_default):
                        L = e - s + 1
                        p1 = s + L // 4; p2 = s + L // 2; p3 = s + (3 * L // 4)
                        P = [p for p in [p1, p2, p3] if s < p < e]
                        if len(P) == 3 and P[0] < P[1] < P[2]:
                            strategies.append({
                                "name": "Even 3-Stop", "pit_stops": P,
                                "pre_model": pre_model_default, "post_model": post_model_default,
                                "total_time": 0.0, "cum_times": [],
                                "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                            })
                        else:
                            st.toast("Window too short for 3 stops")
                    else:
                        st.toast("Select valid Pre/Post models")

            # Row 2: Optimal + Custom
            b5, b6, b7 = st.columns(3)
            with b5:
                if st.button("Optimal 1-Stop", key=f"opt1_{tab_idx}", use_container_width=True, disabled=disabled):
                    if _both_models_selected(pre_model_default, post_model_default):
                        a1, b1_, c1_ = st.session_state.model_params[pre_model_default]
                        a2, b2_, c2_ = st.session_state.model_params[post_model_default]
                        best_time, best_pit = find_optimal_single_stop(s, e, pit_time_default, a1, b1_, c1_, a2, b2_, c2_, 0.0, use_model_base)
                        if best_pit is not None:
                            strategies.append({
                                "name": "Optimal 1-Stop", "pit_stops": [best_pit],
                                "pre_model": pre_model_default, "post_model": post_model_default,
                                "total_time": 0.0, "cum_times": [],
                                "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                            })
                        else:
                            st.toast("No valid 1-stop for this range")
                    else:
                        st.toast("Select valid Pre/Post models")
            with b6:
                if st.button("Optimal 2-Stop", key=f"opt2_{tab_idx}", use_container_width=True, disabled=disabled):
                    if _both_models_selected(pre_model_default, post_model_default):
                        a1, b1_, c1_ = st.session_state.model_params[pre_model_default]
                        a2, b2_, c2_ = st.session_state.model_params[post_model_default]
                        best_time, best_pits = find_optimal_two_stop(s, e, pit_time_default, a1, b1_, c1_, a2, b2_, c2_, 0.0, use_model_base)
                        if best_pits is not None:
                            strategies.append({
                                "name": "Optimal 2-Stop", "pit_stops": list(best_pits),
                                "pre_model": pre_model_default, "post_model": post_model_default,
                                "total_time": 0.0, "cum_times": [],
                                "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                            })
                        else:
                            st.toast("No valid 2-stop for this range")
                    else:
                        st.toast("Select valid Pre/Post models")
            with b7:
                with st.popover("Custom Stops", use_container_width=True, disabled=disabled):
                    st.caption("Enter pit laps (comma-separated)")
                    laps_str = st.text_input("Laps", value="", key=f"custom_laps_{tab_idx}", placeholder="34, 67")
                    if st.button("Add", key=f"add_custom_{tab_idx}"):
                        if not _both_models_selected(pre_model_default, post_model_default):
                            st.warning("Select valid Pre/Post models")
                        else:
                            try:
                                raw = [t.strip() for t in laps_str.split(",") if t.strip()]
                                laps = sorted({int(x) for x in raw})
                                laps = [p for p in laps if s < p < e]
                                if not laps:
                                    st.warning("No valid laps in range")
                                else:
                                    strategies.append({
                                        "name": "Custom Stops", "pit_stops": laps,
                                        "pre_model": pre_model_default, "post_model": post_model_default,
                                        "total_time": 0.0, "cum_times": [],
                                        "start_lap": s, "end_lap": e, "pit_time": pit_time_default,
                                    })
                            except Exception:
                                st.warning("Invalid format")


        st.markdown("---")

        # ---- compute totals per row (using bound models/params)
        if strategies:
            totals = []
            for i, stg in enumerate(strategies):
                pre = stg.get("pre_model"); post = stg.get("post_model")
                if pre not in st.session_state.model_params or post not in st.session_state.model_params:
                    stg["total_time"] = float("nan"); stg["cum_times"] = []; continue
                a1, b1, c1_ = st.session_state.model_params[pre]
                a2, b2, c2_ = st.session_state.model_params[post]
                s_row = int(stg.get("start_lap", s)); e_row = int(stg.get("end_lap", e))
                pit_row = float(stg.get("pit_time", pit_time_default))

                # Check if models are using linear mode
                use_linear1 = st.session_state.model_linear_mode.get(pre, False)
                linear_params1 = st.session_state.model_linear_params.get(pre, None) if use_linear1 else None
                use_linear2 = st.session_state.model_linear_mode.get(post, False)
                linear_params2 = st.session_state.model_linear_params.get(post, None) if use_linear2 else None

                tot, cum = compute_total_time_and_laps_dual_model(
                    stg["pit_stops"], s_row, e_row, pit_row, a1, b1, c1_, a2, b2, c2_, 0.0, use_model_base,
                    pre, post, use_linear1, linear_params1, use_linear2, linear_params2
                )
                stg["total_time"] = float(tot)
                stg["cum_times"] = list(map(float, cum))
                totals.append(stg["total_time"])

            best_time = (min(totals) if totals else None)

            # build a stable color list keyed by row
            colors = _palette(len(strategies))
            series_keys = [f"{stg['name']} | {stg.get('pre_model','?')} | {stg.get('post_model','?')} | #{i}"
                           for i, stg in enumerate(strategies)]

            # ===== LEFT (table + plot) | RIGHT (what-if) =====
            left, right = st.columns([2, 1], gap="large")

            # ---------- table ----------
            with left:
                # Initialize visibility
                for stg in strategies:
                    if "visible" not in stg:
                        stg["visible"] = True

                # Build editable dataframe
                # Color squares matching _palette order: green, orange, blue, red, purple, brown, pink, gray, olive, cyan
                color_squares = ["🟩", "🟧", "🟦", "🟥", "🟪", "🟫", "⬛", "⬜", "🟨", "🔷"]
                df_rows = []
                for i, stg in enumerate(strategies):
                    delta = (stg["total_time"] - best_time) if (best_time is not None and np.isfinite(stg["total_time"])) else None
                    df_rows.append({
                        "Show": stg.get("visible", True),
                        "Color": color_squares[i % len(color_squares)],
                        "Strategy": stg["name"],
                        "Pre | Post": f"{stg.get('pre_model','?')} | {stg.get('post_model','?')}",
                        "Pit Stops": str(stg["pit_stops"]),
                        "Total (s)": round(stg["total_time"], 1) if np.isfinite(stg["total_time"]) else None,
                        "Δ (s)": round(delta, 1) if delta is not None else None,
                        "Delete": False,
                    })

                if df_rows:
                    df = pd.DataFrame(df_rows)
                    edited_df = st.data_editor(
                        df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Show": st.column_config.CheckboxColumn("View", help="Show on plot", default=True, width="small"),
                            "Color": st.column_config.TextColumn("", disabled=True),
                            "Strategy": st.column_config.TextColumn("Strategy", disabled=True),
                            "Pre | Post": st.column_config.TextColumn("Pre | Post", disabled=True),
                            "Pit Stops": st.column_config.TextColumn("Pit Stops", disabled=True),
                            "Total (s)": st.column_config.NumberColumn("Total (s)", disabled=True, format="%.1f"),
                            "Δ (s)": st.column_config.NumberColumn("Δ (s)", disabled=True, format="%.1f"),
                            "Delete": st.column_config.CheckboxColumn("Delete", help="Mark for deletion", default=False, width="small"),
                        },
                        column_order=["Show", "Color", "Strategy", "Pre | Post", "Pit Stops", "Total (s)", "Δ (s)", "Delete"],
                        key=f"strat_editor_{tab_idx}",
                    )

                    # Update visibility
                    for i, stg in enumerate(strategies):
                        if i < len(edited_df):
                            stg["visible"] = bool(edited_df.iloc[i]["Show"])

                    # Show delete button only if rows are marked
                    delete_indices = [i for i in range(len(edited_df)) if edited_df.iloc[i]["Delete"]]
                    if delete_indices:
                        num = len(delete_indices)
                        if st.button(f"Confirm Delete ({num})", key=f"confirm_del_{tab_idx}", type="primary"):
                            for idx in sorted(delete_indices, reverse=True):
                                del st.session_state.strategies_tabs[tab_idx]["strategies"][idx]
                            st.rerun()

                # ---------- Δ vs best plot (colors per row; no cross-connecting) ----------
                drows = []
                # Filter to visible strategies only
                visible_strategies = [(i, stg) for i, stg in enumerate(strategies) if stg.get("visible", True)]
                if visible_strategies:
                    # choose baseline = min total_time among visible
                    visible_times = [stg["total_time"] for _, stg in visible_strategies]
                    best_vis_idx = int(np.argmin(visible_times))
                    _, sb = visible_strategies[best_vis_idx]
                    sb_s = int(sb.get("start_lap", s)); sb_e = int(sb.get("end_lap", e))
                    base_x = np.arange(sb_s, sb_e + 1, dtype=int)
                    base_cum = np.array(sb["cum_times"], dtype=float)

                    for i, stg in visible_strategies:
                        s_row = int(stg.get("start_lap", s)); e_row = int(stg.get("end_lap", e))
                        x_vals = np.arange(s_row, e_row + 1, dtype=int)
                        cum = np.array(stg["cum_times"], dtype=float)

                        lo = max(s_row, sb_s); hi = min(e_row, sb_e)
                        if lo > hi:  # no overlap -> skip
                            continue
                        ia = (x_vals >= lo) & (x_vals <= hi)
                        ib = (base_x >= lo) & (base_x <= hi)
                        diff = cum[ia] - base_cum[ib]

                        key = series_keys[i]
                        for xv, dv in zip(np.arange(lo, hi + 1, dtype=int), diff):
                            drows.append({"Lap": int(xv), "Δ vs Best (s)": float(dv), "SeriesKey": key})

                if drows:
                    ddf = pd.DataFrame(drows)
                    lo, hi = _tight_y(ddf["Δ vs Best (s)"].values)
                    color_scale = alt.Scale(domain=series_keys, range=colors)
                    ch = (
                        alt.Chart(ddf)
                        .mark_line(clip=True)
                        .encode(
                            x=alt.X("Lap:Q",
                                    axis=alt.Axis(title="Lap", tickMinStep=1, format="d"),
                                    scale=alt.Scale(zero=False, nice=True)),
                            y=alt.Y("Δ vs Best (s):Q",
                                    axis=alt.Axis(title="Δ (s)"),
                                    scale=alt.Scale(domain=[lo, hi], zero=False, nice=True)),
                            color=alt.Color("SeriesKey:N", scale=color_scale, legend=None),
                            detail="SeriesKey:N"
                        )
                    ).properties(height=340)
                    st.altair_chart(ch, width="stretch")

            # ---------- right: early pit what-if (uses current dropdown models) ----------
            with right:
                if pre_model_default not in st.session_state.model_params or post_model_default not in st.session_state.model_params:
                    st.info("Select valid Pre/Post models to run the what-if.")
                else:
                    a1, b1, c1_ = st.session_state.model_params[pre_model_default]
                    a2, b2, c2_ = st.session_state.model_params[post_model_default]
                    desired_default = min(max(s + 2, s + (e - s) // 2), e - 1) if (e - s + 1) >= 3 else s + 1
                    desired = st.number_input("Desired pit lap",
                                              min_value=s + 1, max_value=max(s + 1, e - 1),
                                              value=min(desired_default, e - 1),
                                              step=1, key=f"whatif_{tab_idx}")
                    base_total, _ = compute_total_time_and_laps_dual_model(
                        [int(desired)], s, e, pit_time_default, a1, b1, c1_, a2, b2, c2_, base_default, use_model_base
                    )
                    rows = []
                    # Offsets: -15, -10, -5, -3, -1, 0, +1, +3, +5, +10, +15
                    offsets = [-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15]
                    even_idx = None
                    for offset in offsets:
                        pitlap = int(desired) + offset
                        if pitlap <= s or pitlap >= e: continue
                        tot, _ = compute_total_time_and_laps_dual_model(
                            [pitlap], s, e, pit_time_default, a1, b1, c1_, a2, b2, c2_, base_default, use_model_base
                        )
                        final_delta = tot - base_total
                        # Calculate initial gain (laps between pit and desired)
                        init_gain = 0.0
                        if offset < 0:  # early pit
                            for L in range(pitlap + 1, int(desired) + 1):
                                lt_base = compute_effective_lap_time(L - s + 1, a1, b1, c1_, base_default, use_model_base)
                                lt_early = compute_effective_lap_time(L - pitlap, a2, b2, c2_, base_default, use_model_base)
                                init_gain += (lt_base - lt_early)
                        elif offset > 0:  # late pit
                            for L in range(int(desired) + 1, pitlap + 1):
                                lt_base = compute_effective_lap_time(L - s + 1, a1, b1, c1_, base_default, use_model_base)
                                lt_late = compute_effective_lap_time(L - int(desired), a2, b2, c2_, base_default, use_model_base)
                                init_gain += (lt_late - lt_base)
                        label = f"{offset:+d}" if offset != 0 else "0"
                        if offset == 0:
                            even_idx = len(rows)
                        rows.append({"Early/Late": label, "Pit Lap": pitlap,
                                     "Initial Δ (s)": round(float(-init_gain), 2),
                                     "Final Δ (s)": round(float(final_delta), 2)})
                    if rows:
                        wdf = pd.DataFrame(rows)
                        def highlight_even(row):
                            if row.name == even_idx:
                                return ['font-weight: bold'] * len(row)
                            return [''] * len(row)
                        styled = wdf.style.apply(highlight_even, axis=1).format({"Initial Δ (s)": "{:.2f}", "Final Δ (s)": "{:.2f}"})
                        # Calculate height to fit all rows without scrolling (header + rows * ~35px per row)
                        table_height = (len(rows) + 1) * 35 + 3
                        st.dataframe(styled, hide_index=True, use_container_width=True, height=table_height)
                    else:
                        st.info("Adjust desired lap for valid range.")
        else:
            st.info("Add strategies to see results.")

        # ========= strategy pace comparison (instantaneous) =========
        st.markdown("---")
        st.markdown("### Strategy Pace Comparison (instantaneous lap Δ)")
        strategies = st.session_state.strategies_tabs[tab_idx]["strategies"]
        if len(strategies) >= 2:
            names = [f"{i}: {stg['name']} | {stg.get('pre_model','?')}→{stg.get('post_model','?')}" for i, stg in enumerate(strategies)]
            cA, cB = st.columns(2)
            with cA:
                ia = st.selectbox("Compare A", list(range(len(names))), format_func=lambda i: names[i],
                                  index=0, key=f"instA_{tab_idx}")
            with cB:
                ib = st.selectbox("Compare B", list(range(len(names))), format_func=lambda i: names[i],
                                  index=1 if len(names) > 1 else 0, key=f"instB_{tab_idx}")
            A, B = strategies[ia], strategies[ib]

            a_laps = _per_lap_from_cum(A.get("cum_times", []))
            b_laps = _per_lap_from_cum(B.get("cum_times", []))
            sA, eA = int(A.get("start_lap", 1)), int(A.get("end_lap", 1))
            sB, eB = int(B.get("start_lap", 1)), int(B.get("end_lap", 1))
            lo = max(sA, sB); hi = min(eA, eB)
            if lo <= hi:
                xA = np.arange(sA, eA + 1, dtype=int)
                xB = np.arange(sB, eB + 1, dtype=int)
                idxA = (xA >= lo) & (xA <= hi)
                idxB = (xB >= lo) & (xB <= hi)
                x = np.arange(lo, hi + 1, dtype=int)
                inst_delta = a_laps[idxA] - b_laps[idxB]
                lo2, hi2 = _robust_y_lims(inst_delta)
                df_inst = pd.DataFrame({"Lap": x, "Δ Lap Time (A − B) (s)": inst_delta})
                zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[3, 3]).encode(y="y:Q")
                ch = (
                    alt.Chart(df_inst)
                    .mark_line(clip=True)
                    .encode(
                        x=alt.X("Lap:Q", axis=alt.Axis(title="Lap", tickMinStep=1, format="d"),
                                scale=alt.Scale(zero=False, nice=True)),
                        y=alt.Y("Δ Lap Time (A − B) (s):Q",
                                axis=alt.Axis(title="Δ Lap T (s)"),
                                scale=alt.Scale(domain=[lo2, hi2], zero=False, nice=True)),
                        tooltip=[alt.Tooltip("Lap:Q", format="d"),
                                 alt.Tooltip("Δ Lap Time (A − B) (s):Q", format=".3f")],
                    )
                ).properties(height=240)
                st.altair_chart(ch + zero, width="stretch")
            else:
                st.info("Chosen strategies have no overlapping lap window.")
        else:
            st.info("Add at least two strategies to compare instantaneous pace.")

    # ---------- CROSSOVER TAB ----------
    def render_crossover_tab():
        models = list(st.session_state.model_params.keys())
        if not models:
            st.info("Fit at least one model in the Data & Fitting tab.")
            return
        c0, c1 = st.columns(2)
        with c0:
            pre_model = st.selectbox("Model (Pre-Stop)", ["-- Select --"] + models, key="co_pre_model")
        with c1:
            post_model = st.selectbox("Model (Post-Stop)", ["-- Select --"] + models, key="co_post_model")

        c2, c3 = st.columns([1, 1], gap="small")
        with c2:
            pit_time = st.number_input("Pit Time (s)", min_value=0.0,
                                       value=st.session_state.global_pit_time if st.session_state.global_pit_time is not None else 0.0,
                                       step=0.5, key="pit_time_crossover",
                                       placeholder="Required")
            # Only update if user has entered a value > 0
            if pit_time > 0:
                st.session_state.global_pit_time = pit_time
            else:
                st.session_state.global_pit_time = None
        with c3:
            minL = st.number_input("Min stint length (laps)", min_value=6, value=40, step=1, key="co_minL")

        # Always use model's fitted base time
        use_model_base_co = True

        if pre_model not in st.session_state.model_params or post_model not in st.session_state.model_params:
            st.info("Select valid Pre/Post models."); return

        a1, b1, c1_ = st.session_state.model_params[pre_model]
        a2, b2, c2_ = st.session_state.model_params[post_model]

        def _total_ns(L: int) -> float:
            total, _ = compute_total_time_and_laps_dual_model([], 1, L, 0.0, a1, b1, c1_, a2, b2, c2_, 0.0, use_model_base_co)
            return float(total)
        def _total_even(L: int) -> float:
            if L < 2: return float("inf")
            pit = min(max(2, 1 + (L // 2)), L - 1)
            total, _ = compute_total_time_and_laps_dual_model([pit], 1, L, float(pit_time), a1, b1, c1_, a2, b2, c2_, 0.0, use_model_base_co)
            return float(total)

        maxL = max(int(minL) + 10, int(minL) * 3)
        lengths = list(range(int(minL), int(maxL) + 1))
        deltas = [_total_ns(L) - _total_even(L) for L in lengths]
        best_idx = int(np.argmin(np.abs(deltas))) if deltas else None
        bestL = lengths[best_idx] if best_idx is not None else None

        sL, sR = st.columns([1, 1])
        with sL:
            if bestL is not None:
                st.subheader(f"Crossover ≈ **{bestL}** laps")
                st.caption("Where No-Stop and Even-Split are about the same total time.")
        with sR:
            if bestL is not None:
                st.metric("Total Time (No-Stop)", f"{_total_ns(bestL):.1f}s")
                st.metric("Total Time (Even-Split)", f"{_total_even(bestL):.1f}s")

        st.markdown("---")

        if bestL is not None:
            window = list(range(max(bestL - 8, lengths[0]), min(bestL + 8, lengths[-1]) + 1))
            wdf = pd.DataFrame([{
                "Stint L (laps)": L,
                "No-Stop (s)": round(_total_ns(L), 1),
                "Even-Split (s)": round(_total_even(L), 1),
                "Δ = NS − ES (s)": round(_total_ns(L) - _total_even(L), 2),
            } for L in window])
            st.dataframe(wdf, width="stretch", hide_index=True, height=320)

        if lengths:
            df = pd.DataFrame({"Laps": lengths, "Δ (NS − ES) (s)": deltas})
            lo, hi = _tight_y(np.array(deltas, dtype=float))
            zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[3, 3]).encode(y="y:Q")
            ch = (
                alt.Chart(df).mark_line(clip=True).encode(
                    x=alt.X("Laps:Q", axis=alt.Axis(title="Stint Length (laps)", tickMinStep=1, format="d")),
                    y=alt.Y("Δ (NS − ES) (s):Q",
                            axis=alt.Axis(title="Δ (s)"),
                            scale=alt.Scale(domain=[lo, hi], zero=False, nice=True)),
                    tooltip=[alt.Tooltip("Laps:Q", format="d"),
                             alt.Tooltip("Δ (NS − ES) (s):Q", format=".2f")],
                )
            ).properties(height=280, title="Crossover Curve (Δ = No-Stop − Even)")
            st.altair_chart(ch + zero, width="stretch")

    # ---------- build tabs ----------
    tab_names = [t.get("tab_name", f"Tab {i+1}") for i, t in enumerate(st.session_state.strategies_tabs)]
    tab_objs = st.tabs(tab_names)
    for i, tob in enumerate(tab_objs):
        with tob:
            if i < 4:
                render_stage_tab(i)
            else:
                render_crossover_tab()

