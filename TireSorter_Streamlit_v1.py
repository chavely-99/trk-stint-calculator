#!/usr/bin/env python3
"""
TRK Tire Sorter - Streamlit App
Sorts tires into optimal 4-tire sets with user-defined priorities.
"""

import math
import random
import itertools
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_sortables import sort_items

# ============== CONFIG ==============
APP_TITLE = "TRK - Tire Sorter"
RESTARTS = 150
BEAM = 8

# Display name → internal key mapping for priorities
PRIORITY_NAME_MAP = {
    'Cross Weight': 'cross',
    'Shift Code': 'shift',
    'Date Code': 'date',
    'RR Rollout': 'rr_rollout',
    'Rate Preference': 'rate_pref',
}

# ============== PAGE SETUP ==============
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="TH_FullLogo_White.png",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: .6rem; }
h1 { margin-bottom: .25rem; }
[data-testid="stHeader"] { height: 2rem; background: transparent; }
div[role="tablist"] { margin-top: .25rem; }
div[data-testid="stDecoration"] { display: none; }

.car-header {
    color: #333;
    font-weight: bold;
    font-size: 18px;
    margin-bottom: 6px;
    text-align: center;
}
.car-stats {
    color: #444;
    font-size: 14px;
    text-align: center;
    margin-bottom: 8px;
    padding: 6px 10px;
    background: #f5f5f5;
    border-radius: 4px;
    font-weight: 500;
}
.tire-box {
    background: #fafafa;
    border-radius: 8px;
    padding: 10px;
    text-align: left;
    color: #333;
    font-size: 13px;
    border: 1px solid #ddd;
}
/* Oval: left side (neon pink) / right side (grey) */
.tire-box.left { border-left: 5px solid #FF13F0; background: #fff0fd; }
.tire-box.right { border-left: 5px solid #9E9E9E; background: #f5f5f5; }
/* Road Course: A-pool (amber) / non-A pool (teal) */
.tire-box.pool-a { border-left: 5px solid #F57C00; background: #fff8f0; }
.tire-box.pool-b { border-left: 5px solid #00897B; background: #f0faf9; }
.tire-corner { font-weight: bold; color: #555; font-size: 15px; text-transform: uppercase; text-align: center; margin-bottom: 4px; }
.tire-row { display: flex; justify-content: space-between; align-items: center; margin: 3px 0; }
.tire-label { font-size: 14px; color: #555; font-weight: 600; }
.tire-rollout { font-weight: bold; font-size: 20px; color: #1976d2; }
.tire-rate { font-weight: bold; font-size: 17px; color: #d32f2f; }
.tire-shift { font-size: 15px; color: #7b1fa2; font-weight: 600; }
.tire-date { font-size: 13px; color: #666; }

.swap-section {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

/* Compact sortable priority items */
div[data-testid="stVerticalBlockBorderWrapper"] .element-container div[data-testid="stMarkdownContainer"] p {
    font-size: 14px;
}
.info-row {
    display: flex; gap: 30px; align-items: baseline;
    font-size: 15px; color: #444; margin: 4px 0 8px 0;
}
.info-row b { color: #222; font-size: 16px; }

/* Thicker borders on set cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-width: 3px !important;
    border-color: #999 !important;
    border-radius: 10px !important;
}

/* Green Run button — distinct from priority drag items */
button[kind="primary"] {
    background-color: #2e7d32 !important;
    border-color: #2e7d32 !important;
}
button[kind="primary"]:hover {
    background-color: #1b5e20 !important;
    border-color: #1b5e20 !important;
}

/* Compact table view styling - Excel-like */
.compact-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9px;
    margin-bottom: 15px;
}
.compact-table th {
    background: #4CAF50;
    color: white;
    padding: 2px 4px;
    text-align: center;
    font-weight: bold;
    border: 1px solid #ccc;
    font-size: 9px;
}
.compact-table th.sub-header {
    background: #66BB6A;
    font-size: 8px;
    padding: 1px 2px;
}
.compact-table td {
    padding: 1px 3px;
    border: 1px solid #ccc;
    text-align: center;
    font-size: 8px;
    line-height: 1.2;
    vertical-align: middle;
}
.compact-table tbody tr:nth-child(4n+1),
.compact-table tbody tr:nth-child(4n+2) {
    background: #fff;
}
.compact-table tbody tr:nth-child(4n+3),
.compact-table tbody tr:nth-child(4n+4) {
    background: #f5f5f5;
}
.compact-table .set-col {
    font-weight: bold;
    background: #e0e0e0;
    font-size: 9px;
}
.compact-table .metric-col {
    font-weight: 600;
    font-size: 8px;
}
/* Tire cells - Excel style */
.compact-table .tire-cell {
    cursor: pointer;
    font-size: 8px;
    padding: 1px 3px;
}
.compact-table .tire-cell:hover {
    background: #fff9c4 !important;
}
.compact-table .tire-cell.selected {
    background: #a5d6a7 !important;
    font-weight: bold;
}
.compact-table .tire-cell.left {
    border-left: 2px solid #FF13F0;
}
.compact-table .tire-cell.right {
    border-left: 2px solid #9E9E9E;
}
.compact-table .tire-cell.pool-a {
    border-left: 2px solid #F57C00;
}
.compact-table .tire-cell.pool-b {
    border-left: 2px solid #00897B;
}
</style>
""", unsafe_allow_html=True)

# ============== SESSION STATE ==============
SS_DEFAULTS = {
    'tire_df': None,
    'left_tires': None,
    'right_tires': None,
    'results': None,
    'stats': None,
    'available_dcodes': [],
    'ls_dcode': None,
    'track_type': 'Oval',
    'target_stagger': 25.0,
    'stagger_tolerance': 0,  # 0 = exact match required
    'cross_target': 0.50,
    'data_loaded': False,
    '_upload_token': None,
    # Non-negotiables (hard constraints)
    # Priority order (drag-and-drop, stagger is always #1)
    'priority_order': ['Cross Weight', 'Shift Code', 'Date Code', 'RR Rollout', 'Rate Preference'],
    'rate_preference': 'Softer Rear',  # 'Softer Rear', 'Softer Front', 'None'
    # Click-to-swap state
    'selected_tire': None,  # (set_idx, corner) or None
    'selected_set': None,   # set_idx or None (for swapping entire sets)
    # Feature enhancements
    'compact_view': False,  # Compact display for 10+ sets
    'duplicate_warnings': [],  # List of warning messages about duplicates
}

def ss_init():
    for k, v in SS_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

ss_init()


# ============== DATA FUNCTIONS ==============
def load_tire_data(file_source) -> pd.DataFrame:
    """Load tire data from Scan Data sheet (preferred) or TireScan fallback."""
    try:
        raw = pd.read_excel(file_source, sheet_name='Scan Data', header=3)

        # Scan Data has left + right sections with duplicate column names
        # (Number, Size, Date, Shift, Rate all appear twice).
        # Use positional access (iloc) to safely detect the right columns.
        col_names = [str(c) for c in raw.columns]

        # --- Build output df from the specific columns we need ---
        picks = {}  # target_name -> column_index

        # D-Code: first column named D-Code
        for i, c in enumerate(col_names):
            if c.lower().strip() == 'd-code':
                picks['D-Code'] = i
                break

        # Rate: prefer "Spring rate", fall back to first "Rate"
        for i, c in enumerate(col_names):
            if c.lower().strip() == 'spring rate':
                picks['Rate'] = i
                break
        if 'Rate' not in picks:
            for i, c in enumerate(col_names):
                if c.lower().strip() == 'rate':
                    picks['Rate'] = i
                    break

        # Shift: first column named Shift
        for i, c in enumerate(col_names):
            if c.lower().strip() == 'shift':
                picks['Shift'] = i
                break

        # Date Code: first column starting with "date"
        for i, c in enumerate(col_names):
            if c.lower().strip().startswith('date'):
                picks['Date Code'] = i
                break

        # Rollout/Dia: first "Size" or "Dia" column with numeric values > 100
        for i, c in enumerate(col_names):
            if 'size' not in c.lower() and 'dia' not in c.lower():
                continue
            try:
                vals = pd.to_numeric(raw.iloc[:, i], errors='coerce').dropna()
                if len(vals) > 0 and vals.min() > 100:
                    picks['Rollout/Dia'] = i
                    break
            except Exception:
                continue

        # Number: second "Number" column = tire ID (1-100).
        # First "Number" is the wheel serial. Pandas names them "Number", "Number.1".
        number_cols = [(i, c) for i, c in enumerate(col_names) if 'number' in c.lower()]
        if len(number_cols) >= 2:
            picks['Number'] = number_cols[1][0]  # second occurrence
        elif len(number_cols) == 1:
            picks['Number'] = number_cols[0][0]

        # Wheel: first column named "Wheel" (used for Road Course position assignment)
        for i, c in enumerate(col_names):
            if c.lower().strip() == 'wheel':
                picks['Wheel'] = i
                break

        # Build clean DataFrame from picked columns
        df = pd.DataFrame()
        for name, idx in picks.items():
            df[name] = raw.iloc[:, idx]

    except Exception as scan_err:
        # Fall back to TireScan sheet
        st.sidebar.warning(f"Scan Data sheet not found — using TireScan fallback.")
        df = pd.read_excel(file_source, sheet_name='TireScan')

    # --- Standard cleanup (works for both sheet sources) ---
    df = df[df['D-Code'].notna()]
    df['D-Code'] = pd.to_numeric(df['D-Code'], errors='coerce')
    df = df[df['D-Code'].notna()]
    df['D-Code'] = df['D-Code'].astype(int).astype(str).str.strip()
    df = df[df['D-Code'] != '']
    df = df[df['D-Code'] != ' ']

    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    df['Rollout/Dia'] = pd.to_numeric(df['Rollout/Dia'], errors='coerce')
    df['Shift'] = df['Shift'].fillna('').astype(str).str.strip()
    if 'Date Code' in df.columns:
        df['Date Code'] = df['Date Code'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        df['Date Code'] = df['Date Code'].replace({'nan': '', 'None': '', 'NaT': ''})
    if 'Number' in df.columns:
        df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    if 'Wheel' in df.columns:
        df['Wheel'] = df['Wheel'].astype(str).str.strip()

    df = df.dropna(subset=['Rate', 'Rollout/Dia'])
    df = df.reset_index(drop=True)

    return df


def detect_input_duplicates(df: pd.DataFrame) -> List[str]:
    """Detect duplicate tire Numbers in input data.

    Returns list of warning messages.
    """
    if 'Number' not in df.columns:
        return []

    warnings = []
    tire_numbers = df['Number'].dropna()

    # Find duplicates
    dup_mask = tire_numbers.duplicated(keep=False)
    if dup_mask.any():
        dup_numbers = tire_numbers[dup_mask].unique()
        warnings.append(f"⚠️ Duplicate tire numbers in input: {', '.join(map(str, map(int, dup_numbers)))}")

        # Detail: which tires are duplicated with their D-Codes
        for num in dup_numbers:
            dup_rows = df[df['Number'] == num]
            dcodes = dup_rows['D-Code'].tolist()
            warnings.append(f"  Tire #{int(num)} appears {len(dup_rows)} times (D-Codes: {', '.join(dcodes)})")

    return warnings


def detect_solution_duplicates(solution: List[dict]) -> List[str]:
    """Detect if same tire appears in multiple sets (should never happen).

    Returns list of warning messages.
    """
    warnings = []
    tire_numbers_used = []

    for set_idx, s in enumerate(solution):
        for corner in ['lf_data', 'rf_data', 'lr_data', 'rr_data']:
            tire = s[corner]
            if 'Number' in tire.index:
                num = int(tire['Number'])
                if num in tire_numbers_used:
                    warnings.append(f"❌ Tire #{num} appears in multiple sets!")
                else:
                    tire_numbers_used.append(num)

    return warnings


def assign_positions(df: pd.DataFrame, ls_dcode: str, track_type: str = 'Oval') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split tires into two pools based on track type.

    Oval: left pool (LS D-Code → LF/LR), right pool (other → RF/RR)
    Road Course: A pool (Wheel ends with A → LR/RF), non-A pool (→ LF/RR)

    For Road Course, returns (a_tires, non_a_tires) — the caller uses them
    with build_candidates_road() which maps A→LR/RF and non-A→LF/RR.
    """
    if track_type == 'Road Course' and 'Wheel' in df.columns:
        wheel_str = df['Wheel'].str.upper()
        a_tires = df[wheel_str.str.endswith('A')].copy()
        non_a_tires = df[~wheel_str.str.endswith('A')].copy()
        return a_tires, non_a_tires
    else:
        left_tires = df[df['D-Code'] == str(ls_dcode)].copy()
        right_tires = df[df['D-Code'] != str(ls_dcode)].copy()
        return left_tires, right_tires


def analyze_stagger_range(left_tires: pd.DataFrame, right_tires: pd.DataFrame) -> dict:
    """Calculate possible stagger ranges. Stagger = RR - LR."""
    if len(left_tires) == 0 or len(right_tires) == 0:
        return {'max_single': 0, 'min_single': 0, 'min_all_sets': 0, 'max_all_sets': 0, 'n_sets': 0}

    lr_rollouts = left_tires['Rollout/Dia'].values
    rr_rollouts = right_tires['Rollout/Dia'].values

    max_single = rr_rollouts.max() - lr_rollouts.min()
    min_single = rr_rollouts.min() - lr_rollouts.max()

    lr_sorted = np.sort(lr_rollouts)
    rr_sorted = np.sort(rr_rollouts)
    n_sets = min(len(lr_sorted), len(rr_sorted)) // 2

    if n_sets == 0:
        return {'max_single': max_single, 'min_single': min_single, 'min_all_sets': min_single, 'max_all_sets': max_single, 'n_sets': 0}

    staggers_max = rr_sorted[-n_sets:] - lr_sorted[:n_sets]
    max_achievable_all = staggers_max.min()

    staggers_min = rr_sorted[:n_sets] - lr_sorted[-n_sets:][::-1]
    min_achievable_all = staggers_min.max()

    return {
        'max_single': max_single,
        'min_single': min_single,
        'min_all_sets': min_achievable_all,
        'max_all_sets': max_achievable_all,
        'n_sets': n_sets
    }


# ============== OPTIMIZATION FUNCTIONS ==============
def compute_set_metrics(lf, rf, lr, rr) -> dict:
    """Compute all metrics for a 4-tire set."""
    stagger = rr['Rollout/Dia'] - lr['Rollout/Dia']  # rear stagger
    front_stagger = rf['Rollout/Dia'] - lf['Rollout/Dia']  # front stagger
    total_rate = lf['Rate'] + rf['Rate'] + lr['Rate'] + rr['Rate']
    cross = (rf['Rate'] + lr['Rate']) / total_rate

    # Shift code matching (1 = all same, higher = more different)
    shifts = [lf['Shift'], rf['Shift'], lr['Shift'], rr['Shift']]
    shifts_clean = [s for s in shifts if s and s != '']
    shift_score = len(set(shifts_clean)) if shifts_clean else 1

    # Date code matching - prioritize left side match and right side match separately
    def get_date(tire):
        d = tire.get('Date Code', '')
        return str(d).strip() if d and str(d) != 'nan' else ''

    left_dates = [get_date(lf), get_date(lr)]
    right_dates = [get_date(rf), get_date(rr)]

    # Score: 0 = both sides match, 1 = one side matches, 2 = neither matches
    left_match = 1 if (left_dates[0] and left_dates[1] and left_dates[0] == left_dates[1]) else 0
    right_match = 1 if (right_dates[0] and right_dates[1] and right_dates[0] == right_dates[1]) else 0
    date_score = 2 - (left_match + right_match)  # 0 = best (both match), 2 = worst

    # Rate preferences
    rear_avg = (lr['Rate'] + rr['Rate']) / 2
    front_avg = (lf['Rate'] + rf['Rate']) / 2
    rear_softer = rear_avg < front_avg
    front_softer = front_avg < rear_avg

    return {
        'stagger': stagger,
        'front_stagger': front_stagger,
        'cross': cross,
        'shift_score': shift_score,
        'date_score': date_score,
        'left_date_match': left_match,
        'right_date_match': right_match,
        'rear_avg_rate': rear_avg,
        'front_avg_rate': front_avg,
        'rear_softer': rear_softer,
        'front_softer': front_softer,
        'rr_rollout': rr['Rollout/Dia'],
    }


def build_candidates(left_tires: pd.DataFrame, right_tires: pd.DataFrame,
                     target_stagger: float = None, stagger_tolerance: float = None) -> List[dict]:
    """Build all valid 4-tire combinations, optionally filtered by stagger."""
    candidates = []

    left_list = list(left_tires.iterrows())
    right_list = list(right_tires.iterrows())

    for (lf_idx, lf), (lr_idx, lr) in itertools.permutations(left_list, 2):
        for (rf_idx, rf), (rr_idx, rr) in itertools.permutations(right_list, 2):
            metrics = compute_set_metrics(lf, rf, lr, rr)

            # Filter by stagger if exact stagger is required
            if target_stagger is not None and stagger_tolerance is not None:
                if abs(metrics['stagger'] - target_stagger) > stagger_tolerance:
                    continue

            candidates.append({
                'tires': (lf_idx, rf_idx, lr_idx, rr_idx),
                'tset': {lf_idx, rf_idx, lr_idx, rr_idx},
                'lf_data': lf,
                'rf_data': rf,
                'lr_data': lr,
                'rr_data': rr,
                **metrics
            })

    return candidates


def build_candidates_road(a_tires: pd.DataFrame, non_a_tires: pd.DataFrame,
                          target_stagger: float = None, stagger_tolerance: float = None) -> List[dict]:
    """Build valid 4-tire combos for Road Course.

    A tires (Wheel ends with A) → LR and RF positions (diagonal).
    Non-A tires → LF and RR positions (diagonal).
    """
    candidates = []

    a_list = list(a_tires.iterrows())
    non_a_list = list(non_a_tires.iterrows())

    # LF/RR from non-A pool, RF/LR from A pool
    for (lf_idx, lf), (rr_idx, rr) in itertools.permutations(non_a_list, 2):
        for (rf_idx, rf), (lr_idx, lr) in itertools.permutations(a_list, 2):
            metrics = compute_set_metrics(lf, rf, lr, rr)

            if target_stagger is not None and stagger_tolerance is not None:
                if abs(metrics['stagger'] - target_stagger) > stagger_tolerance:
                    continue

            candidates.append({
                'tires': (lf_idx, rf_idx, lr_idx, rr_idx),
                'tset': {lf_idx, rf_idx, lr_idx, rr_idx},
                'lf_data': lf,
                'rf_data': rf,
                'lr_data': lr,
                'rr_data': rr,
                **metrics
            })

    return candidates


def score_solution(solution: List[dict], target_stagger: float, cross_target: float,
                   priorities: dict, rate_pref: str,
                   road_course: bool = False) -> tuple:
    """Score a solution lexicographically — higher priorities are fully resolved
    before lower ones are even considered.

    Returns a tuple ordered by priority rank. Python's native tuple comparison
    is lexicographic, so (1, 99) < (2, 0) — priority 1 always wins.

    Road Course: cross weight consistency (spread) is weighted 5x heavier than
    absolute deviation from target, so the optimizer prefers all sets at 50.1%
    over 5 sets at 50.0% and one outlier at 50.5%.
    """
    if not solution:
        return (float('inf'),) * 6

    scores = {}

    staggers = [c['stagger'] for c in solution]
    stagger_dev = sum(abs(s - target_stagger) for s in staggers)
    stagger_spread = max(staggers) - min(staggers) if len(staggers) > 1 else 0
    scores['stagger'] = round(stagger_dev + stagger_spread, 4)

    crosses = [c['cross'] for c in solution]
    cross_dev = sum(abs(c - cross_target) * 100 for c in crosses)
    cross_spread = (max(crosses) - min(crosses)) * 100 if len(crosses) > 1 else 0
    # Road Course: consistency is paramount — penalize spread 5x more
    spread_w = 5.0 if road_course else 1.0
    scores['cross'] = round(cross_dev + cross_spread * spread_w, 4)

    scores['shift'] = sum(c['shift_score'] for c in solution)

    scores['date'] = sum(c.get('date_score', 0) for c in solution)

    rr_rollouts = [c.get('rr_rollout', 0) for c in solution]
    scores['rr_rollout'] = round(max(rr_rollouts) - min(rr_rollouts), 4) if len(rr_rollouts) > 1 else 0

    if rate_pref == 'Softer Rear':
        scores['rate_pref'] = sum(0 if c['rear_softer'] else 1 for c in solution)
    elif rate_pref == 'Softer Front':
        scores['rate_pref'] = sum(0 if c['front_softer'] else 1 for c in solution)
    else:
        scores['rate_pref'] = 0

    ordered = sorted(priorities.items(), key=lambda x: x[1])
    return tuple(scores.get(name, 0) for name, _ in ordered)


def randomized_greedy(candidates: List[dict], n_tires: int,
                      target_stagger: float, cross_target: float,
                      priorities: dict, rate_pref: str,
                      restarts: int = RESTARTS, beam: int = BEAM,
                      progress_callback=None,
                      road_course: bool = False) -> Optional[List[dict]]:
    """Find optimal tire assignment using randomized greedy beam search.

    Optimised with pre-indexed candidate filtering (boolean mask),
    incremental vectorised scoring via numpy, and early termination.

    Road Course: cross weight spread is weighted 5x to enforce consistency.
    """
    if not candidates:
        return None

    n_cands = len(candidates)

    # --- Pre-extract per-candidate metrics into numpy arrays ----------
    all_staggers   = np.array([c['stagger'] for c in candidates])
    all_crosses    = np.array([c['cross'] for c in candidates])
    all_shifts     = np.array([c['shift_score'] for c in candidates], dtype=np.float64)
    all_dates      = np.array([c.get('date_score', 0) for c in candidates], dtype=np.float64)
    all_rr         = np.array([c.get('rr_rollout', 0) for c in candidates], dtype=np.float64)

    if rate_pref == 'Softer Rear':
        all_rate = np.array([0.0 if c['rear_softer'] else 1.0 for c in candidates])
    elif rate_pref == 'Softer Front':
        all_rate = np.array([0.0 if c['front_softer'] else 1.0 for c in candidates])
    else:
        all_rate = np.zeros(n_cands)

    # --- Priority weights for composite score (lexicographic via floats) -
    ordered = sorted(priorities.items(), key=lambda x: x[1])
    priority_names = [name for name, _ in ordered]
    # Use 1e10, 1e8, … so the largest contribution stays well within float64
    priority_weights = {name: 10.0 ** (10 - rank * 2)
                        for rank, (name, _) in enumerate(ordered)}

    # --- Pre-build tire → candidate-index arrays for fast deactivation ----
    tire_to_cands: Dict[int, np.ndarray] = {}
    for i, c in enumerate(candidates):
        for tid in c['tset']:
            if tid not in tire_to_cands:
                tire_to_cands[tid] = []
            tire_to_cands[tid].append(i)
    tire_to_cands = {tid: np.array(idxs, dtype=np.intp)
                     for tid, idxs in tire_to_cands.items()}

    best_score = (float('inf'),) * 6
    best_solution = None

    for r in range(restarts):
        if progress_callback:
            progress_callback(r / restarts)

        # Boolean mask: True = candidate still usable this restart
        active = np.ones(n_cands, dtype=bool)
        solution: List[dict] = []

        # Running accumulators for incremental scoring
        sum_stag_dev = 0.0;  min_stag = np.inf;  max_stag = -np.inf
        sum_cross_dev = 0.0; min_cross = np.inf;  max_cross = -np.inf
        sum_shift = 0.0
        sum_date  = 0.0
        min_rr_val = np.inf; max_rr_val = -np.inf
        sum_rate  = 0.0

        while True:
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break

            # --- Incremental scores for every active candidate ----------
            s  = all_staggers[active_idx]
            c_ = all_crosses[active_idx]
            rr = all_rr[active_idx]

            stag_score  = (sum_stag_dev + np.abs(s - target_stagger)
                           + np.maximum(max_stag, s) - np.minimum(min_stag, s))
            # Road Course: spread weighted 5x to enforce cross consistency
            _sw = 5.0 if road_course else 1.0
            cross_score = (sum_cross_dev + np.abs(c_ - cross_target) * 100
                           + (np.maximum(max_cross, c_) - np.minimum(min_cross, c_)) * 100 * _sw)
            shift_score = sum_shift + all_shifts[active_idx]
            date_score  = sum_date  + all_dates[active_idx]
            rr_score    = np.maximum(max_rr_val, rr) - np.minimum(min_rr_val, rr)
            rate_score  = sum_rate  + all_rate[active_idx]

            # Fix spread terms for first candidate (inf - inf → 0)
            if not solution:
                stag_score  = np.abs(s - target_stagger)
                cross_score = np.abs(c_ - cross_target) * 100
                rr_score    = np.zeros_like(rr)

            score_map = {
                'stagger':   stag_score,
                'cross':     cross_score,
                'shift':     shift_score,
                'date':      date_score,
                'rr_rollout': rr_score,
                'rate_pref': rate_score,
            }

            # Composite float score (mimics lexicographic tuple comparison)
            composite = np.zeros(active_idx.size)
            for pname in priority_names:
                if pname in score_map:
                    composite += score_map[pname] * priority_weights[pname]

            # Tiny jitter for diversity across restarts
            composite += np.random.random(active_idx.size) * 0.001

            # Select from top-beam candidates
            if active_idx.size <= beam:
                top_local = np.arange(active_idx.size)
            else:
                top_local = np.argpartition(composite, beam)[:beam]

            choice_local  = top_local[np.random.randint(top_local.size)]
            choice_global = active_idx[choice_local]
            chosen = candidates[choice_global]

            solution.append(chosen)

            # Update running accumulators
            sv = chosen['stagger']
            sum_stag_dev += abs(sv - target_stagger)
            min_stag = min(min_stag, sv); max_stag = max(max_stag, sv)

            cv = chosen['cross']
            sum_cross_dev += abs(cv - cross_target) * 100
            min_cross = min(min_cross, cv); max_cross = max(max_cross, cv)

            sum_shift += chosen['shift_score']
            sum_date  += chosen.get('date_score', 0)

            rv = chosen.get('rr_rollout', 0)
            min_rr_val = min(min_rr_val, rv); max_rr_val = max(max_rr_val, rv)

            if rate_pref == 'Softer Rear':
                sum_rate += 0 if chosen['rear_softer'] else 1
            elif rate_pref == 'Softer Front':
                sum_rate += 0 if chosen['front_softer'] else 1

            # Deactivate all candidates sharing any tire with chosen set
            for tid in chosen['tset']:
                if tid in tire_to_cands:
                    active[tire_to_cands[tid]] = False

        # --- Evaluate complete solutions ---
        used = sum(len(s['tset']) for s in solution)
        if used == n_tires:
            sc = score_solution(solution, target_stagger, cross_target, priorities, rate_pref, road_course=road_course)
            if sc < best_score:
                best_score = sc
                best_solution = solution
                # Early termination when near-perfect
                if all(v < 0.001 for v in sc):
                    break

    if progress_callback:
        progress_callback(1.0)

    return best_solution


def polish_solution(solution: List[dict], priorities: dict,
                    cross_target: float, stagger_tolerance: float,
                    rate_pref: str, road_course: bool = False,
                    max_passes: int = 10) -> List[dict]:
    """Auto-refine the solution in priority order until no more swaps help.

    Runs each refine function from highest to lowest priority, repeating
    until a full pass finds no improvement (or max_passes reached).
    """
    if not solution:
        return solution

    stag_tol = stagger_tolerance if stagger_tolerance > 0 else 0.5
    rc = road_course

    # Map priority keys to their refine functions
    # (skip 'stagger' — handled by candidate filtering, not swaps)
    refine_map = {
        'cross':     lambda sol: refine_cross(sol, cross_target, stag_tol, road_course=rc),
        'shift':     lambda sol: refine_shift(sol, cross_target, stag_tol, road_course=rc),
        'date':      lambda sol: refine_date(sol, stag_tol, road_course=rc),
        'rr_rollout': lambda sol: refine_rr_rollout(sol, stag_tol, road_course=rc),
        'rate_pref': lambda sol: refine_rate(sol, rate_pref, stag_tol, road_course=rc),
    }

    # Order by priority rank (lowest number = highest priority)
    ordered_keys = [k for k, _ in sorted(priorities.items(), key=lambda x: x[1])
                    if k in refine_map]

    for _ in range(max_passes):
        any_improved = False
        for key in ordered_keys:
            # Keep refining this criterion until it can't improve
            while True:
                success, _ = refine_map[key](solution)
                if success:
                    any_improved = True
                else:
                    break
        if not any_improved:
            break

    # Final pass: re-refine highest priority (cross usually) since lower-priority
    # refinements may have degraded it slightly within their tolerance.
    if ordered_keys:
        while True:
            success, _ = refine_map[ordered_keys[0]](solution)
            if not success:
                break

    return solution


def find_same_side_swaps(solution: List[dict], road_course: bool = False) -> List[Tuple]:
    """Find all valid same-pool swaps between sets.

    Oval: Left corners (LF, LR) swap with left; Right (RF, RR) with right.
    Road Course: A-pool corners (LR, RF) swap together; non-A (LF, RR) together.
    """
    swaps = []
    n = len(solution)

    if road_course:
        pool_a = ['LR', 'RF']   # A tires (diagonal)
        pool_b = ['LF', 'RR']   # non-A tires (diagonal)
    else:
        pool_a = ['LF', 'LR']   # left side
        pool_b = ['RF', 'RR']   # right side

    for i in range(n):
        for j in range(i + 1, n):
            for c1 in pool_a:
                for c2 in pool_a:
                    swaps.append((i, c1, j, c2))
            for c1 in pool_b:
                for c2 in pool_b:
                    swaps.append((i, c1, j, c2))

    return swaps


def evaluate_swap(solution: List[dict], set_a: int, corner_a: str, set_b: int, corner_b: str) -> dict:
    """Evaluate metrics after a hypothetical swap without modifying original."""
    corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}

    # Create temporary copies
    new_set_a = dict(solution[set_a])
    new_set_b = dict(solution[set_b])

    # Perform swap
    tire_a = solution[set_a][corner_map[corner_a]]
    tire_b = solution[set_b][corner_map[corner_b]]
    new_set_a[corner_map[corner_a]] = tire_b
    new_set_b[corner_map[corner_b]] = tire_a

    # Recalculate metrics for both sets
    metrics_a = compute_set_metrics(
        new_set_a['lf_data'], new_set_a['rf_data'],
        new_set_a['lr_data'], new_set_a['rr_data']
    )
    metrics_b = compute_set_metrics(
        new_set_b['lf_data'], new_set_b['rf_data'],
        new_set_b['lr_data'], new_set_b['rr_data']
    )

    return {
        'set_a_metrics': metrics_a,
        'set_b_metrics': metrics_b,
        'set_a_idx': set_a,
        'set_b_idx': set_b
    }


def refine_cross(solution: List[dict], cross_target: float, stagger_tolerance: float, road_course: bool = False) -> Tuple[bool, str]:
    """Refine solution for better cross weight without changing stagger."""
    if not solution:
        return False, "No solution"

    # Current cross variance
    current_crosses = [s['cross'] for s in solution]
    current_cross_var = np.var(current_crosses)
    current_cross_dev = sum(abs(c - cross_target) for c in current_crosses)

    best_swap = None
    best_improvement = 0

    swaps = find_same_side_swaps(solution, road_course=road_course)

    for set_a, corner_a, set_b, corner_b in swaps:
        result = evaluate_swap(solution, set_a, corner_a, set_b, corner_b)

        # Check stagger constraint - must not change more than tolerance
        old_stagger_a = solution[set_a]['stagger']
        old_stagger_b = solution[set_b]['stagger']
        new_stagger_a = result['set_a_metrics']['stagger']
        new_stagger_b = result['set_b_metrics']['stagger']

        if abs(new_stagger_a - old_stagger_a) > stagger_tolerance:
            continue
        if abs(new_stagger_b - old_stagger_b) > stagger_tolerance:
            continue

        # Calculate new cross metrics
        new_crosses = current_crosses.copy()
        new_crosses[set_a] = result['set_a_metrics']['cross']
        new_crosses[set_b] = result['set_b_metrics']['cross']

        new_cross_dev = sum(abs(c - cross_target) for c in new_crosses)
        new_spread = max(new_crosses) - min(new_crosses) if len(new_crosses) > 1 else 0

        if road_course:
            # Road Course: consistency dominates — spread reduction worth 5x
            current_spread = max(current_crosses) - min(current_crosses) if len(current_crosses) > 1 else 0
            improvement = (current_spread - new_spread) * 5 + (current_cross_dev - new_cross_dev)
        else:
            improvement = current_cross_dev - new_cross_dev

        if improvement > best_improvement:
            best_improvement = improvement
            best_swap = (set_a, corner_a, set_b, corner_b)

    if best_swap:
        # Apply the best swap
        corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}
        set_a, corner_a, set_b, corner_b = best_swap

        tire_a = solution[set_a][corner_map[corner_a]]
        tire_b = solution[set_b][corner_map[corner_b]]
        solution[set_a][corner_map[corner_a]] = tire_b
        solution[set_b][corner_map[corner_b]] = tire_a

        # Update metrics
        for idx in [set_a, set_b]:
            s = solution[idx]
            metrics = compute_set_metrics(s['lf_data'], s['rf_data'], s['lr_data'], s['rr_data'])
            solution[idx].update(metrics)

        return True, f"Swapped Set {set_a+1} {corner_a} with Set {set_b+1} {corner_b}"

    return False, "No beneficial swap found"


def refine_shift(solution: List[dict], cross_target: float, stagger_tolerance: float, cross_tolerance: float = 0.005, road_course: bool = False) -> Tuple[bool, str]:
    """Refine solution for better shift code matching without changing stagger or cross."""
    if not solution:
        return False, "No solution"

    # Current shift score (lower is better)
    current_shift_total = sum(s['shift_score'] for s in solution)

    best_swap = None
    best_improvement = 0

    swaps = find_same_side_swaps(solution, road_course=road_course)

    for set_a, corner_a, set_b, corner_b in swaps:
        result = evaluate_swap(solution, set_a, corner_a, set_b, corner_b)

        # Check stagger constraint
        old_stagger_a = solution[set_a]['stagger']
        old_stagger_b = solution[set_b]['stagger']
        new_stagger_a = result['set_a_metrics']['stagger']
        new_stagger_b = result['set_b_metrics']['stagger']

        if abs(new_stagger_a - old_stagger_a) > stagger_tolerance:
            continue
        if abs(new_stagger_b - old_stagger_b) > stagger_tolerance:
            continue

        # Check cross constraint
        old_cross_a = solution[set_a]['cross']
        old_cross_b = solution[set_b]['cross']
        new_cross_a = result['set_a_metrics']['cross']
        new_cross_b = result['set_b_metrics']['cross']

        if abs(new_cross_a - old_cross_a) > cross_tolerance:
            continue
        if abs(new_cross_b - old_cross_b) > cross_tolerance:
            continue

        # Calculate new shift score
        new_shift_a = result['set_a_metrics']['shift_score']
        new_shift_b = result['set_b_metrics']['shift_score']
        old_shift_a = solution[set_a]['shift_score']
        old_shift_b = solution[set_b]['shift_score']

        improvement = (old_shift_a + old_shift_b) - (new_shift_a + new_shift_b)

        if improvement > best_improvement:
            best_improvement = improvement
            best_swap = (set_a, corner_a, set_b, corner_b)

    if best_swap:
        corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}
        set_a, corner_a, set_b, corner_b = best_swap

        tire_a = solution[set_a][corner_map[corner_a]]
        tire_b = solution[set_b][corner_map[corner_b]]
        solution[set_a][corner_map[corner_a]] = tire_b
        solution[set_b][corner_map[corner_b]] = tire_a

        for idx in [set_a, set_b]:
            s = solution[idx]
            metrics = compute_set_metrics(s['lf_data'], s['rf_data'], s['lr_data'], s['rr_data'])
            solution[idx].update(metrics)

        return True, f"Swapped Set {set_a+1} {corner_a} with Set {set_b+1} {corner_b}"

    return False, "No beneficial swap found"


def refine_rate(solution: List[dict], rate_pref: str, stagger_tolerance: float, cross_tolerance: float = 0.005, road_course: bool = False) -> Tuple[bool, str]:
    """Refine solution for rate preference without changing stagger, cross, or shift."""
    if not solution or rate_pref == 'None':
        return False, "No rate preference set"

    # Current rate score (count of sets matching preference)
    if rate_pref == 'Softer Rear':
        current_score = sum(1 for s in solution if s['rear_softer'])
    else:  # Softer Front
        current_score = sum(1 for s in solution if s['front_softer'])

    best_swap = None
    best_improvement = 0

    swaps = find_same_side_swaps(solution, road_course=road_course)

    for set_a, corner_a, set_b, corner_b in swaps:
        result = evaluate_swap(solution, set_a, corner_a, set_b, corner_b)

        # Check stagger constraint
        old_stagger_a = solution[set_a]['stagger']
        old_stagger_b = solution[set_b]['stagger']
        new_stagger_a = result['set_a_metrics']['stagger']
        new_stagger_b = result['set_b_metrics']['stagger']

        if abs(new_stagger_a - old_stagger_a) > stagger_tolerance:
            continue
        if abs(new_stagger_b - old_stagger_b) > stagger_tolerance:
            continue

        # Check cross constraint
        old_cross_a = solution[set_a]['cross']
        old_cross_b = solution[set_b]['cross']
        new_cross_a = result['set_a_metrics']['cross']
        new_cross_b = result['set_b_metrics']['cross']

        if abs(new_cross_a - old_cross_a) > cross_tolerance:
            continue
        if abs(new_cross_b - old_cross_b) > cross_tolerance:
            continue

        # Check shift constraint (must not get worse)
        old_shift_a = solution[set_a]['shift_score']
        old_shift_b = solution[set_b]['shift_score']
        new_shift_a = result['set_a_metrics']['shift_score']
        new_shift_b = result['set_b_metrics']['shift_score']

        if (new_shift_a + new_shift_b) > (old_shift_a + old_shift_b):
            continue

        # Calculate rate improvement
        if rate_pref == 'Softer Rear':
            old_match = (1 if solution[set_a]['rear_softer'] else 0) + (1 if solution[set_b]['rear_softer'] else 0)
            new_match = (1 if result['set_a_metrics']['rear_softer'] else 0) + (1 if result['set_b_metrics']['rear_softer'] else 0)
        else:
            old_match = (1 if solution[set_a]['front_softer'] else 0) + (1 if solution[set_b]['front_softer'] else 0)
            new_match = (1 if result['set_a_metrics']['front_softer'] else 0) + (1 if result['set_b_metrics']['front_softer'] else 0)

        improvement = new_match - old_match

        if improvement > best_improvement:
            best_improvement = improvement
            best_swap = (set_a, corner_a, set_b, corner_b)

    if best_swap:
        corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}
        set_a, corner_a, set_b, corner_b = best_swap

        tire_a = solution[set_a][corner_map[corner_a]]
        tire_b = solution[set_b][corner_map[corner_b]]
        solution[set_a][corner_map[corner_a]] = tire_b
        solution[set_b][corner_map[corner_b]] = tire_a

        for idx in [set_a, set_b]:
            s = solution[idx]
            metrics = compute_set_metrics(s['lf_data'], s['rf_data'], s['lr_data'], s['rr_data'])
            solution[idx].update(metrics)

        return True, f"Swapped Set {set_a+1} {corner_a} with Set {set_b+1} {corner_b}"

    return False, "No beneficial swap found"


def refine_date(solution: List[dict], stagger_tolerance: float, cross_tolerance: float = 0.005, road_course: bool = False) -> Tuple[bool, str]:
    """Refine solution for better date code matching without changing stagger, cross, or shift."""
    if not solution:
        return False, "No solution"

    # Current date score (lower is better, 0 = both sides match in all sets)
    current_date_total = sum(s.get('date_score', 0) for s in solution)

    best_swap = None
    best_improvement = 0

    swaps = find_same_side_swaps(solution, road_course=road_course)

    for set_a, corner_a, set_b, corner_b in swaps:
        result = evaluate_swap(solution, set_a, corner_a, set_b, corner_b)

        # Check stagger constraint
        old_stagger_a = solution[set_a]['stagger']
        old_stagger_b = solution[set_b]['stagger']
        new_stagger_a = result['set_a_metrics']['stagger']
        new_stagger_b = result['set_b_metrics']['stagger']

        if abs(new_stagger_a - old_stagger_a) > stagger_tolerance:
            continue
        if abs(new_stagger_b - old_stagger_b) > stagger_tolerance:
            continue

        # Check cross constraint
        old_cross_a = solution[set_a]['cross']
        old_cross_b = solution[set_b]['cross']
        new_cross_a = result['set_a_metrics']['cross']
        new_cross_b = result['set_b_metrics']['cross']

        if abs(new_cross_a - old_cross_a) > cross_tolerance:
            continue
        if abs(new_cross_b - old_cross_b) > cross_tolerance:
            continue

        # Check shift constraint (must not get worse)
        old_shift_a = solution[set_a]['shift_score']
        old_shift_b = solution[set_b]['shift_score']
        new_shift_a = result['set_a_metrics']['shift_score']
        new_shift_b = result['set_b_metrics']['shift_score']

        if (new_shift_a + new_shift_b) > (old_shift_a + old_shift_b):
            continue

        # Calculate date code improvement (lower date_score is better)
        old_date_score = solution[set_a].get('date_score', 0) + solution[set_b].get('date_score', 0)
        new_date_score = result['set_a_metrics'].get('date_score', 0) + result['set_b_metrics'].get('date_score', 0)

        improvement = old_date_score - new_date_score

        if improvement > best_improvement:
            best_improvement = improvement
            best_swap = (set_a, corner_a, set_b, corner_b)

    if best_swap:
        corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}
        set_a, corner_a, set_b, corner_b = best_swap

        tire_a = solution[set_a][corner_map[corner_a]]
        tire_b = solution[set_b][corner_map[corner_b]]
        solution[set_a][corner_map[corner_a]] = tire_b
        solution[set_b][corner_map[corner_b]] = tire_a

        for idx in [set_a, set_b]:
            s = solution[idx]
            metrics = compute_set_metrics(s['lf_data'], s['rf_data'], s['lr_data'], s['rr_data'])
            solution[idx].update(metrics)

        return True, f"Swapped Set {set_a+1} {corner_a} with Set {set_b+1} {corner_b}"

    return False, "No beneficial swap found"


def refine_rr_rollout(solution: List[dict], stagger_tolerance: float, cross_tolerance: float = 0.005, road_course: bool = False) -> Tuple[bool, str]:
    """Refine solution for more consistent RR rollout without changing higher priorities."""
    if not solution:
        return False, "No solution"

    # Current RR rollout variance
    rr_rollouts = [s.get('rr_rollout', 0) for s in solution]
    current_variance = np.var(rr_rollouts)

    best_swap = None
    best_improvement = 0

    swaps = find_same_side_swaps(solution, road_course=road_course)

    for set_a, corner_a, set_b, corner_b in swaps:
        # Only consider RR swaps for RR rollout consistency
        if corner_a != 'RR' or corner_b != 'RR':
            continue

        result = evaluate_swap(solution, set_a, corner_a, set_b, corner_b)

        # Check stagger constraint
        old_stagger_a = solution[set_a]['stagger']
        old_stagger_b = solution[set_b]['stagger']
        new_stagger_a = result['set_a_metrics']['stagger']
        new_stagger_b = result['set_b_metrics']['stagger']

        if abs(new_stagger_a - old_stagger_a) > stagger_tolerance:
            continue
        if abs(new_stagger_b - old_stagger_b) > stagger_tolerance:
            continue

        # Check cross constraint
        old_cross_a = solution[set_a]['cross']
        old_cross_b = solution[set_b]['cross']
        new_cross_a = result['set_a_metrics']['cross']
        new_cross_b = result['set_b_metrics']['cross']

        if abs(new_cross_a - old_cross_a) > cross_tolerance:
            continue
        if abs(new_cross_b - old_cross_b) > cross_tolerance:
            continue

        # Check shift constraint (must not get worse)
        old_shift_a = solution[set_a]['shift_score']
        old_shift_b = solution[set_b]['shift_score']
        new_shift_a = result['set_a_metrics']['shift_score']
        new_shift_b = result['set_b_metrics']['shift_score']

        if (new_shift_a + new_shift_b) > (old_shift_a + old_shift_b):
            continue

        # Check date constraint (must not get worse)
        old_date_a = solution[set_a].get('date_score', 0)
        old_date_b = solution[set_b].get('date_score', 0)
        new_date_a = result['set_a_metrics'].get('date_score', 0)
        new_date_b = result['set_b_metrics'].get('date_score', 0)

        if (new_date_a + new_date_b) > (old_date_a + old_date_b):
            continue

        # Calculate new RR rollout variance
        new_rollouts = rr_rollouts.copy()
        new_rollouts[set_a] = result['set_a_metrics'].get('rr_rollout', 0)
        new_rollouts[set_b] = result['set_b_metrics'].get('rr_rollout', 0)
        new_variance = np.var(new_rollouts)

        improvement = current_variance - new_variance

        if improvement > best_improvement:
            best_improvement = improvement
            best_swap = (set_a, corner_a, set_b, corner_b)

    if best_swap:
        corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}
        set_a, corner_a, set_b, corner_b = best_swap

        tire_a = solution[set_a][corner_map[corner_a]]
        tire_b = solution[set_b][corner_map[corner_b]]
        solution[set_a][corner_map[corner_a]] = tire_b
        solution[set_b][corner_map[corner_b]] = tire_a

        for idx in [set_a, set_b]:
            s = solution[idx]
            metrics = compute_set_metrics(s['lf_data'], s['rf_data'], s['lr_data'], s['rr_data'])
            solution[idx].update(metrics)

        return True, f"Swapped Set {set_a+1} {corner_a} with Set {set_b+1} {corner_b}"

    return False, "No beneficial swap found"


def sort_by_rr_rollout(solution: List[dict]) -> List[dict]:
    """Sort sets to group similar RR rollouts together."""
    if not solution or len(solution) <= 1:
        return solution

    # Sort by RR rollout
    return sorted(solution, key=lambda s: s.get('rr_rollout', 0))


def render_tire_html(tire, corner: str, highlight: bool = False, road_course: bool = False, compact: bool = False) -> str:
    """Render a single tire position as a styled HTML block."""
    if road_course:
        # Road Course: color by pool — A-pool (LR/RF) amber, non-A (LF/RR) teal
        css_class = "pool-a" if corner in ('LR', 'RF') else "pool-b"
    else:
        # Oval: color by side — left green, right blue
        css_class = "left" if corner in ('LF', 'LR') else "right"

    # Add compact-table class for table view
    if compact:
        css_class += " compact-table"

    shift = tire.get('Shift', '') or '-'
    date = tire.get('Date Code', '')
    date_str = str(date).strip() if date and str(date).strip() not in ('', 'nan') else '-'
    hl = "border: 2px solid #2e7d32; box-shadow: 0 0 8px rgba(46,125,50,0.35);" if highlight else ""
    return (
        f'<div class="tire-box {css_class}" style="{hl}">'
        f'<div class="tire-corner">{corner}</div>'
        f'<div class="tire-row"><span class="tire-label">Roll:</span><span class="tire-rollout">{tire["Rollout/Dia"]:.0f}</span></div>'
        f'<div class="tire-row"><span class="tire-label">Rate:</span><span class="tire-rate">{int(tire["Rate"])}</span></div>'
        f'<div class="tire-row"><span class="tire-label">Shift:</span><span class="tire-shift">{shift}</span></div>'
        f'<div class="tire-row"><span class="tire-label">Date:</span><span class="tire-date">{date_str}</span></div>'
        f'</div>'
    )


def build_reference_table(solution: List[dict]) -> pd.DataFrame:
    """Build sortable reference table from solution sets."""
    rows = []
    for idx, s in enumerate(solution):
        rear_avg = s['rear_avg_rate']
        front_avg = s['front_avg_rate']

        def get_date(tire_data):
            d = tire_data.get('Date Code', '')
            return str(d).strip() if d and str(d) != 'nan' else '-'

        dates = [get_date(s[k]) for k in ['lf_data', 'rf_data', 'lr_data', 'rr_data']]
        newest_date = max([d for d in dates if d != '-']) if any(d != '-' for d in dates) else '-'

        rows.append({
            'Set': idx + 1,
            'Rear Avg Rate': round(rear_avg, 1),
            'Front Avg Rate': round(front_avg, 1),
            'Stagger': round(s['stagger'], 1),
            'Front Stagger': round(s.get('front_stagger', 0), 1),
            'Cross %': round(s['cross'] * 100, 2),
            'Newest Date': newest_date,
            'LF Date': dates[0],
            'RF Date': dates[1],
            'LR Date': dates[2],
            'RR Date': dates[3],
        })

    return pd.DataFrame(rows)


# ============== SIDEBAR ==============
st.sidebar.markdown("### Import Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File",
    type=['xlsx', 'xlsm', 'xls'],
    help="Excel file with Scan Data sheet"
)

if uploaded_file is not None:
    token = (uploaded_file.name, uploaded_file.size)
    if st.session_state._upload_token != token:
        try:
            df = load_tire_data(uploaded_file)
            st.session_state.tire_df = df
            st.session_state._upload_token = token

            dcodes = sorted(df['D-Code'].unique().tolist())
            st.session_state.available_dcodes = dcodes

            # Auto-select LS D-Code as the one with smaller avg rollout
            # so that stagger (RR - LR) is positive
            if len(dcodes) >= 2:
                avg_rollouts = {d: df[df['D-Code'] == d]['Rollout/Dia'].mean() for d in dcodes}
                st.session_state.ls_dcode = min(avg_rollouts, key=avg_rollouts.get)
            elif dcodes:
                st.session_state.ls_dcode = dcodes[0]
            else:
                st.session_state.ls_dcode = None

            # Set default stagger target to max achievable for all sets
            if st.session_state.ls_dcode is not None:
                _left, _right = assign_positions(df, st.session_state.ls_dcode)
                _sa = analyze_stagger_range(_left, _right)
                st.session_state.target_stagger = float(_sa['max_all_sets'])

            st.session_state.data_loaded = True
            st.session_state.results = None

            # Check for duplicates
            dup_warnings = detect_input_duplicates(df)
            st.session_state.duplicate_warnings = dup_warnings

            if dup_warnings:
                for warning in dup_warnings:
                    st.sidebar.warning(warning)

            st.sidebar.success(f"Loaded {len(df)} tires")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

st.sidebar.divider()

track_type = st.sidebar.radio(
    "Track Type",
    options=['Oval', 'Road Course'],
    index=0 if st.session_state.track_type == 'Oval' else 1,
    horizontal=True
)
if track_type != st.session_state.track_type:
    st.session_state.results = None  # clear stale results on mode switch
st.session_state.track_type = track_type

if st.session_state.data_loaded and st.session_state.tire_df is not None:
    st.sidebar.markdown("### Tire Setup")

    if track_type == 'Oval':
        # Oval: user picks the LS D-Code
        if st.session_state.available_dcodes:
            ls_dcode = st.sidebar.selectbox(
                "Left Side D-Code (LF/LR)",
                options=st.session_state.available_dcodes,
                index=st.session_state.available_dcodes.index(st.session_state.ls_dcode) if st.session_state.ls_dcode in st.session_state.available_dcodes else 0,
            )

            if ls_dcode != st.session_state.ls_dcode:
                st.session_state.ls_dcode = ls_dcode
                st.session_state.results = None

            rs_dcodes = [d for d in st.session_state.available_dcodes if d != ls_dcode]
            if rs_dcodes:
                st.sidebar.caption(f"Right Side: {', '.join(rs_dcodes)}")

            left, right = assign_positions(st.session_state.tire_df, ls_dcode, track_type='Oval')
            st.session_state.left_tires = left
            st.session_state.right_tires = right

    else:
        # Road Course: split by Wheel suffix (A → LR/RF, non-A → LF/RR)
        if 'Wheel' not in st.session_state.tire_df.columns:
            st.sidebar.warning("No Wheel column found — Road Course requires Wheel data.")
            st.session_state.left_tires = pd.DataFrame()
            st.session_state.right_tires = pd.DataFrame()
        else:
            a_pool, non_a_pool = assign_positions(st.session_state.tire_df, '', track_type='Road Course')
            st.session_state.left_tires = a_pool       # A tires → LR/RF pool
            st.session_state.right_tires = non_a_pool  # non-A tires → LF/RR pool
            st.sidebar.caption(f"'A' Wheels (LR/RF): **{len(a_pool)}** tires")
            st.sidebar.caption(f"Non-A Wheels (LF/RR): **{len(non_a_pool)}** tires")


# ============== MAIN AREA ==============
st.markdown(f"#### {APP_TITLE}")

tab_settings, tab_results = st.tabs(["Setup & Sort", "Results & Refine"])

# ---------- TAB 1: SETUP & SORT ----------
with tab_settings:
    if st.session_state.data_loaded:
        left = st.session_state.left_tires
        right = st.session_state.right_tires

        if left is not None and right is not None and len(left) > 0 and len(right) > 0:
            sa = analyze_stagger_range(left, right)
            n_sets = sa.get('n_sets', 0)

            # --- Info row ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Sets Available", n_sets)
            c2.metric("Stagger Range (any set)", f"{sa['min_single']:.0f} to {sa['max_single']:.0f}")
            c3.metric("Stagger Range (all sets)", f"{sa['min_all_sets']:.0f} to {sa['max_all_sets']:.0f}")

            # --- Controls ---
            col_l, col_r = st.columns(2)

            with col_l:
                if track_type == 'Road Course':
                    st.session_state.target_stagger = 0.0
                    st.caption("Road Course — stagger locked to **0**")
                else:
                    default_stag = float(sa['max_all_sets'])
                    cur = float(st.session_state.target_stagger)
                    mn, mx = float(sa['min_single']), float(sa['max_single'])
                    init_val = cur if mn <= cur <= mx else default_stag
                    init_val = max(mn, min(mx, init_val))

                    s1, s2 = st.columns(2)
                    with s1:
                        target_stagger = st.number_input(
                            "Stagger Target (RR - LR)", min_value=mn, max_value=mx,
                            value=init_val, step=1.0,
                        )
                        st.session_state.target_stagger = target_stagger
                    with s2:
                        stagger_tol = st.number_input(
                            "Tolerance (+/-)", min_value=0, max_value=10,
                            value=int(st.session_state.stagger_tolerance), step=1,
                        )
                        st.session_state.stagger_tolerance = stagger_tol

            with col_r:
                rate_pref = st.selectbox(
                    "Front/Rear Rate Preference",
                    options=['Softer Rear', 'Softer Front', 'None'],
                    index=['Softer Rear', 'Softer Front', 'None'].index(st.session_state.rate_preference),
                )
                st.session_state.rate_preference = rate_pref

            st.divider()

            # --- Priorities ---
            st.caption("Stagger is always #1. Drag to reorder the rest.")
            sorted_order = sort_items(st.session_state.priority_order, direction="vertical")
            st.session_state.priority_order = sorted_order

            # --- Run button ---
            run_sort = st.button("Run Tire Sort", use_container_width=True, type="primary")

            if run_sort:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Building candidates...")

                is_road = track_type == 'Road Course'
                if is_road:
                    candidates = build_candidates_road(
                        left, right,
                        target_stagger=st.session_state.target_stagger,
                        stagger_tolerance=st.session_state.stagger_tolerance
                    )
                else:
                    candidates = build_candidates(
                        left, right,
                        target_stagger=st.session_state.target_stagger,
                        stagger_tolerance=st.session_state.stagger_tolerance
                    )

                if not candidates:
                    st.error(f"No combinations with stagger {st.session_state.target_stagger:.0f} +/- {st.session_state.stagger_tolerance}. Increase tolerance or change target.")
                else:
                    status_text.text(f"{len(candidates)} combinations found. Optimizing...")

                    n_tires = len(left) + len(right)
                    priorities = {'stagger': 1}
                    for i, name in enumerate(st.session_state.priority_order, start=2):
                        priorities[PRIORITY_NAME_MAP[name]] = i

                    def update_progress(pct):
                        progress_bar.progress(pct)
                        status_text.text(f"Optimizing... {int(pct * 100)}%")

                    solution = randomized_greedy(
                        candidates, n_tires,
                        st.session_state.target_stagger,
                        st.session_state.cross_target,
                        priorities,
                        st.session_state.rate_preference,
                        restarts=RESTARTS, beam=BEAM,
                        progress_callback=update_progress,
                        road_course=is_road
                    )

                    progress_bar.progress(1.0)

                    if solution:
                        status_text.text("Polishing solution...")
                        solution = polish_solution(
                            solution, priorities,
                            st.session_state.cross_target,
                            st.session_state.stagger_tolerance,
                            st.session_state.rate_preference,
                            road_course=is_road,
                        )
                        solution = sort_by_rr_rollout(solution)
                        st.session_state.results = solution
                        st.session_state.selected_tire = None
                        st.session_state.selected_set = None

                        staggers = [s['stagger'] for s in solution]
                        crosses = [s['cross'] for s in solution]
                        st.session_state.stats = {
                            'n_sets': len(solution),
                            'mean_stagger': np.mean(staggers),
                            'std_stagger': np.std(staggers),
                            'mean_cross': np.mean(crosses),
                            'std_cross': np.std(crosses),
                        }
                        status_text.text(f"Done — {len(solution)} sets sorted.")
                    else:
                        st.warning("Could not find a complete solution.")
                        status_text.text("No solution found.")

        else:
            st.warning("Need tires in both left and right pools.")
    else:
        st.info("Upload an Excel file in the sidebar to get started.")


# ---------- TAB 2: RESULTS & REFINE ----------
with tab_results:
    if st.session_state.results is not None:
        solution = st.session_state.results
        stats = st.session_state.stats
        has_results = True

        # --- Check for duplicate tires in solution ---
        sol_warnings = detect_solution_duplicates(solution)
        if sol_warnings:
            st.error("⚠️ Duplicate tires detected in solution:")
            for warning in sol_warnings:
                st.error(warning)

        # --- Refine toolbar ---
        rate_pref = st.session_state.rate_preference
        btn_row = st.columns(5)

        with btn_row[0]:
            refine_cross_clicked = st.button("Refine Cross", use_container_width=True, help="Optimize cross weight")
        with btn_row[1]:
            refine_shift_clicked = st.button("Refine Shift", use_container_width=True, help="Match shift codes")
        with btn_row[2]:
            refine_date_clicked = st.button("Refine Date", use_container_width=True, help="Match date codes (L/R)")
        with btn_row[3]:
            refine_rr_clicked = st.button("Refine RR", use_container_width=True, help="Consistent RR rollout")
        with btn_row[4]:
            refine_rate_clicked = st.button("Refine Rate", use_container_width=True, disabled=(rate_pref == 'None'), help="Softer front/rear preference")

        # --- Helper to update stats after refinement ---
        def _update_stats(solution):
            staggers = [s['stagger'] for s in solution]
            crosses = [s['cross'] for s in solution]
            st.session_state.stats = {
                'n_sets': len(solution),
                'mean_stagger': np.mean(staggers),
                'std_stagger': np.std(staggers),
                'mean_cross': np.mean(crosses),
                'std_cross': np.std(crosses),
            }

        stagger_tol = st.session_state.stagger_tolerance if st.session_state.stagger_tolerance > 0 else 0.5
        is_road = st.session_state.track_type == 'Road Course'

        if refine_cross_clicked:
            success, msg = refine_cross(solution, st.session_state.cross_target, stagger_tol, road_course=is_road)
            if success:
                _update_stats(solution)
                st.toast(f"Cross refined: {msg}")
                st.rerun()
            else:
                st.toast(msg)

        if refine_shift_clicked:
            success, msg = refine_shift(solution, st.session_state.cross_target, stagger_tol, road_course=is_road)
            if success:
                _update_stats(solution)
                st.toast(f"Shift refined: {msg}")
                st.rerun()
            else:
                st.toast(msg)

        if refine_date_clicked:
            success, msg = refine_date(solution, stagger_tol, road_course=is_road)
            if success:
                _update_stats(solution)
                st.toast(f"Date refined: {msg}")
                st.rerun()
            else:
                st.toast(msg)

        if refine_rr_clicked:
            success, msg = refine_rr_rollout(solution, stagger_tol, road_course=is_road)
            if success:
                _update_stats(solution)
                st.session_state.results = sort_by_rr_rollout(solution)
                st.toast(f"RR refined: {msg}")
                st.rerun()
            else:
                st.toast(msg)

        if refine_rate_clicked:
            success, msg = refine_rate(solution, rate_pref, stagger_tol, road_course=is_road)
            if success:
                _update_stats(solution)
                st.toast(f"Rate refined: {msg}")
                st.rerun()
            else:
                st.toast(msg)

        selected = st.session_state.selected_tire
        selected_set = st.session_state.selected_set

        # --- Helper to perform manual tire swap ---
        def do_swap(from_set, from_corner, to_set, to_corner):
            corner_map = {'LF': 'lf_data', 'RF': 'rf_data', 'LR': 'lr_data', 'RR': 'rr_data'}
            from_tire = solution[from_set][corner_map[from_corner]]
            to_tire = solution[to_set][corner_map[to_corner]]

            solution[from_set][corner_map[from_corner]] = to_tire
            solution[to_set][corner_map[to_corner]] = from_tire

            for idx in [from_set, to_set]:
                s = solution[idx]
                metrics = compute_set_metrics(s['lf_data'], s['rf_data'], s['lr_data'], s['rr_data'])
                solution[idx].update(metrics)

            st.session_state.results = solution
            _update_stats(solution)
            st.session_state.selected_tire = None

        # --- Helper to swap entire sets ---
        def do_set_swap(set_a, set_b):
            solution[set_a], solution[set_b] = solution[set_b], solution[set_a]
            st.session_state.results = solution
            st.session_state.selected_set = None

        # --- Compact view toggle (appears when 10+ sets) ---
        if len(solution) >= 10:
            compact = st.toggle("Compact View", value=st.session_state.compact_view,
                              help="Show 6-8 sets per row with smaller cards")
            st.session_state.compact_view = compact
        else:
            st.session_state.compact_view = False  # Reset if < 10 sets

        # --- Interactive car cards grid ---
        is_compact = st.session_state.compact_view
        is_road_results = st.session_state.track_type == 'Road Course'

        if is_compact:
            # --- COMPACT TABLE VIEW (11 columns, 2 rows per set) ---
            st.subheader("Tire Sets")

            # Helper to get tire data safely
            def get_val(tire, field, default='-'):
                val = tire.get(field, default) if field in tire.index else default
                return str(val) if val and str(val) not in ['nan', 'None', ''] else '-'

            # Build DataFrame for editable table
            table_rows = []
            row_metadata = []  # Track set_idx and position (front/rear) for each row

            for set_idx, s in enumerate(solution):
                lf = s['lf_data']
                rf = s['rf_data']
                lr = s['lr_data']
                rr = s['rr_data']

                # Front row
                lf_num = int(lf['Number']) if 'Number' in lf.index else 0
                rf_num = int(rf['Number']) if 'Number' in rf.index else 0
                table_rows.append({
                    'Set': set_idx + 1,
                    'LF ID': lf_num,
                    'RF ID': rf_num,
                    'LF Roll': int(lf['Rollout/Dia']),
                    'RF Roll': int(rf['Rollout/Dia']),
                    'LF Rate': int(lf['Rate']),
                    'RF Rate': int(rf['Rate']),
                    'LF Date': get_val(lf, 'Date Code'),
                    'RF Date': get_val(rf, 'Date Code'),
                    'LF Shift': get_val(lf, 'Shift'),
                    'RF Shift': get_val(rf, 'Shift'),
                    'Cross%': f"{s['cross']*100:.2f}",
                    'Stagger': f"{s.get('front_stagger', 0):.1f}"
                })
                row_metadata.append({'set_idx': set_idx, 'position': 'front'})

                # Rear row
                lr_num = int(lr['Number']) if 'Number' in lr.index else 0
                rr_num = int(rr['Number']) if 'Number' in rr.index else 0
                table_rows.append({
                    'Set': '',  # Empty for merged appearance
                    'LF ID': lr_num,
                    'RF ID': rr_num,
                    'LF Roll': int(lr['Rollout/Dia']),
                    'RF Roll': int(rr['Rollout/Dia']),
                    'LF Rate': int(lr['Rate']),
                    'RF Rate': int(rr['Rate']),
                    'LF Date': get_val(lr, 'Date Code'),
                    'RF Date': get_val(rr, 'Date Code'),
                    'LF Shift': get_val(lr, 'Shift'),
                    'RF Shift': get_val(rr, 'Shift'),
                    'Cross%': '',  # Empty for merged appearance
                    'Stagger': f"{s['stagger']:.1f}"
                })
                row_metadata.append({'set_idx': set_idx, 'position': 'rear'})

            df = pd.DataFrame(table_rows)

            # Display editable table
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Set': st.column_config.TextColumn('Set #', width='small', disabled=True),
                    'LF ID': st.column_config.NumberColumn('LF ID', width='small', min_value=0, max_value=9999),
                    'RF ID': st.column_config.NumberColumn('RF ID', width='small', min_value=0, max_value=9999),
                    'LF Roll': st.column_config.NumberColumn('LF Roll', width='small', disabled=True),
                    'RF Roll': st.column_config.NumberColumn('RF Roll', width='small', disabled=True),
                    'LF Rate': st.column_config.NumberColumn('LF Rate', width='small', disabled=True),
                    'RF Rate': st.column_config.NumberColumn('RF Rate', width='small', disabled=True),
                    'LF Date': st.column_config.TextColumn('LF Date', width='small', disabled=True),
                    'RF Date': st.column_config.TextColumn('RF Date', width='small', disabled=True),
                    'LF Shift': st.column_config.TextColumn('LF Shift', width='small', disabled=True),
                    'RF Shift': st.column_config.TextColumn('RF Shift', width='small', disabled=True),
                    'Cross%': st.column_config.TextColumn('Cross%', width='small', disabled=True),
                    'Stagger': st.column_config.TextColumn('Stagger', width='small', disabled=True),
                },
                key='compact_table_editor'
            )

            # Detect changes and perform swaps
            if edited_df is not None:
                changes_detected = False
                for row_idx in range(len(df)):
                    meta = row_metadata[row_idx]
                    set_idx = meta['set_idx']
                    is_front = meta['position'] == 'front'

                    # Check LF ID change
                    old_lf = df.at[row_idx, 'LF ID']
                    new_lf = edited_df.at[row_idx, 'LF ID']
                    if old_lf != new_lf and new_lf > 0:
                        corner = 'lf' if is_front else 'lr'
                        # Find tire with new ID in available pool
                        target_tire = None
                        if is_road_results:
                            pool = left_a if solution[set_idx][f'{corner}_data']['D-Code'].endswith('A') else left_non_a
                        else:
                            pool = left_tires

                        for tire in pool:
                            if 'Number' in tire.index and int(tire['Number']) == int(new_lf):
                                target_tire = tire
                                break

                        if target_tire is not None:
                            solution[set_idx][f'{corner}_data'] = target_tire
                            changes_detected = True

                    # Check RF ID change
                    old_rf = df.at[row_idx, 'RF ID']
                    new_rf = edited_df.at[row_idx, 'RF ID']
                    if old_rf != new_rf and new_rf > 0:
                        corner = 'rf' if is_front else 'rr'
                        # Find tire with new ID in available pool
                        target_tire = None
                        if is_road_results:
                            pool = right_a if solution[set_idx][f'{corner}_data']['D-Code'].endswith('A') else right_non_a
                        else:
                            pool = right_tires

                        for tire in pool:
                            if 'Number' in tire.index and int(tire['Number']) == int(new_rf):
                                target_tire = tire
                                break

                        if target_tire is not None:
                            solution[set_idx][f'{corner}_data'] = target_tire
                            changes_detected = True

                if changes_detected:
                    # Recalculate metrics for affected sets
                    for set_idx in range(len(solution)):
                        metrics = compute_set_metrics(
                            solution[set_idx]['lf_data'],
                            solution[set_idx]['rf_data'],
                            solution[set_idx]['lr_data'],
                            solution[set_idx]['rr_data']
                        )
                        solution[set_idx].update(metrics)

                    st.session_state.solution = solution
                    st.rerun()
        else:
            # --- NORMAL CARD VIEW ---
            n_cols = min(len(solution), 5)
            cols = st.columns(n_cols)

            for set_idx, s in enumerate(solution):
                col_idx = set_idx % n_cols
                with cols[col_idx]:
                    with st.container(border=True):
                        # Set header — clickable for set-level swap
                        is_set_sel = selected_set == set_idx
                        if st.button(f"Set {set_idx + 1}", key=f"setswap_{set_idx}",
                                     use_container_width=True,
                                     type="primary" if is_set_sel else "secondary"):
                            if selected_set is None:
                                st.session_state.selected_set = set_idx
                                st.session_state.selected_tire = None
                            elif is_set_sel:
                                st.session_state.selected_set = None
                            else:
                                do_set_swap(selected_set, set_idx)
                            st.rerun()

                        # Stats bar
                        st.markdown(
                            f'<div class="car-stats">'
                            f'R.Stag: {s["stagger"]:.1f} &nbsp;|&nbsp; '
                            f'F.Stag: {s.get("front_stagger", 0):.1f} &nbsp;|&nbsp; '
                            f'Cross: {s["cross"]*100:.2f}%'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        # 2x2 tire grid — matches car layout
                        tire_rows = [
                            ('LF', 'lf_data', 'RF', 'rf_data'),
                            ('LR', 'lr_data', 'RR', 'rr_data'),
                        ]
                        for l_corner, l_key, r_corner, r_key in tire_rows:
                            left_col, right_col = st.columns(2)
                            for col, corner, tire_key in [(left_col, l_corner, l_key), (right_col, r_corner, r_key)]:
                                with col:
                                    is_sel = selected == (set_idx, corner)
                                    st.markdown(
                                        render_tire_html(s[tire_key], corner, highlight=is_sel,
                                                       road_course=is_road_results, compact=False),
                                        unsafe_allow_html=True
                                    )
                                    btn_type = "primary" if is_sel else "secondary"
                                    if st.button(corner, key=f"tire_{set_idx}_{corner}",
                                                 use_container_width=True, type=btn_type):
                                        st.session_state.selected_set = None
                                        if selected is None:
                                            st.session_state.selected_tire = (set_idx, corner)
                                        elif is_sel:
                                            st.session_state.selected_tire = None
                                        else:
                                            # Enforce pool constraints on manual swaps
                                            from_corner = selected[1]
                                            if is_road_results:
                                                pool_a = {'LR', 'RF'}
                                                same_pool = (from_corner in pool_a) == (corner in pool_a)
                                            else:
                                                left_pool = {'LF', 'LR'}
                                                same_pool = (from_corner in left_pool) == (corner in left_pool)
                                            if same_pool:
                                                do_swap(selected[0], selected[1], set_idx, corner)
                                            else:
                                                st.session_state.selected_tire = None
                                                st.toast(f"Can't swap {from_corner} with {corner} — different pools")
                                        st.rerun()

        st.divider()

        # --- Reference Table ---
        st.subheader("Reference Table")

        ref_df = build_reference_table(solution)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sort: Softest Rear → Stiffest", use_container_width=True):
                ref_df = ref_df.sort_values('Rear Avg Rate')
        with col2:
            if st.button("Sort: Newest → Oldest", use_container_width=True):
                ref_df = ref_df.sort_values('Newest Date', ascending=False)

        st.dataframe(
            ref_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Set": st.column_config.NumberColumn("Set #", width="small"),
                "Rear Avg Rate": st.column_config.NumberColumn("Rear Avg", format="%.1f"),
                "Front Avg Rate": st.column_config.NumberColumn("Front Avg", format="%.1f"),
                "Stagger": st.column_config.NumberColumn("Stagger", format="%.1f"),
                "Front Stagger": st.column_config.NumberColumn("F.Stag", format="%.1f"),
                "Cross %": st.column_config.NumberColumn("Cross %", format="%.2f"),
            }
        )

        st.divider()

        # --- Copy Tire Numbers to Clipboard ---
        # Detect the tire ID column name (Number, Seq#, ID, etc.)
        sample_tire = solution[0]['lf_data']
        id_col = None
        for candidate in ['Number', 'number', 'Seq#', 'ID', 'Ref', 'Tire Number', 'Tire_Number']:
            if candidate in sample_tire.index:
                id_col = candidate
                break

        if id_col is None:
            st.warning(f"Could not find tire ID column. Available columns: {list(sample_tire.index)}")
        else:
            # Build 2-column clipboard text: Left | Right per row
            # Each set = 2 rows: LF/RF then LR/RR, ordered by set number
            clip_rows = []
            for idx, s in enumerate(solution):
                lf_num = int(s['lf_data'][id_col])
                rf_num = int(s['rf_data'][id_col])
                lr_num = int(s['lr_data'][id_col])
                rr_num = int(s['rr_data'][id_col])
                if idx > 0:
                    clip_rows.append("")  # blank row between sets
                clip_rows.append(f"{lf_num}\t{rf_num}")
                clip_rows.append(f"{lr_num}\t{rr_num}")
            clip_text = "\n".join(clip_rows)

            # JavaScript clipboard copy button
            escaped = clip_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            copy_js = f"""
            <button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{this.innerText='Copied!';setTimeout(()=>this.innerText='Copy Tire Numbers to Clipboard',2000)}})"
                style="width:100%;padding:10px 20px;font-size:16px;font-weight:600;
                       background:#2e7d32;color:white;border:none;border-radius:8px;
                       cursor:pointer;">
                Copy Tire Numbers to Clipboard
            </button>
            """
            components.html(copy_js, height=50)

            with st.expander("View / Download"):
                st.code(clip_text, language=None)
                st.download_button(
                    "Download as TXT",
                    data=clip_text,
                    file_name="tire_numbers.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    else:
        st.info("Configure settings in **Setup & Sort**, then run the sort.")


# ============== FOOTER ==============
st.sidebar.divider()
st.sidebar.caption("TRK Tire Sorter v1.0")
