#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRK Damper DOE Viewer v1.0
A Streamlit application for analyzing damper setup data with interactive filtering and visualization.

Requires:
    pip install streamlit pandas matplotlib plotly adjustText

Run:
    streamlit run "Damper DOE Viewer v1.py"
"""
from __future__ import annotations

import base64
import traceback
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline
# ----------------------------- Configuration ---------------------------------- #
LOGO_PATH = r"C:\Users\chavely\OneDrive - Trackhouse Entertainment Group, LLC\Pictures\Logos\TH_FullLogo_White.png"

st.set_page_config(
    page_title="TRK Damper DOE Viewer",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished look
st.markdown("""
    <style>
    .main > div {
        padding-top: 0.5rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stPlotlyChart {
        background-color: transparent;
    }
    h1 {
        color: #004E89;
        font-weight: 600;
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #004E89;
        font-weight: 600;
    }
    h3 {
        color: #1A659E;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.3rem;
    }
    hr {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stMarkdown {
        margin-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- Data Model --------------------------------- #
@dataclass
class ParsedCols:
    metrics: List[str]
    segments: List[str]
    types: List[str]
    triplets: List[Tuple[str, str, str]]  # (metric, segment, type)

@st.cache_data
def parse_metric_columns(df: pd.DataFrame) -> ParsedCols:
    """Parse columns in format: Metric_Segment_Type"""
    metrics, segments, types, triplets = [], [], [], []
    seen_m, seen_s, seen_t = set(), set(), set()
    for col in df.columns:
        if not isinstance(col, str):
            continue
        parts = col.rsplit("_", 2)
        if len(parts) != 3:
            continue
        m, s, t = parts
        if pd.api.types.is_numeric_dtype(df[col]):
            triplets.append((m, s, t))
            if m not in seen_m:
                metrics.append(m)
                seen_m.add(m)
            if s not in seen_s:
                segments.append(s)
                seen_s.add(s)
            if t not in seen_t:
                types.append(t)
                seen_t.add(t)
    metrics.sort()
    segments.sort()
    types.sort()
    triplets.sort()
    return ParsedCols(metrics, segments, types, triplets)

def build_col_name(metric: str, segment: str, dtype: str) -> str:
    """Build column name from components"""
    return f"{metric}_{segment}_{dtype}"

def add_tireforce_all(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-add averaged TireForceVariationZAll columns"""
    groups: Dict[Tuple[str, str], List[str]] = {}
    for col in df.columns:
        if not isinstance(col, str):
            continue
        if not col.startswith("TireForceVariationZ"):
            continue
        parts = col.rsplit("_", 2)
        if len(parts) != 3:
            continue
        base, seg, typ = parts
        groups.setdefault((seg, typ), []).append(col)

    for (seg, typ), cols in groups.items():
        if len(cols) >= 1:
            new_col = f"TireForceVariationZAll_{seg}_{typ}"
            if new_col not in df.columns:
                df[new_col] = df[cols].mean(axis=1, skipna=True)
    return df

def find_damper_click_cols(df: pd.DataFrame) -> List[str]:
    """Find all damper click columns"""
    keywords = ("Clicks LF", "Clicks RF", "Clicks LR", "Clicks RR")
    return sorted([
        c for c in df.columns
        if isinstance(c, str)
        and any(k in c for k in keywords)
        and pd.api.types.is_numeric_dtype(df[c])
    ])

def categorize_damper_columns(columns: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Categorize damper columns by type, corner, and speed"""
    categories = {
        'type': {'Compression': [], 'Rebound': []},
        'corner': {'LF': [], 'RF': [], 'LR': [], 'RR': []},
        'speed': {'Low Speed': [], 'High Speed': [], 'Blowoff': []}
    }

    for col in columns:
        col_lower = col.lower()

        # Categorize by type
        if 'compression' in col_lower:
            categories['type']['Compression'].append(col)
        elif 'rebound' in col_lower:
            categories['type']['Rebound'].append(col)

        # Categorize by corner
        if 'lf' in col_lower:
            categories['corner']['LF'].append(col)
        elif 'rf' in col_lower:
            categories['corner']['RF'].append(col)
        elif 'lr' in col_lower:
            categories['corner']['LR'].append(col)
        elif 'rr' in col_lower:
            categories['corner']['RR'].append(col)

        # Categorize by speed
        if 'low spd' in col_lower or 'low speed' in col_lower:
            categories['speed']['Low Speed'].append(col)
        elif 'high spd' in col_lower or 'high speed' in col_lower:
            categories['speed']['High Speed'].append(col)
        elif 'blowoff' in col_lower:
            categories['speed']['Blowoff'].append(col)

    return categories

def filter_damper_columns(
    columns: List[str],
    damper_type: Optional[str] = None,
    corner: Optional[str] = None,
    speed: Optional[str] = None
) -> List[str]:
    """Filter damper columns based on type, corner, and speed"""
    if not any([damper_type, corner, speed]):
        return columns

    filtered = columns.copy()

    if damper_type and damper_type != "All":
        filtered = [c for c in filtered if damper_type.lower() in c.lower()]

    if corner and corner != "All":
        filtered = [c for c in filtered if corner.lower() in c.lower()]

    if speed and speed != "All":
        speed_mapping = {
            "Low Speed": ["low spd", "low speed"],
            "High Speed": ["high spd", "high speed"],
            "Blowoff": ["blowoff"]
        }
        if speed in speed_mapping:
            filtered = [c for c in filtered if any(s in c.lower() for s in speed_mapping[speed])]

    return sorted(filtered)

# ----------------------------- Plotting Functions ----------------------------- #
# Extended color palette for discrete legends (24 distinct colors)
DISCRETE_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
    "#dcbeff",  # lavender
    "#9a6324",  # brown
    "#fffac8",  # beige
    "#800000",  # maroon
    "#aaffc3",  # mint
    "#808000",  # olive
    "#ffd8b1",  # apricot
    "#000075",  # navy
    "#a9a9a9",  # gray
    "#000000",  # black
    "#e6beff",  # light purple
    "#aa6e28",  # tan
    "#808080",  # dark gray
]

# Original ROYGBIV for gradients
ROYGBIV_COLORS = [
    "#ff0000", "#ff7f00", "#ffff00", "#00ff00",
    "#007fff", "#4b0082", "#8b00ff"
]

def add_polynomial_trendline(fig: go.Figure, x_data: np.ndarray, y_data: np.ndarray, order: int = 3):
    """Add a polynomial trendline to a plotly figure"""
    # Fit polynomial
    coeffs = np.polyfit(x_data, y_data, order)
    poly = np.poly1d(coeffs)

    # Generate smooth line
    x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
    y_smooth = poly(x_smooth)

    # Add trendline to figure
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name=f'Polynomial (order {order})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Trendline<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        )
    )

    return fig

def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    label_col: Optional[str] = None,
    title: str = "",
    show_labels: bool = True,
    point_size: int = 12,
    show_trendline: bool = False,
    show_stats: bool = False,
    polynomial_order: Optional[int] = None,
) -> go.Figure:
    """Create an interactive scatter plot with Plotly"""
    # Clean data
    use_cols = [x_col, y_col]
    if color_col:
        use_cols.append(color_col)
    if label_col:
        use_cols.append(label_col)

    clean = df.dropna(subset=use_cols).copy()

    if clean.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # Create figure with trendline if requested
    trendline_param = "ols" if show_trendline else None

    if color_col and color_col in clean.columns:
        # Get unique values and sort them for consistent coloring
        unique_vals = sorted(clean[color_col].dropna().unique())

        # Create discrete color map using extended palette (24 colors)
        n_colors = len(DISCRETE_COLORS)
        color_map = {}
        for i, val in enumerate(unique_vals):
            color_map[val] = DISCRETE_COLORS[i % n_colors]

        # Convert to string for discrete legend with scatter points
        clean['_color_cat'] = clean[color_col].astype(str)

        # Create figure with discrete colors (shows legend with scatter points)
        fig = px.scatter(
            clean,
            x=x_col,
            y=y_col,
            color='_color_cat',
            hover_data=use_cols,
            title=title,
            trendline=trendline_param,
            category_orders={'_color_cat': [str(v) for v in unique_vals]},
            color_discrete_map={str(v): color_map[v] for v in unique_vals},
        )

        # Rename legend title to original column name
        fig.update_layout(legend_title_text=color_col)
    else:
        fig = px.scatter(
            clean,
            x=x_col,
            y=y_col,
            hover_data=use_cols,
            title=title,
            trendline=trendline_param,
        )

    # Update marker size
    fig.update_traces(marker=dict(size=point_size))

    # Add labels if specified and enabled
    if show_labels and label_col and label_col in clean.columns:
        # Limit labels to avoid overcrowding
        max_labels = 50
        sample_df = clean.sample(n=min(len(clean), max_labels)) if len(clean) > max_labels else clean

        for _, row in sample_df.iterrows():
            label_text = str(row[label_col])
            if label_text and label_text.lower() != 'nan':
                fig.add_annotation(
                    x=row[x_col],
                    y=row[y_col],
                    text=label_text,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="gray",
                    ax=20,
                    ay=-20,
                    font=dict(size=12),
                )

    # Add statistics if requested
    if show_stats:
        x_data = clean[x_col].dropna()
        y_data = clean[y_col].dropna()

        # Calculate correlation
        if len(x_data) > 1 and len(y_data) > 1:
            corr = np.corrcoef(x_data, y_data)[0, 1]
            stats_text = f"Correlation: {corr:.3f}<br>N: {len(clean)}<br>X: μ={x_data.mean():.2f}, σ={x_data.std():.2f}<br>Y: μ={y_data.mean():.2f}, σ={y_data.std():.2f}"

            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=13),
                align="left",
                xanchor="left",
                yanchor="top",
            )

    # Add polynomial trendline if requested
    if polynomial_order is not None:
        x_data = clean[x_col].dropna().values
        y_data = clean[y_col].dropna().values
        if len(x_data) > polynomial_order + 1:  # Need more points than order
            fig = add_polynomial_trendline(fig, x_data, y_data, order=polynomial_order)

    # Update layout with larger fonts
    fig.update_layout(
        template="plotly_white",
        hovermode='closest',
        height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        font=dict(size=14),
        xaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=13),
        ),
        yaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=13),
        ),
    )

    return fig

def create_multi_scatter_with_filter(
    df: pd.DataFrame,
    y_cols: List[str],
    plot_titles: List[str],
    filtered_indices: Optional[Set[int]] = None,
) -> go.Figure:
    """Create 2x2 grid of scatter plots showing filtered vs all data"""
    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=plot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Add scatter plots
    for i, y_col in enumerate(y_cols):
        if y_col not in df.columns:
            continue

        row = (i // 2) + 1
        col = (i % 2) + 1

        clean_all = df.dropna(subset=[y_col])

        if not clean_all.empty:
            # Plot all data (dimmed)
            fig.add_trace(
                go.Scatter(
                    x=clean_all.index,
                    y=clean_all[y_col],
                    mode='markers',
                    marker=dict(size=6, color='lightgray', opacity=0.3),
                    name='All data' if i == 0 else '',
                    showlegend=(i == 0),
                    hoverinfo='skip',
                ),
                row=row, col=col
            )

            # Plot filtered/selected data (highlighted)
            if filtered_indices:
                filtered_data = clean_all[clean_all.index.isin(filtered_indices)]
                if not filtered_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data.index,
                            y=filtered_data[y_col],
                            mode='markers',
                            marker=dict(size=8, color='#FF6B35', opacity=0.8),
                            name='Selected' if i == 0 else '',
                            showlegend=(i == 0),
                            customdata=filtered_data.index,
                            hovertemplate=f'Index: %{{x}}<br>{y_col}: %{{y:.2f}}<extra></extra>',
                        ),
                        row=row, col=col
                    )
            else:
                # No filter applied, show all data highlighted
                fig.add_trace(
                    go.Scatter(
                        x=clean_all.index,
                        y=clean_all[y_col],
                        mode='markers',
                        marker=dict(size=8, color='#4A90E2', opacity=0.6),
                        name='All data' if i == 0 else '',
                        showlegend=(i == 0),
                        customdata=clean_all.index,
                        hovertemplate=f'Index: %{{x}}<br>{y_col}: %{{y:.2f}}<extra></extra>',
                    ),
                    row=row, col=col
                )

        # Update axes
        fig.update_xaxes(title_text="Row Index", row=row, col=col)
        fig.update_yaxes(title_text=y_col, row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        height=800,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
        dragmode='select',  # Enable box select by default
        hovermode='closest',
    )

    return fig

def compute_damper_averages(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[int]]]:
    """Compute average damper clicks for each corner"""
    corners = ["LF", "RF", "LR", "RR"]
    kinds = {
        "LSC": "Damper Compression Low Spd Clicks {corner}",
        "HSC": "Damper Compression High Spd Clicks {corner}",
        "LSR": "Damper Rebound Low Spd Clicks {corner}",
        "HSR": "Damper Rebound High Spd Clicks {corner}",
    }

    vals = {c: {k: None for k in kinds} for c in corners}

    for corner in corners:
        for kind, pattern in kinds.items():
            # Find matching column
            pattern_lower = pattern.format(corner=corner).lower()
            matching_col = None
            for col in df.columns:
                if isinstance(col, str) and pattern_lower in col.lower():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        matching_col = col
                        break

            if matching_col:
                mean_val = df[matching_col].mean(skipna=True)
                if pd.notna(mean_val):
                    vals[corner][kind] = int(round(mean_val))

    return vals

# ----------------------------- Helper Functions ------------------------------- #
def get_base64_image(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ----------------------------- DOE Batch Builder Functions -------------------- #
# Damper click constraints
DAMPER_CONSTRAINTS = {
    'low_speed': {'min': 0, 'max': 60},
    'high_speed': {'min': 0, 'max': 40},
    'low_speed_rebound': {'min': 6, 'max': 60},  # Special case: LSR can't go below 6
}

# Damper parameter definitions: (short_id, display_name, constraint_type, search_patterns)
# Search patterns: list of terms that must ALL appear in column name (case-insensitive)
DAMPER_PARAMS = [
    ('LSC_LF', 'Low Spd Comp LF', 'low_speed', ['compression', 'low', 'lf']),
    ('LSC_RF', 'Low Spd Comp RF', 'low_speed', ['compression', 'low', 'rf']),
    ('LSC_LR', 'Low Spd Comp LR', 'low_speed', ['compression', 'low', 'lr']),
    ('LSC_RR', 'Low Spd Comp RR', 'low_speed', ['compression', 'low', 'rr']),
    ('HSC_LF', 'High Spd Comp LF', 'high_speed', ['compression', 'high', 'lf']),
    ('HSC_RF', 'High Spd Comp RF', 'high_speed', ['compression', 'high', 'rf']),
    ('HSC_LR', 'High Spd Comp LR', 'high_speed', ['compression', 'high', 'lr']),
    ('HSC_RR', 'High Spd Comp RR', 'high_speed', ['compression', 'high', 'rr']),
    ('LSR_LF', 'Low Spd Reb LF', 'low_speed_rebound', ['rebound', 'low', 'lf']),
    ('LSR_RF', 'Low Spd Reb RF', 'low_speed_rebound', ['rebound', 'low', 'rf']),
    ('LSR_LR', 'Low Spd Reb LR', 'low_speed_rebound', ['rebound', 'low', 'lr']),
    ('LSR_RR', 'Low Spd Reb RR', 'low_speed_rebound', ['rebound', 'low', 'rr']),
    ('HSR_LF', 'High Spd Reb LF', 'high_speed', ['rebound', 'high', 'lf']),
    ('HSR_RF', 'High Spd Reb RF', 'high_speed', ['rebound', 'high', 'rf']),
    ('HSR_LR', 'High Spd Reb LR', 'high_speed', ['rebound', 'high', 'lr']),
    ('HSR_RR', 'High Spd Reb RR', 'high_speed', ['rebound', 'high', 'rr']),
]

def extract_baseline_clicks(df: pd.DataFrame) -> Tuple[Dict[str, Optional[int]], Dict[str, str]]:
    """
    Extract baseline damper click values from a dataframe.
    Returns:
        - baseline_values: {short_id: value}
        - column_names: {short_id: actual_column_name}
    """
    # Terms that should exclude a column from matching
    EXCLUDE_TERMS = ['blowoff', 'blow-off', 'blow off']

    baseline_values = {}
    column_names = {}

    for param_id, _, _, patterns in DAMPER_PARAMS:
        baseline_values[param_id] = None
        column_names[param_id] = None

        for col in df.columns:
            if not isinstance(col, str):
                continue
            col_lower = col.lower()

            # Skip columns with excluded terms
            if any(excl in col_lower for excl in EXCLUDE_TERMS):
                continue

            # Check if all patterns match AND contains 'click'
            if all(p in col_lower for p in patterns) and 'click' in col_lower:
                # Found matching column
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].iloc[0] if len(df) > 0 else None
                    if pd.notna(val):
                        baseline_values[param_id] = int(round(val))
                        column_names[param_id] = col  # Store actual column name
                break

    return baseline_values, column_names

def apply_constraint(value: int, constraint_type: str) -> int:
    """Apply damper click constraints to a value"""
    constraints = DAMPER_CONSTRAINTS[constraint_type]
    return max(constraints['min'], min(constraints['max'], value))

def generate_doe_matrix(
    baseline_values: Dict[str, int],
    column_names: Dict[str, str],
    param_configs: Dict[str, Dict],
    sampling_method: str = "Full Factorial",
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Generate DOE matrix from baseline and parameter configurations.

    Args:
        baseline_values: {short_id: baseline_value}
        column_names: {short_id: actual_column_name}
        param_configs: {short_id: {'delta_min', 'delta_max', 'increment', 'constraint_type'}}
        sampling_method: "Full Factorial" or "Sobol"
        n_samples: Number of samples for Sobol method
    """
    # Build value ranges for each parameter
    param_ranges = {}  # {short_id: (min_val, max_val, [discrete_values])}
    param_order = []

    for param_id, config in param_configs.items():
        if param_id not in baseline_values or baseline_values[param_id] is None:
            continue

        baseline = baseline_values[param_id]
        delta_min = config['delta_min']
        delta_max = config['delta_max']
        increment = config['increment']
        constraint_type = config['constraint_type']

        if increment <= 0:
            continue

        # Generate discrete values from baseline + delta_min to baseline + delta_max
        values = []
        current = baseline + delta_min
        while current <= baseline + delta_max:
            constrained_val = apply_constraint(current, constraint_type)
            if constrained_val not in values:
                values.append(constrained_val)
            current += increment

        if values:
            values = sorted(set(values))
            param_ranges[param_id] = (min(values), max(values), values)
            param_order.append(param_id)

    if not param_ranges:
        return pd.DataFrame()

    # Generate combinations based on sampling method
    if sampling_method == "Full Factorial":
        # Full factorial: all combinations
        all_values = [param_ranges[p][2] for p in param_order]
        all_combinations = list(itertools.product(*all_values))
        data = {param_order[i]: [c[i] for c in all_combinations] for i in range(len(param_order))}

    else:  # Sobol
        try:
            from scipy.stats import qmc
            # Generate Sobol sequence
            n_dims = len(param_order)
            sampler = qmc.Sobol(d=n_dims, scramble=True)
            samples = sampler.random(n_samples)

            # Map [0,1] samples to discrete values for each parameter
            data = {}
            for i, param_id in enumerate(param_order):
                discrete_vals = param_ranges[param_id][2]
                n_levels = len(discrete_vals)
                # Map continuous [0,1] to discrete index
                indices = (samples[:, i] * n_levels).astype(int)
                indices = np.clip(indices, 0, n_levels - 1)
                data[param_id] = [discrete_vals[idx] for idx in indices]
        except ImportError:
            # Fallback to random sampling if scipy.stats.qmc not available
            data = {}
            for param_id in param_order:
                discrete_vals = param_ranges[param_id][2]
                data[param_id] = list(np.random.choice(discrete_vals, size=n_samples))

    # Create dataframe with SHORT IDs first
    doe_df = pd.DataFrame(data)

    # Add Sim ID column
    doe_df.insert(0, 'Sim ID', range(1, len(doe_df) + 1))

    # Add baseline values for parameters not being varied
    for param_id, _, _, _ in DAMPER_PARAMS:
        if param_id not in doe_df.columns and param_id in baseline_values and baseline_values[param_id] is not None:
            doe_df[param_id] = baseline_values[param_id]

    # Now rename columns to actual column names for export
    rename_map = {}
    for param_id in doe_df.columns:
        if param_id != 'Sim ID' and param_id in column_names and column_names[param_id] is not None:
            rename_map[param_id] = column_names[param_id]

    doe_df_export = doe_df.rename(columns=rename_map)

    return doe_df_export

# ----------------------------- Main Application ------------------------------- #
def main():
    st.title("Damper DOE Viewer")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'parsed' not in st.session_state:
        st.session_state.parsed = None
    if 'baseline_values' not in st.session_state:
        st.session_state.baseline_values = None
    if 'baseline_col_names' not in st.session_state:
        st.session_state.baseline_col_names = None
    if 'doe_param_configs' not in st.session_state:
        st.session_state.doe_param_configs = {}
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None

    # Sidebar - File upload and configuration
    with st.sidebar:
        st.header("📁 Data Import")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Validate data
                if df.empty:
                    st.error("⚠️ CSV file is empty")
                    st.session_state.df = None
                    st.session_state.parsed = None
                else:
                    df = add_tireforce_all(df)
                    st.session_state.df = df
                    st.session_state.parsed = parse_metric_columns(df)
                    # Store filename for DOE batch naming
                    st.session_state.uploaded_filename = uploaded_file.name

                    # Show data summary
                    st.success(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")

                    with st.expander("📊 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Rows", len(df))
                        with col_b:
                            st.metric("Total Columns", len(df.columns))
                        with col_c:
                            parsed = st.session_state.parsed
                            st.metric("Metrics Found", len(parsed.metrics) if parsed else 0)

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.session_state.df = None
                st.session_state.parsed = None

        st.markdown("---")
        st.header("⚙️ Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"], index=0)

    # Main content
    if st.session_state.df is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Features:
        - **Single Scatter Analysis**: 4D visualization with color-coded legends and data callouts
        - **Multi-Scatter Filter**: Four synchronized plots with range filtering
        - **Damper Setup Summary**: Average click settings across all corners
        - **Interactive**: Zoom, pan, and hover for detailed data inspection
        """)
        return

    df = st.session_state.df
    parsed = st.session_state.parsed

    # Tab selection
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 DOE Batch Builder", "📊 Single Scatter", "📈 Multi-Scatter", "🌪️ Tornado Plot", "📐 % Delta", "⚖️ Balance"])

    # ========================= TAB 0: DOE Batch Builder ========================= #
    with tab0:
        st.markdown("### DOE Batch Builder")
        st.markdown("Generate a Design of Experiments (DOE) batch for damper tuning using the imported data as baseline.")

        # Use the main imported data for baseline (first row)
        if df is not None and len(df) > 0:
            # Extract baseline from first row of main data
            baseline_values, baseline_col_names = extract_baseline_clicks(df)
            st.session_state.baseline_values = baseline_values
            st.session_state.baseline_col_names = baseline_col_names

            st.markdown("#### Configure DOE Parameters by Corner")
            st.caption("Constraints: Low Speed 0-60 | High Speed 0-40 | Low Speed Rebound 6-60")

            # Group parameters by corner
            corner_params = {
                'LF': [p for p in DAMPER_PARAMS if p[0].endswith('_LF')],
                'RF': [p for p in DAMPER_PARAMS if p[0].endswith('_RF')],
                'LR': [p for p in DAMPER_PARAMS if p[0].endswith('_LR')],
                'RR': [p for p in DAMPER_PARAMS if p[0].endswith('_RR')],
            }

            selected_params = {}

            def render_corner(corner_name, params):
                """Render DOE controls for a corner"""
                with st.expander(f"**{corner_name}**", expanded=False):
                    # Header row
                    col_h1, col_h2, col_h3, col_h4, col_h5, col_h6 = st.columns([1.2, 1, 1, 1, 1, 2])
                    with col_h1:
                        st.caption("**Param**")
                    with col_h2:
                        st.caption("**Δ Min**")
                    with col_h3:
                        st.caption("**Base**")
                    with col_h4:
                        st.caption("**Δ Max**")
                    with col_h5:
                        st.caption("**Inc**")
                    with col_h6:
                        st.caption("**Preview**")

                    for param_id, display_name, constraint_type, _ in params:
                        baseline_val = baseline_values.get(param_id)
                        if baseline_val is None:
                            st.caption(f"{param_id}: No baseline")
                            continue

                        damper_type = param_id.split('_')[0]
                        constraints = DAMPER_CONSTRAINTS[constraint_type]
                        c_min, c_max = constraints['min'], constraints['max']

                        # Smart defaults: 24 click range, shifted if constrained
                        target_range = 24
                        default_inc = 4 if constraint_type == 'high_speed' else 3

                        ideal_min_val = baseline_val - target_range // 2
                        ideal_max_val = baseline_val + target_range // 2

                        if ideal_min_val < c_min:
                            final_min_val = c_min
                            final_max_val = min(c_min + target_range, c_max)
                        elif ideal_max_val > c_max:
                            final_max_val = c_max
                            final_min_val = max(c_max - target_range, c_min)
                        else:
                            final_min_val = ideal_min_val
                            final_max_val = ideal_max_val

                        default_delta_min = final_min_val - baseline_val
                        default_delta_max = final_max_val - baseline_val

                        col_check, col_min, col_base, col_max, col_inc, col_preview = st.columns([1.2, 1, 1, 1, 1, 2])

                        with col_check:
                            include = st.checkbox(f"{damper_type}", key=f"doe_include_{param_id}")

                        with col_min:
                            delta_min = st.number_input("min", value=default_delta_min, step=1,
                                key=f"doe_delta_min_{param_id}", disabled=not include, label_visibility="collapsed")

                        with col_base:
                            st.markdown(f"**{baseline_val}**")

                        with col_max:
                            delta_max = st.number_input("max", value=default_delta_max, step=1,
                                key=f"doe_delta_max_{param_id}", disabled=not include, label_visibility="collapsed")

                        with col_inc:
                            increment = st.number_input("inc", value=default_inc, min_value=1, step=1,
                                key=f"doe_increment_{param_id}", disabled=not include, label_visibility="collapsed")

                        with col_preview:
                            if include and increment > 0:
                                preview_vals = []
                                current = baseline_val + delta_min
                                while current <= baseline_val + delta_max:
                                    constrained = apply_constraint(current, constraint_type)
                                    if constrained not in preview_vals:
                                        preview_vals.append(constrained)
                                    current += increment
                                preview_vals = sorted(set(preview_vals))
                                st.caption(f"{preview_vals}")

                                selected_params[param_id] = {
                                    'delta_min': delta_min,
                                    'delta_max': delta_max,
                                    'increment': increment,
                                    'constraint_type': constraint_type,
                                }

            # 2x2 grid layout like a car
            # Row 1: Front (LF, RF)
            col_lf, col_rf = st.columns(2)
            with col_lf:
                render_corner('LF', corner_params['LF'])
            with col_rf:
                render_corner('RF', corner_params['RF'])

            # Row 2: Rear (LR, RR)
            col_lr, col_rr = st.columns(2)
            with col_lr:
                render_corner('LR', corner_params['LR'])
            with col_rr:
                render_corner('RR', corner_params['RR'])

            st.markdown("---")
            st.markdown("#### Generate DOE Matrix")

            # Sampling method selection
            col_method, col_samples = st.columns([2, 1])
            with col_method:
                sampling_method = st.selectbox(
                    "Sampling Method",
                    ["Full Factorial", "Sobol"],
                    key='doe_sampling_method',
                    help="Full Factorial: all combinations. Sobol: quasi-random sampling for large spaces."
                )
            with col_samples:
                n_samples = st.number_input(
                    "# Samples (Sobol)",
                    value=100,
                    min_value=10,
                    max_value=10000,
                    step=10,
                    key='doe_n_samples',
                    disabled=(sampling_method != "Sobol")
                )

            if selected_params:
                # Calculate total combinations for full factorial
                total_combos = 1
                for param_id, config in selected_params.items():
                    baseline_val = baseline_values.get(param_id, 0)
                    n_vals = len(set([
                        apply_constraint(baseline_val + d, config['constraint_type'])
                        for d in range(config['delta_min'], config['delta_max'] + 1, config['increment'])
                    ]))
                    total_combos *= max(1, n_vals)

                if sampling_method == "Full Factorial":
                    st.info(f"**{len(selected_params)}** parameters selected → **{total_combos}** total combinations")
                else:
                    st.info(f"**{len(selected_params)}** parameters selected → **{n_samples}** Sobol samples (from {total_combos} possible)")

                if st.button("🚀 Generate DOE Matrix", key='generate_doe'):
                    doe_df = generate_doe_matrix(
                        baseline_values,
                        baseline_col_names,
                        selected_params,
                        sampling_method=sampling_method,
                        n_samples=n_samples
                    )

                    if doe_df.empty:
                        st.warning("⚠️ No valid DOE matrix could be generated. Check your parameters.")
                    else:
                        st.success(f"✓ Generated DOE matrix with {len(doe_df)} simulations")

                        # Create box and whiskers plot for varying parameters
                        # Get columns that are actually varying (exclude Sim ID and constants)
                        varying_cols = []
                        for col in doe_df.columns:
                            if col == 'Sim ID':
                                continue
                            if doe_df[col].nunique() > 1:
                                varying_cols.append(col)

                        if varying_cols:
                            # Create short labels for display
                            short_labels = []
                            for col in varying_cols:
                                # Extract corner and type from column name
                                col_lower = col.lower()
                                corner = ""
                                for c in ["LF", "RF", "LR", "RR"]:
                                    if c.lower() in col_lower:
                                        corner = c
                                        break
                                speed = "LS" if ("low spd" in col_lower or "low speed" in col_lower) else "HS"
                                dtype = "C" if "compression" in col_lower else "R"
                                short_labels.append(f"{corner} {speed}{dtype}")

                            # Create box plot
                            fig = go.Figure()
                            for i, (col, label) in enumerate(zip(varying_cols, short_labels)):
                                fig.add_trace(go.Box(
                                    y=doe_df[col],
                                    name=label,
                                    boxpoints='outliers',
                                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                                    hovertemplate=f'<b>{label}</b><br>Value: %{{y}}<extra></extra>'
                                ))

                            fig.update_layout(
                                title="DOE Parameter Ranges",
                                yaxis_title="Click Value",
                                template="plotly_white",
                                height=400,
                                showlegend=False,
                                margin=dict(l=60, r=40, t=60, b=60),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Download button - auto-name based on uploaded file
                        # Remove Sim ID column for export
                        export_df = doe_df.drop(columns=['Sim ID'], errors='ignore')
                        csv_data = export_df.to_csv(index=False)

                        # Determine which corners are being varied
                        corners_varied = set()
                        for param_id in selected_params.keys():
                            if param_id.endswith('_LF'):
                                corners_varied.add('LF')
                            elif param_id.endswith('_RF'):
                                corners_varied.add('RF')
                            elif param_id.endswith('_LR'):
                                corners_varied.add('LR')
                            elif param_id.endswith('_RR'):
                                corners_varied.add('RR')

                        # Generate corner suffix
                        if corners_varied == {'LF'}:
                            corner_suffix = "_LFDamp"
                        elif corners_varied == {'RF'}:
                            corner_suffix = "_RFDamp"
                        elif corners_varied == {'LR'}:
                            corner_suffix = "_LRDamp"
                        elif corners_varied == {'RR'}:
                            corner_suffix = "_RRDamp"
                        elif corners_varied == {'LF', 'RF'}:
                            corner_suffix = "_FrontDamp"
                        elif corners_varied == {'LR', 'RR'}:
                            corner_suffix = "_RearDamp"
                        elif corners_varied == {'LF', 'LR'}:
                            corner_suffix = "_LeftDamp"
                        elif corners_varied == {'RF', 'RR'}:
                            corner_suffix = "_RightDamp"
                        elif corners_varied == {'LF', 'RF', 'LR', 'RR'}:
                            corner_suffix = "_AllDamp"
                        else:
                            # Mixed combination - list the corners
                            corner_suffix = "_" + "".join(sorted(corners_varied)) + "Damp"

                        # Generate filename from uploaded file
                        base_name = "damper_doe_matrix"
                        if st.session_state.uploaded_filename:
                            # Remove .csv extension
                            base_name = st.session_state.uploaded_filename.replace('.csv', '').replace('.CSV', '')
                            # Strip common suffixes (case-insensitive)
                            if base_name.lower().endswith('_filteredresults'):
                                base_name = base_name[:-16]  # len('_filteredresults') = 16
                            elif base_name.lower().endswith('_filtered'):
                                base_name = base_name[:-9]  # len('_filtered') = 9

                        st.download_button(
                            label="📥 Download DOE Matrix (CSV)",
                            data=csv_data,
                            file_name=f"{base_name}{corner_suffix}.csv",
                            mime="text/csv",
                            key='download_doe'
                        )
            else:
                st.info("👆 Select at least one parameter to vary")

            # Sequential Low Speed Batch Export - One Parameter at a Time
            st.markdown("---")
            st.markdown("#### Quick Export: Sequential Low Speed Sweep")
            st.caption("Sweep each low speed parameter (LSC + LSR) one at a time 0-60, all 4 corners. Others stay at baseline.")

            if st.button("📥 Export Low Speed Sweep", key='export_ls_batch'):
                # Get column names for low speed params
                ls_params = [p for p in DAMPER_PARAMS if p[2] in ['low_speed', 'low_speed_rebound']]

                # Build baseline row dict
                baseline_row = {}
                for param_id, _, _, _ in ls_params:
                    col_name = baseline_col_names.get(param_id)
                    if col_name:
                        baseline_row[col_name] = baseline_values.get(param_id, 0)

                rows = []

                # Row 0: Baseline values
                rows.append(baseline_row.copy())

                # For each parameter, sweep 0-60 while others stay at baseline
                for sweep_param_id, _, constraint_type, _ in ls_params:
                    sweep_col_name = baseline_col_names.get(sweep_param_id)
                    if not sweep_col_name:
                        continue

                    # Sweep this parameter from 0 to 60
                    for click_val in range(0, 61):
                        row = baseline_row.copy()  # Start with baseline for all
                        # Apply constraint and set the sweeping parameter
                        constrained = apply_constraint(click_val, constraint_type)
                        row[sweep_col_name] = constrained
                        rows.append(row)

                ls_batch_df = pd.DataFrame(rows)

                # Generate filename
                base_name = "LowSpeed_Sweep"
                if st.session_state.uploaded_filename:
                    base_name = st.session_state.uploaded_filename.replace('.csv', '').replace('.CSV', '')
                    if base_name.lower().endswith('_filteredresults'):
                        base_name = base_name[:-16]
                    elif base_name.lower().endswith('_filtered'):
                        base_name = base_name[:-9]
                    base_name = f"{base_name}_LSSweep"

                csv_data = ls_batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Low Speed Sweep",
                    data=csv_data,
                    file_name=f"{base_name}.csv",
                    mime="text/csv",
                    key='download_ls_batch'
                )
                st.success(f"✓ Generated {len(ls_batch_df)} rows (1 baseline + 8 params × 61 values = 489 runs)")

    # ========================= TAB 1: Single Scatter ========================= #
    with tab1:
        with st.expander("⚙️ Axis Configuration", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**X-Axis**")
                x_metric = st.selectbox("Metric", parsed.metrics, key='x_metric')
                x_segment = st.selectbox("Segment", parsed.segments, key='x_segment')
                x_type = st.selectbox("Type", parsed.types, key='x_type')

                # Option to use click columns for X
                use_click_x = st.checkbox("Use damper clicks for X", key='use_click_x')
                if use_click_x:
                    click_cols = find_damper_click_cols(df)
                    x_click_col = st.selectbox("Click column", click_cols, key='x_click')
                    x_col = x_click_col
                else:
                    x_col = build_col_name(x_metric, x_segment, x_type)

            with col2:
                st.markdown("**Y-Axis**")
                y_metric = st.selectbox("Metric", parsed.metrics, key='y_metric')
                y_segment = st.selectbox("Segment", parsed.segments, key='y_segment')
                y_type = st.selectbox("Type", parsed.types, key='y_type')

                # Option to use click columns for Y
                use_click_y = st.checkbox("Use damper clicks for Y", key='use_click_y')
                if use_click_y:
                    click_cols = find_damper_click_cols(df)
                    y_click_col = st.selectbox("Click column", click_cols, key='y_click')
                    y_col = y_click_col
                else:
                    y_col = build_col_name(y_metric, y_segment, y_type)

        # Only show damper filters if NOT using damper clicks for X
        if not use_click_x:
            # Damper column filters - side by side
            all_click_cols = find_damper_click_cols(df)

            col_left, col_right = st.columns(2)

            # LEFT: Color by (Legend) filters
            with col_left:
                st.markdown("**Color by (Legend)**")
                color_corner_filter = st.selectbox(
                    "Corner",
                    ["All", "LF", "RF", "LR", "RR"],
                    key='color_corner_filter'
                )
                color_speed_filter = st.selectbox(
                    "Speed",
                    ["All", "Low Speed", "High Speed", "Blowoff"],
                    key='color_speed_filter'
                )
                color_type_filter = st.selectbox(
                    "Type",
                    ["All", "Compression", "Rebound"],
                    key='color_type_filter'
                )

                # Apply filters for color column
                filtered_color_cols = filter_damper_columns(
                    all_click_cols,
                    damper_type=color_type_filter,
                    corner=color_corner_filter,
                    speed=color_speed_filter
                )

                # Auto-select if only one option
                if len(filtered_color_cols) == 1:
                    color_col = filtered_color_cols[0]
                    st.caption(f"✓ {color_col}")
                    avg_by_legend = st.checkbox("Average by legend", key='avg_legend')
                else:
                    color_col = None
                    if filtered_color_cols:
                        st.caption(f"⚠️ {len(filtered_color_cols)} matches - refine filters")
                    else:
                        st.caption("⚠️ No matches")
                    avg_by_legend = False

            # RIGHT: Label points with filters
            with col_right:
                st.markdown("**Label points with**")
                label_corner_filter = st.selectbox(
                    "Corner",
                    ["All", "LF", "RF", "LR", "RR"],
                    key='label_corner_filter'
                )
                label_speed_filter = st.selectbox(
                    "Speed",
                    ["All", "Low Speed", "High Speed", "Blowoff"],
                    key='label_speed_filter'
                )
                label_type_filter = st.selectbox(
                    "Type",
                    ["All", "Compression", "Rebound"],
                    key='label_type_filter'
                )

                # Apply filters for label column
                filtered_label_cols = filter_damper_columns(
                    all_click_cols,
                    damper_type=label_type_filter,
                    corner=label_corner_filter,
                    speed=label_speed_filter
                )

                # Auto-select if only one option
                if len(filtered_label_cols) == 1:
                    label_col = filtered_label_cols[0]
                    st.caption(f"✓ {label_col}")
                    avg_by_callout = st.checkbox("Average by callout", key='avg_callout')
                else:
                    label_col = None
                    if filtered_label_cols:
                        st.caption(f"⚠️ {len(filtered_label_cols)} matches - refine filters")
                    else:
                        st.caption("⚠️ No matches")
                    avg_by_callout = False

            # Process averaging if requested
            plot_df = df.copy()
            if avg_by_legend or avg_by_callout:
                group_cols = []
                if avg_by_legend and color_col:
                    group_cols.append(color_col)
                if avg_by_callout and label_col:
                    group_cols.append(label_col)

                if group_cols and x_col in plot_df.columns and y_col in plot_df.columns:
                    agg_dict = {x_col: 'mean', y_col: 'mean'}
                    plot_df = plot_df.dropna(subset=[x_col, y_col] + group_cols)
                    plot_df = plot_df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        else:
            # Using damper clicks: simple scatter with polynomial trendline
            color_col = None
            label_col = None
            avg_by_legend = False
            avg_by_callout = False
            plot_df = df.copy()

        # Create and display plot
        if x_col in df.columns and y_col in df.columns:
            fig = create_scatter_plot(
                plot_df, x_col, y_col, color_col, label_col, "",
                show_labels=True,
                point_size=8,
                show_trendline=False,
                show_stats=False,
                polynomial_order=3 if use_click_x else None,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Copy to clipboard button
            col_copy, col_spacer = st.columns([1, 4])
            with col_copy:
                if st.button("📋 Copy Plot to Clipboard", key='copy_scatter_plot'):
                    try:
                        # Convert figure to PNG bytes
                        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)

                        # Display as static image (user can right-click copy)
                        st.image(img_bytes, caption="Right-click image → Copy Image", use_container_width=True)

                        # Also provide download
                        st.download_button(
                            label="💾 Download PNG",
                            data=img_bytes,
                            file_name="scatter_plot.png",
                            mime="image/png",
                            key='download_scatter_png'
                        )
                    except Exception as e:
                        st.error(f"Could not generate image. Install kaleido: `pip install kaleido`")
        else:
            st.warning("⚠️ Selected columns not found in dataset")

    # ========================= TAB 2: Multi-Scatter ========================== #
    with tab2:
        with st.expander("⚙️ Plot Configuration", expanded=True):
            # Configure 4 plots
            plot_cols = st.columns(4)
            y_cols = []
            plot_titles = []

            for i in range(4):
                with plot_cols[i]:
                    st.markdown(f"**Plot {i+1}**")
                    metric = st.selectbox(
                        "Metric",
                        parsed.metrics,
                        key=f'multi_metric_{i}',
                        index=min(i, len(parsed.metrics)-1)
                    )
                    segment = st.selectbox(
                        "Segment",
                        parsed.segments,
                        key=f'multi_segment_{i}',
                        index=0
                    )
                    dtype = st.selectbox(
                        "Type",
                        parsed.types,
                        key=f'multi_type_{i}',
                        index=0
                    )
                    col_name = build_col_name(metric, segment, dtype)
                    y_cols.append(col_name)
                    plot_titles.append(f"Plot {i+1}: {col_name}")

        # Initialize session state for selected indices
        if 'selected_indices' not in st.session_state:
            st.session_state.selected_indices = None

        col_info, col_reset = st.columns([3, 1])
        with col_info:
            st.info("💡 **Tip:** Use box select (click & drag) or lasso select (click lasso icon in toolbar) to filter data points. Selections across multiple plots combine with AND logic.")
        with col_reset:
            if st.button("🔄 Reset Selection", key="reset_selection"):
                st.session_state.selected_indices = None
                st.rerun()

        # Create multi-scatter plot with current filter
        fig = create_multi_scatter_with_filter(df, y_cols, plot_titles, st.session_state.selected_indices)

        # Display plot and capture selection events
        selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="multi_scatter_plot")

        # Process selected points
        if selected_data and selected_data.selection and selected_data.selection.point_indices:
            # Get the actual row indices from the selected points
            new_selection = set()

            # The point_indices correspond to the data points in the scatter
            # We need to map them back to the original dataframe indices
            for point_idx in selected_data.selection.point_indices:
                # Map to actual dataframe index
                if point_idx < len(df):
                    new_selection.add(df.index[point_idx])

            if new_selection:
                # AND logic: intersect with existing selection
                if st.session_state.selected_indices is None:
                    st.session_state.selected_indices = new_selection
                else:
                    st.session_state.selected_indices = st.session_state.selected_indices & new_selection

                st.rerun()

        # Apply filter to dataframe
        if st.session_state.selected_indices:
            filtered_df = df.loc[list(st.session_state.selected_indices)]
            st.success(f"✓ {len(filtered_df)} / {len(df)} data points selected")
        else:
            filtered_df = df
            st.info("No selection active - showing all data")

        # Display damper setup summary
        st.markdown("---")
        st.markdown(f"**Damper Setup Summary** - Filtered rows: {len(filtered_df)} / {len(df)}")

        damper_vals = compute_damper_averages(filtered_df)

        # Create visual summary
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### FRONT")
            front_cols = st.columns(2)

            with front_cols[0]:
                st.markdown("**LF (Left Front)**")
                lf = damper_vals['LF']
                st.markdown(f"- **LSC:** {lf['LSC'] if lf['LSC'] is not None else '-'}")
                st.markdown(f"- **HSC:** {lf['HSC'] if lf['HSC'] is not None else '-'}")
                st.markdown(f"- **LSR:** {lf['LSR'] if lf['LSR'] is not None else '-'}")
                st.markdown(f"- **HSR:** {lf['HSR'] if lf['HSR'] is not None else '-'}")

            with front_cols[1]:
                st.markdown("**RF (Right Front)**")
                rf = damper_vals['RF']
                st.markdown(f"- **LSC:** {rf['LSC'] if rf['LSC'] is not None else '-'}")
                st.markdown(f"- **HSC:** {rf['HSC'] if rf['HSC'] is not None else '-'}")
                st.markdown(f"- **LSR:** {rf['LSR'] if rf['LSR'] is not None else '-'}")
                st.markdown(f"- **HSR:** {rf['HSR'] if rf['HSR'] is not None else '-'}")

        with col_right:
            st.markdown("##### REAR")
            rear_cols = st.columns(2)

            with rear_cols[0]:
                st.markdown("**LR (Left Rear)**")
                lr = damper_vals['LR']
                st.markdown(f"- **LSC:** {lr['LSC'] if lr['LSC'] is not None else '-'}")
                st.markdown(f"- **HSC:** {lr['HSC'] if lr['HSC'] is not None else '-'}")
                st.markdown(f"- **LSR:** {lr['LSR'] if lr['LSR'] is not None else '-'}")
                st.markdown(f"- **HSR:** {lr['HSR'] if lr['HSR'] is not None else '-'}")

            with rear_cols[1]:
                st.markdown("**RR (Right Rear)**")
                rr = damper_vals['RR']
                st.markdown(f"- **LSC:** {rr['LSC'] if rr['LSC'] is not None else '-'}")
                st.markdown(f"- **HSC:** {rr['HSC'] if rr['HSC'] is not None else '-'}")
                st.markdown(f"- **LSR:** {rr['LSR'] if rr['LSR'] is not None else '-'}")
                st.markdown(f"- **HSR:** {rr['HSR'] if rr['HSR'] is not None else '-'}")

        # Export filtered data
        st.markdown("---")
        col_download1, col_download2 = st.columns([1, 3])
        with col_download1:
            csv_filtered = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_filtered,
                file_name="filtered_damper_data.csv",
                mime="text/csv",
                help="Download the filtered dataset as CSV"
            )

        # Summary statistics for filtered data
        with st.expander("📊 Filtered Data Statistics"):
            stats_cols = st.columns(len(y_cols))
            for i, y_col in enumerate(y_cols):
                if y_col in filtered_df.columns:
                    with stats_cols[i]:
                        col_data = filtered_df[y_col].dropna()
                        if len(col_data) > 0:
                            st.markdown(f"**{y_col}**")
                            st.metric("Mean", f"{col_data.mean():.2f}")
                            st.metric("Std Dev", f"{col_data.std():.2f}")
                            st.metric("Min", f"{col_data.min():.2f}")
                            st.metric("Max", f"{col_data.max():.2f}")

    # ========================= TAB 3: Tornado Plot ============================ #
    with tab3:
        st.markdown("### Damper Sensitivity Analysis")
        st.markdown("Analyze which damper settings have the most impact on a selected metric.")

        with st.expander("⚙️ Metric Selection", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                tornado_metric = st.selectbox(
                    "Metric",
                    parsed.metrics,
                    key='tornado_metric',
                    index=0
                )

            with col2:
                tornado_segment = st.selectbox(
                    "Segment",
                    parsed.segments,
                    key='tornado_segment',
                    index=0
                )

            with col3:
                tornado_type = st.selectbox(
                    "Type",
                    parsed.types,
                    key='tornado_type',
                    index=0
                )

            # Build the target column
            target_col = build_col_name(tornado_metric, tornado_segment, tornado_type)

            # Direction toggle
            lower_is_better = st.checkbox(
                "Lower is better (e.g., TireForceVariation, LapTime)",
                value=True,
                key='lower_is_better',
                help="Check if lower values of this metric are desirable"
            )

        # Validate target column exists
        if target_col not in df.columns:
            st.warning(f"⚠️ Column '{target_col}' not found in dataset")
        else:
            # Get all damper click columns
            click_cols = find_damper_click_cols(df)

            if not click_cols:
                st.warning("⚠️ No damper click columns found in dataset")
            else:
                # Calculate sensitivity for each damper setting
                sensitivities = []

                for click_col in click_cols:
                    # Get clean data for this pair
                    clean = df[[click_col, target_col]].dropna()

                    if len(clean) < 3:
                        continue

                    x = clean[click_col].values
                    y = clean[target_col].values

                    # Skip if no variation in x
                    if np.std(x) < 1e-10:
                        continue

                    # Calculate correlation
                    corr = np.corrcoef(x, y)[0, 1]

                    if np.isnan(corr):
                        continue

                    # Calculate regression slope (normalized by std for comparability)
                    slope = corr * (np.std(y) / np.std(x))

                    # Find optimal click value
                    # Note: Higher clicks = opening damper (softer), Lower clicks = closing (stiffer)
                    if lower_is_better:
                        # Want to minimize metric
                        optimal_idx = np.argmin(y)
                        # Positive corr means more clicks = higher metric, so close (decrease) to improve
                        direction = "close" if corr > 0 else "open"
                        # Positive bar = increasing clicks helps (negative corr with lower-is-better)
                        effect = -corr
                    else:
                        # Want to maximize metric
                        optimal_idx = np.argmax(y)
                        direction = "open" if corr > 0 else "close"
                        effect = corr

                    optimal_click = x[optimal_idx]
                    optimal_metric = y[optimal_idx]

                    # Parse click column name for cleaner display
                    # Format: "Damper {Type} {Speed} Clicks {Corner}" -> "{Corner} {Speed} {Type}"
                    display_name = click_col.replace("Damper ", "").replace(" Clicks", "")
                    # Extract components
                    corner = ""
                    speed = ""
                    dtype = ""
                    col_lower = click_col.lower()
                    # Corner
                    for c in ["LF", "RF", "LR", "RR"]:
                        if c.lower() in col_lower:
                            corner = c
                            break
                    # Speed
                    if "low spd" in col_lower or "low speed" in col_lower:
                        speed = "Low Spd"
                    elif "high spd" in col_lower or "high speed" in col_lower:
                        speed = "High Spd"
                    # Type
                    if "compression" in col_lower:
                        dtype = "Comp"
                    elif "rebound" in col_lower:
                        dtype = "Reb"
                    # Build display name: Corner Speed Type
                    if corner and speed and dtype:
                        display_name = f"{corner} {speed} {dtype}"

                    sensitivities.append({
                        'column': click_col,
                        'display_name': display_name,
                        'correlation': corr,
                        'effect': effect,
                        'abs_effect': abs(corr),
                        'slope': slope,
                        'direction': direction,
                        'optimal_click': int(round(optimal_click)),
                        'optimal_metric': optimal_metric,
                        'n_points': len(clean),
                        'click_range': f"{int(x.min())}-{int(x.max())}",
                    })

                if not sensitivities:
                    st.warning("⚠️ Could not calculate sensitivities - check data quality")
                else:
                    # Sort by absolute effect
                    sens_df = pd.DataFrame(sensitivities)
                    sens_df = sens_df.sort_values('abs_effect', ascending=True)

                    # Create tornado plot
                    fig = go.Figure()

                    # Color based on direction (green = good direction, red = bad)
                    colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in sens_df['effect']]

                    fig.add_trace(go.Bar(
                        y=sens_df['display_name'],
                        x=sens_df['effect'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"r={c:.3f}" for c in sens_df['correlation']],
                        textposition='outside',
                        hovertemplate=(
                            '<b>%{y}</b><br>'
                            'Correlation: %{customdata[0]:.3f}<br>'
                            'Direction: %{customdata[1]}<br>'
                            'Optimal Click: %{customdata[2]}<br>'
                            'Click Range: %{customdata[3]}<br>'
                            'N: %{customdata[4]}<extra></extra>'
                        ),
                        customdata=list(zip(
                            sens_df['correlation'],
                            sens_df['direction'],
                            sens_df['optimal_click'],
                            sens_df['click_range'],
                            sens_df['n_points']
                        )),
                    ))

                    # Add center line
                    fig.add_vline(x=0, line_width=2, line_color="gray")

                    # Update layout
                    direction_label = "Lower is Better" if lower_is_better else "Higher is Better"
                    fig.update_layout(
                        title=dict(
                            text=f"Sensitivity Analysis: {target_col}<br><sup>{direction_label} | Green = Beneficial Direction</sup>",
                            font=dict(size=18),
                        ),
                        xaxis_title="Effect (Correlation with Desired Direction)",
                        yaxis_title="Damper Setting",
                        template="plotly_white",
                        height=max(500, len(sens_df) * 45 + 120),
                        margin=dict(l=120, r=80, t=100, b=60),
                        showlegend=False,
                        font=dict(size=14),
                        yaxis=dict(tickfont=dict(size=13)),
                        xaxis=dict(tickfont=dict(size=12)),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Summary table with optimal values
                    st.markdown("---")
                    st.markdown("### Optimal Damper Settings")
                    st.markdown(f"*Settings that minimize/maximize **{target_col}** based on the data*")

                    # Sort by absolute effect (most impactful first)
                    summary_df = sens_df.sort_values('abs_effect', ascending=False)[
                        ['display_name', 'direction', 'optimal_click', 'click_range', 'correlation', 'n_points']
                    ].copy()
                    summary_df.columns = ['Damper Setting', 'Recommended', 'Optimal Click', 'Range', 'Correlation', 'N']
                    summary_df['Correlation'] = summary_df['Correlation'].apply(lambda x: f"{x:.3f}")

                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True,
                    )

    # ========================= TAB 4: % Delta ========================= #
    with tab4:
        st.markdown("### % Delta Analysis")
        st.markdown("Compare metric changes vs baseline (Sim ID 1) across damper settings for a single corner.")

        # Check for Sim ID column
        sim_id_col = None
        for col in df.columns:
            if col.lower() in ['sim id', 'simid', 'sim_id', 'simulation id']:
                sim_id_col = col
                break

        if sim_id_col is None:
            st.warning("⚠️ No 'Sim ID' column found in dataset. This tab requires a Sim ID column to identify baseline.")
        else:
            # Initialize session state for selected metrics
            if 'corner_selected_metrics' not in st.session_state:
                st.session_state.corner_selected_metrics = []

            # Two columns for Configuration and Add Metrics (narrower config column)
            config_col, metrics_col = st.columns([1, 2])

            with config_col:
                with st.expander("⚙️ Configuration", expanded=True):
                    # Corner selection
                    corner = st.selectbox(
                        "Select Corner",
                        ["LF", "RF", "LR", "RR"],
                        key='corner_analysis_corner'
                    )

                    # Aggregation method
                    agg_method = st.selectbox(
                        "Aggregation Method",
                        ["Median", "Mean"],
                        key='corner_agg_method'
                    )

            with metrics_col:
                with st.expander("📊 Add Metrics", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 2])

                    with col1:
                        add_metric = st.selectbox(
                            "Metric",
                            parsed.metrics,
                            key='corner_add_metric'
                        )

                    with col2:
                        add_segment = st.selectbox(
                            "Segment",
                            parsed.segments,
                            key='corner_add_segment'
                        )

                    with col3:
                        add_type = st.selectbox(
                            "Type",
                            parsed.types,
                            key='corner_add_type'
                        )

                    if st.button("➕ Add", key='corner_add_btn'):
                        new_col = build_col_name(add_metric, add_segment, add_type)
                        if new_col in df.columns and new_col not in st.session_state.corner_selected_metrics:
                            st.session_state.corner_selected_metrics.append(new_col)
                            st.rerun()
                        elif new_col not in df.columns:
                            st.warning(f"Column '{new_col}' not found")

            # Show selected metrics compactly below the expanders
            if st.session_state.corner_selected_metrics:
                st.markdown("**Selected Metrics:**")
                # Display as horizontal buttons in rows of 4
                metrics_per_row = 4
                for row_start in range(0, len(st.session_state.corner_selected_metrics), metrics_per_row):
                    row_metrics = st.session_state.corner_selected_metrics[row_start:row_start + metrics_per_row]
                    cols = st.columns(metrics_per_row)
                    for idx, metric in enumerate(row_metrics):
                        with cols[idx]:
                            if st.button(f"✕ {metric.split('_')[0]}", key=f'corner_remove_{row_start + idx}'):
                                st.session_state.corner_selected_metrics.remove(metric)
                                st.rerun()

                if st.button("🗑️ Clear All", key='corner_clear_all'):
                    st.session_state.corner_selected_metrics = []
                    st.rerun()

            selected_metrics = st.session_state.corner_selected_metrics

            if not selected_metrics:
                st.info("👆 Add at least one metric to analyze")
            else:
                # Find damper click columns for selected corner
                all_click_cols = find_damper_click_cols(df)
                corner_clicks = [c for c in all_click_cols if corner.lower() in c.lower()]

                # Categorize into low speed quadrants only
                quadrants = {
                    'Low Spd Comp': None,
                    'Low Spd Reb': None,
                }

                for col in corner_clicks:
                    col_lower = col.lower()
                    if ('low spd' in col_lower or 'low speed' in col_lower):
                        if 'compression' in col_lower:
                            quadrants['Low Spd Comp'] = col
                        elif 'rebound' in col_lower:
                            quadrants['Low Spd Reb'] = col

                # Check if we have the quadrants
                missing = [k for k, v in quadrants.items() if v is None]
                if missing:
                    st.warning(f"⚠️ Missing damper columns for {corner}: {', '.join(missing)}")

                # Get baseline values (Sim ID == 1)
                baseline_df = df[df[sim_id_col] == 1]
                if baseline_df.empty:
                    st.warning("⚠️ No baseline data found (Sim ID = 1)")
                else:
                    # Calculate baseline values for each metric and baseline click values
                    baseline_values = {}
                    baseline_clicks = {}
                    for metric in selected_metrics:
                        if metric in baseline_df.columns:
                            val = baseline_df[metric].median() if agg_method == "Median" else baseline_df[metric].mean()
                            baseline_values[metric] = val

                    # Get baseline click values for each quadrant
                    for quad_name, click_col in quadrants.items():
                        if click_col and click_col in baseline_df.columns:
                            baseline_clicks[quad_name] = baseline_df[click_col].iloc[0] if len(baseline_df) > 0 else None

                    # Color palette for metrics
                    metric_colors = px.colors.qualitative.Plotly[:len(selected_metrics)]

                    quadrant_positions = [
                        ('Low Spd Comp', 1, 1),
                        ('Low Spd Reb', 1, 2),
                    ]

                    # First pass: collect all data to determine global y-axis range
                    all_pct_diffs = []
                    plot_data = []  # Store data for second pass

                    for quad_name, row, col in quadrant_positions:
                        click_col = quadrants.get(quad_name)
                        if click_col is None:
                            continue

                        baseline_click = baseline_clicks.get(quad_name)

                        for metric_idx, metric in enumerate(selected_metrics):
                            if metric not in baseline_values or baseline_values[metric] == 0:
                                continue

                            baseline_val = baseline_values[metric]

                            clean = df[[click_col, metric, sim_id_col]].dropna()
                            if clean.empty:
                                continue

                            # Remove outliers using IQR method
                            Q1 = clean[metric].quantile(0.25)
                            Q3 = clean[metric].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            clean = clean[(clean[metric] >= lower_bound) & (clean[metric] <= upper_bound)]

                            if clean.empty:
                                continue

                            if agg_method == "Median":
                                grouped = clean.groupby(click_col)[metric].median().reset_index()
                            else:
                                grouped = clean.groupby(click_col)[metric].mean().reset_index()

                            if len(grouped) < 2:
                                all_data = df[[click_col, metric]].dropna()
                                Q1 = all_data[metric].quantile(0.25)
                                Q3 = all_data[metric].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                all_data = all_data[(all_data[metric] >= lower_bound) & (all_data[metric] <= upper_bound)]
                                if agg_method == "Median":
                                    grouped = all_data.groupby(click_col)[metric].median().reset_index()
                                else:
                                    grouped = all_data.groupby(click_col)[metric].mean().reset_index()

                            if len(grouped) < 1:
                                continue

                            grouped['pct_diff'] = ((grouped[metric] - baseline_val) / abs(baseline_val)) * 100
                            grouped = grouped.sort_values(click_col)

                            # Collect all pct_diff values for global range
                            all_pct_diffs.extend(grouped['pct_diff'].tolist())

                            # Store for plotting
                            plot_data.append({
                                'quad_name': quad_name,
                                'row': row,
                                'col': col,
                                'click_col': click_col,
                                'metric': metric,
                                'metric_idx': metric_idx,
                                'grouped': grouped,
                                'baseline_click': baseline_click,
                            })

                    # Calculate global y-axis range with small padding
                    if all_pct_diffs:
                        y_min = min(all_pct_diffs)
                        y_max = max(all_pct_diffs)
                        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                        y_range = [y_min - y_padding, y_max + y_padding]
                    else:
                        y_range = [-1, 1]

                    # Create 1x2 subplot for low speed only
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=[
                            f"{corner} Low Spd Compression",
                            f"{corner} Low Spd Rebound",
                        ],
                        horizontal_spacing=0.08,
                    )

                    # Second pass: add traces
                    for data in plot_data:
                        row = data['row']
                        col = data['col']
                        grouped = data['grouped']
                        metric = data['metric']
                        metric_idx = data['metric_idx']
                        click_col = data['click_col']

                        show_legend = (row == 1 and col == 1)
                        fig.add_trace(
                            go.Scatter(
                                x=grouped[click_col],
                                y=grouped['pct_diff'],
                                mode='lines+markers',
                                name=metric.split('_')[0] if show_legend else None,
                                legendgroup=metric,
                                showlegend=show_legend,
                                line=dict(color=metric_colors[metric_idx], width=2),
                                marker=dict(size=6, color=metric_colors[metric_idx]),
                                hovertemplate=(
                                    f'<b>{metric.split("_")[0]}</b><br>'
                                    'Click: %{x}<br>'
                                    '% Diff: %{y:.2f}%<br>'
                                    '<extra></extra>'
                                ),
                            ),
                            row=row, col=col
                        )

                    # Add baseline vertical lines and update axes
                    for quad_name, row, col in quadrant_positions:
                        baseline_click = baseline_clicks.get(quad_name)
                        if baseline_click is not None:
                            fig.add_vline(
                                x=baseline_click,
                                line_dash="dash",
                                line_color="black",
                                line_width=1.5,
                                row=row, col=col
                            )

                        fig.update_xaxes(row=row, col=col)
                        fig.update_yaxes(title_text="% Diff", range=y_range, row=row, col=col)

                    # Add horizontal line at 0 for reference
                    for c in [1, 2]:
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=c)

                    fig.update_layout(
                        title=dict(
                            text=f"{corner} Low Speed % Delta vs Baseline ({agg_method}) - Dashed Line = Baseline",
                            font=dict(size=16),
                        ),
                        template="plotly_white",
                        height=500,
                        margin=dict(l=60, r=60, t=100, b=60),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=12),
                        ),
                        font=dict(size=13),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show baseline values
                    with st.expander("📊 Baseline Values (Sim ID = 1)"):
                        baseline_data = []
                        for metric, val in baseline_values.items():
                            baseline_data.append({
                                'Metric': metric,
                                'Baseline Value': f"{val:.4f}"
                            })
                        # Add baseline click values
                        st.markdown("**Baseline Click Settings:**")
                        click_data = []
                        for quad_name, click_val in baseline_clicks.items():
                            if click_val is not None:
                                click_data.append({
                                    'Setting': quad_name,
                                    'Baseline Clicks': int(click_val)
                                })
                        if click_data:
                            st.dataframe(pd.DataFrame(click_data), use_container_width=True, hide_index=True)
                        st.markdown("**Baseline Metric Values:**")
                        st.dataframe(pd.DataFrame(baseline_data), use_container_width=True, hide_index=True)

    # ========================= TAB 5: Balance ========================= #
    with tab5:
        st.markdown("### Balance Analysis")
        st.markdown("Compare one metric across 3 segments for each corner (LF, RF, LR, RR) vs baseline.")

        # Check for Sim ID column
        sim_id_col = None
        for col in df.columns:
            if col.lower() in ['sim id', 'simid', 'sim_id', 'simulation id']:
                sim_id_col = col
                break

        if sim_id_col is None:
            st.warning("⚠️ No 'Sim ID' column found in dataset. This tab requires a Sim ID column to identify baseline.")
        else:
            # Configuration
            with st.expander("⚙️ Configuration", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    # Metric selection
                    balance_metric = st.selectbox(
                        "Metric",
                        parsed.metrics,
                        key='balance_metric'
                    )

                    # Type selection
                    balance_type = st.selectbox(
                        "Type",
                        parsed.types,
                        key='balance_type'
                    )

                    # Damper type for x-axis (Compression, Rebound, or Both)
                    balance_damper_type = st.selectbox(
                        "X-Axis Damper Type",
                        ["Compression", "Rebound", "Both (2x4 Grid)"],
                        key='balance_damper_type',
                        help="'Both' shows 2x4 grid: each corner has LSC and LSR side-by-side"
                    )

                with col2:
                    # 3 segment selections with colors - auto-select preferred segments if available
                    st.markdown("**Segments (Red, Blue, Green):**")

                    # Find preferred segment indices
                    def get_segment_index(preferred, segments, default):
                        for i, seg in enumerate(segments):
                            if seg.lower() == preferred.lower():
                                return i
                        return default

                    seg1_idx = get_segment_index("Ent-1", parsed.segments, 0)
                    seg2_idx = get_segment_index("Apx-1-2", parsed.segments, min(1, len(parsed.segments)-1))
                    seg3_idx = get_segment_index("Ext-2", parsed.segments, min(2, len(parsed.segments)-1))

                    seg1 = st.selectbox("Segment 1 (Red)", parsed.segments, key='balance_seg1', index=seg1_idx)
                    seg2 = st.selectbox("Segment 2 (Blue)", parsed.segments, key='balance_seg2', index=seg2_idx)
                    seg3 = st.selectbox("Segment 3 (Green)", parsed.segments, key='balance_seg3', index=seg3_idx)

                # Aggregation method
                balance_agg = st.selectbox(
                    "Aggregation Method",
                    ["Median", "Mean"],
                    key='balance_agg'
                )

            # Build column names for each segment
            col1_name = build_col_name(balance_metric, seg1, balance_type)
            col2_name = build_col_name(balance_metric, seg2, balance_type)
            col3_name = build_col_name(balance_metric, seg3, balance_type)

            # Check if columns exist
            missing_cols = []
            for col_name in [col1_name, col2_name, col3_name]:
                if col_name not in df.columns:
                    missing_cols.append(col_name)

            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            else:
                # Get baseline values (Sim ID == 1)
                baseline_df = df[df[sim_id_col] == 1]
                if baseline_df.empty:
                    st.warning("⚠️ No baseline data found (Sim ID = 1)")
                else:
                    # Calculate baseline values for each segment
                    baseline_values = {}
                    for col_name, seg_name in [(col1_name, seg1), (col2_name, seg2), (col3_name, seg3)]:
                        if balance_agg == "Median":
                            baseline_values[col_name] = baseline_df[col_name].median()
                        else:
                            baseline_values[col_name] = baseline_df[col_name].mean()

                    # Get damper click columns for each corner
                    all_click_cols = find_damper_click_cols(df)
                    corners = ["LF", "RF", "LR", "RR"]

                    # Check if "Both" mode is selected
                    both_mode = "Both" in balance_damper_type

                    # Pre-calculate click columns for each corner (both comp and reb)
                    corner_click_cols = {}
                    for corner in corners:
                        corner_clicks = [c for c in all_click_cols if corner.lower() in c.lower()]
                        comp_col = None
                        reb_col = None
                        for c in corner_clicks:
                            if 'low spd' in c.lower() or 'low speed' in c.lower():
                                if 'compression' in c.lower():
                                    comp_col = c
                                elif 'rebound' in c.lower():
                                    reb_col = c
                        corner_click_cols[corner] = {'comp': comp_col, 'reb': reb_col}

                    if both_mode:
                        # 2x4 grid: Row 1 = Front (LF-LSC, LF-LSR, RF-LSC, RF-LSR)
                        #           Row 2 = Rear  (LR-LSC, LR-LSR, RR-LSC, RR-LSR)
                        subplot_titles = [
                            "LF - LSC", "LF - LSR", "RF - LSC", "RF - LSR",
                            "LR - LSC", "LR - LSR", "RR - LSC", "RR - LSR",
                        ]
                        fig = make_subplots(
                            rows=2, cols=4,
                            subplot_titles=subplot_titles,
                            vertical_spacing=0.12,
                            horizontal_spacing=0.05,
                            column_widths=[0.25, 0.25, 0.25, 0.25],
                        )
                        # Plot positions: (corner, damper_type, row, col)
                        plot_positions = [
                            ('LF', 'comp', 1, 1), ('LF', 'reb', 1, 2), ('RF', 'comp', 1, 3), ('RF', 'reb', 1, 4),
                            ('LR', 'comp', 2, 1), ('LR', 'reb', 2, 2), ('RR', 'comp', 2, 3), ('RR', 'reb', 2, 4),
                        ]
                    else:
                        # 2x2 grid: standard car layout
                        damper_type_lower = balance_damper_type.lower()
                        damper_key = 'comp' if 'compression' in damper_type_lower else 'reb'

                        def get_click_title(corner):
                            click_col = corner_click_cols[corner][damper_key]
                            if click_col is None:
                                return f"{corner} Corner"
                            parts = click_col.replace("Damper ", "").replace(f" {corner}", "").replace(" Clicks", "")
                            return f"{corner} - {parts}"

                        subplot_titles = [get_click_title(corner) for corner in corners]
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=subplot_titles,
                            vertical_spacing=0.15,
                            horizontal_spacing=0.10,
                        )
                        plot_positions = [
                            ('LF', damper_key, 1, 1), ('RF', damper_key, 1, 2),
                            ('LR', damper_key, 2, 1), ('RR', damper_key, 2, 2),
                        ]

                    # Colors for segments
                    segment_colors = {
                        col1_name: '#e74c3c',  # Red
                        col2_name: '#3498db',  # Blue
                        col3_name: '#2ecc71',  # Green
                    }
                    segment_labels = {
                        col1_name: seg1,
                        col2_name: seg2,
                        col3_name: seg3,
                    }

                    # FIRST PASS: Collect all data and calculate global y-range
                    all_pct_diffs = []
                    plot_data = []

                    for corner, damper_key, row, col in plot_positions:
                        click_col = corner_click_cols[corner][damper_key]
                        if click_col is None:
                            continue

                        baseline_click = baseline_df[click_col].iloc[0] if click_col in baseline_df.columns and len(baseline_df) > 0 else None

                        for metric_col, color in segment_colors.items():
                                baseline_val = baseline_values[metric_col]
                                if baseline_val == 0:
                                    continue

                                clean = df[[click_col, metric_col, sim_id_col]].dropna()
                                if clean.empty:
                                    continue

                                if balance_agg == "Median":
                                    grouped = clean.groupby(click_col)[metric_col].median().reset_index()
                                else:
                                    grouped = clean.groupby(click_col)[metric_col].mean().reset_index()

                                if len(grouped) < 1:
                                    continue

                                grouped['pct_diff'] = ((grouped[metric_col] - baseline_val) / abs(baseline_val)) * 100
                                grouped = grouped.sort_values(click_col)

                                all_pct_diffs.extend(grouped['pct_diff'].tolist())

                                plot_data.append({
                                    'corner': corner,
                                    'row': row,
                                    'col': col,
                                    'click_col': click_col,
                                    'metric_col': metric_col,
                                    'color': color,
                                    'grouped': grouped,
                                    'baseline_click': baseline_click,
                                })

                    # Calculate global y-axis range
                    if all_pct_diffs:
                        y_min = min(all_pct_diffs)
                        y_max = max(all_pct_diffs)
                        y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                        y_range = [y_min - y_padding, y_max + y_padding]
                    else:
                        y_range = [-1, 1]

                    # SECOND PASS: Plot the data
                    for data in plot_data:
                        row = data['row']
                        col = data['col']
                        click_col = data['click_col']
                        metric_col = data['metric_col']
                        color = data['color']
                        grouped = data['grouped']

                        x_data = grouped[click_col].values
                        y_data = grouped['pct_diff'].values

                        seg_name = segment_labels[metric_col]
                        legend_name = seg_name
                        legend_group = metric_col

                        show_legend = (row == 1 and col == 1)

                        if len(x_data) >= 4:
                            try:
                                x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
                                spline = make_interp_spline(x_data, y_data, k=3)
                                y_smooth = spline(x_smooth)

                                fig.add_trace(
                                    go.Scatter(
                                        x=x_smooth, y=y_smooth, mode='lines',
                                        name=legend_name if show_legend else None,
                                        legendgroup=legend_group, showlegend=show_legend,
                                        line=dict(color=color, width=2),
                                        hovertemplate=f'<b>{seg_name}</b><br>Click: %{{x:.0f}}<br>% Diff: %{{y:.2f}}%<extra></extra>',
                                    ),
                                    row=row, col=col
                                )
                            except Exception:
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_data, y=y_data, mode='lines',
                                        name=legend_name if show_legend else None,
                                        legendgroup=legend_group, showlegend=show_legend,
                                        line=dict(color=color, width=2),
                                        hovertemplate=f'<b>{seg_name}</b><br>Click: %{{x:.0f}}<br>% Diff: %{{y:.2f}}%<extra></extra>',
                                    ),
                                    row=row, col=col
                                )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=x_data, y=y_data, mode='lines',
                                    name=legend_name if show_legend else None,
                                    legendgroup=legend_group, showlegend=show_legend,
                                    line=dict(color=color, width=2),
                                    hovertemplate=f'<b>{seg_name}</b><br>Click: %{{x:.0f}}<br>% Diff: %{{y:.2f}}%<extra></extra>',
                                ),
                                row=row, col=col
                            )

                    # Add baseline vertical lines and update axes
                    for corner, damper_key, row, col in plot_positions:
                        click_col = corner_click_cols[corner][damper_key]

                        if click_col and click_col in baseline_df.columns and len(baseline_df) > 0:
                            baseline_click = baseline_df[click_col].iloc[0]
                            if baseline_click is not None:
                                fig.add_vline(x=baseline_click, line_dash="dash", line_color="black", line_width=1.5, row=row, col=col)

                        # Auto-scaled y-axis based on all data
                        fig.update_yaxes(title_text="% Diff", range=y_range, row=row, col=col)

                    # Add horizontal line at 0
                    num_cols = 4 if both_mode else 2
                    for r in [1, 2]:
                        for c in range(1, num_cols + 1):
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=r, col=c)

                    # Adjust title and height based on mode (33% taller)
                    if both_mode:
                        title_text = f"Balance: {balance_metric} ({balance_type}) - LSC & LSR - Dashed Line = Baseline"
                        chart_height = 865
                    else:
                        title_text = f"Balance: {balance_metric} ({balance_type}) - Dashed Line = Baseline"
                        chart_height = 1000

                    fig.update_layout(
                        title=dict(
                            text=title_text,
                            font=dict(size=16),
                        ),
                        template="plotly_white",
                        height=chart_height,
                        margin=dict(l=60, r=60, t=120, b=60),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=12),
                        ),
                        font=dict(size=12),
                    )

                    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
