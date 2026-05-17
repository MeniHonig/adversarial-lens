"""Plotly + image helpers for the HW1 UI."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from pipeline import CLASSES

PLOTLY_TEMPLATE = "plotly_white"
CLASS_COLORS = ["#4F8DFD", "#F9A825", "#26A69A", "#EF5350"]


# ----- image helpers ---------------------------------------------------------

def hwc_to_pil(arr: np.ndarray, scale: int = 8) -> Image.Image:
    """Take an (H, W, 3) float array in [0, 1] and return an upscaled PIL.Image."""
    arr = np.clip(arr, 0.0, 1.0)
    img = (arr * 255.0).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGB")
    if scale > 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    return pil


def perturbation_visual(delta_hwc: np.ndarray, amplify: float = 16.0,
                        scale: int = 8) -> Image.Image:
    """Visualise an adversarial perturbation by amplifying & shifting to gray=0."""
    vis = np.clip(delta_hwc * amplify + 0.5, 0.0, 1.0)
    return hwc_to_pil(vis, scale=scale)


def diff_heatmap(delta_hwc: np.ndarray, scale: int = 8) -> Image.Image:
    """Per-pixel L_inf magnitude as a viridis-like grayscale heatmap."""
    mag = np.abs(delta_hwc).max(axis=-1)
    mag = mag / max(mag.max(), 1e-9)
    rgb = np.stack([mag, mag * 0.4, 1.0 - mag], axis=-1)
    return hwc_to_pil(rgb, scale=scale)


# ----- plotly charts ---------------------------------------------------------

def probability_bars(probs: np.ndarray, *, true_class: Optional[int] = None,
                     pred_class: Optional[int] = None,
                     title: str = "Class probabilities") -> go.Figure:
    colors = []
    for i in range(len(probs)):
        if i == true_class and i == pred_class:
            colors.append("#2E7D32")
        elif i == pred_class:
            colors.append("#1976D2")
        elif i == true_class:
            colors.append("#FBC02D")
        else:
            colors.append("#90A4AE")
    fig = go.Figure(go.Bar(
        x=CLASSES, y=probs, marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in probs], textposition="outside",
        hovertemplate="<b>%{x}</b><br>p = %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13)),
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        xaxis_title=None, yaxis_title="Probability",
        margin=dict(l=20, r=20, t=30, b=10), height=220,
        font=dict(size=11),
    )
    return fig


def probability_compare(probs_before: np.ndarray, probs_after: np.ndarray,
                        true_class: Optional[int] = None,
                        title: str = "Probabilities: clean vs. adversarial"
                        ) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Clean", x=CLASSES, y=probs_before,
                marker_color="#1976D2",
                text=[f"{p * 100:.1f}%" for p in probs_before], textposition="outside")
    fig.add_bar(name="Adversarial", x=CLASSES, y=probs_after,
                marker_color="#E53935",
                text=[f"{p * 100:.1f}%" for p in probs_after], textposition="outside")
    if true_class is not None:
        fig.add_vrect(
            x0=true_class - 0.5, x1=true_class + 0.5,
            line_width=0, fillcolor="#FBC02D", opacity=0.12,
            annotation_text="true class", annotation_position="top left",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, barmode="group",
        title=dict(text=title, font=dict(size=13)),
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        xaxis_title=None, yaxis_title="Probability",
        margin=dict(l=20, r=20, t=30, b=10), height=240,
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    return fig


def confusion_matrix_fig(cm: np.ndarray, *, title: str = "Confusion matrix") -> go.Figure:
    cm = np.asarray(cm)
    fig = go.Figure(go.Heatmap(
        z=cm, x=CLASSES, y=CLASSES,
        colorscale="Blues", showscale=True,
        text=cm, texttemplate="%{text}", hoverongaps=False,
        hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Predicted", yaxis_title="True",
        margin=dict(l=40, r=10, t=30, b=30), height=260,
        font=dict(size=11),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def transferability_heatmap(matrix: np.ndarray, *, title: str) -> go.Figure:
    labels = [f"CNN-{i}" for i in range(matrix.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale="Reds", zmin=0.0, zmax=1.0,
        text=[[f"{v * 100:.1f}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="src=%{y}<br>target=%{x}<br>success=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Target model", yaxis_title="Source model",
        margin=dict(l=50, r=10, t=30, b=30), height=280,
        font=dict(size=11),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def per_class_accuracy_bars(per_class: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=CLASSES, y=per_class, marker_color=CLASS_COLORS,
        text=[f"{v * 100:.1f}%" for v in per_class], textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Per-class accuracy", font=dict(size=13)),
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        margin=dict(l=20, r=10, t=30, b=10), height=200,
        font=dict(size=11),
    )
    return fig


def queries_box(n_queries_groups: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for name, vals in n_queries_groups.items():
        fig.add_box(y=np.asarray(vals).flatten(), name=name, boxmean=True)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Black-box queries per sample", font=dict(size=13)),
        yaxis_title="# queries",
        margin=dict(l=20, r=10, t=30, b=10), height=240,
        font=dict(size=11),
    )
    return fig


def rad_vs_bit_idx_box(rad_per_bit: dict[int, list[float]]) -> go.Figure:
    fig = go.Figure()
    for bit in sorted(rad_per_bit.keys()):
        fig.add_box(y=rad_per_bit[bit], name=str(bit), boxpoints=False)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="RAD by flipped bit index", font=dict(size=13)),
        xaxis_title="Bit index (0 = sign / MSB)",
        yaxis_title="RAD",
        margin=dict(l=30, r=10, t=30, b=30), height=260,
        font=dict(size=11),
        showlegend=False,
    )
    return fig


def fig_to_png_bytes(fig: go.Figure) -> bytes:
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    return buf.getvalue()
