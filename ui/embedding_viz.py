"""Plotly figures for the Embeddings tab."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import CLASSES
from viz import PLOTLY_TEMPLATE, CLASS_COLORS

# Colour palette per class — mirrors the one used in the rest of the UI.
_PALETTE = CLASS_COLORS


# --------- single scatter --------------------------------------------------

def scatter_2d(coords: np.ndarray, y_true: np.ndarray, preds: np.ndarray,
               *, title: str, indices: Optional[np.ndarray] = None,
               image_uris: Optional[Iterable[str]] = None,
               show_misclassified: bool = True,
               height: int = 420) -> go.Figure:
    """Single scatter plot of 2D embeddings.

    Args:
        coords:    (N, 2) float coordinates from the reducer.
        y_true:    (N,) ground-truth labels.
        preds:     (N,) predictions on the (perhaps perturbed) image.
        title:     subplot title.
        indices:   optional dataset indices used in hover text.
        image_uris: optional list of base64 PNG data URIs for hover preview.
        show_misclassified: if True, points where preds != y_true get a red ring.
    """
    fig = go.Figure()
    n_classes = len(CLASSES)
    indices = np.arange(coords.shape[0]) if indices is None else np.asarray(indices)
    image_uris = list(image_uris) if image_uris is not None else None

    for c in range(n_classes):
        mask = y_true == c
        if not mask.any():
            continue
        sub_idx = np.where(mask)[0]
        hover_html: list[str] = []
        for j in sub_idx:
            parts = [
                f"<b>idx {int(indices[j])}</b>",
                f"true: {CLASSES[int(y_true[j])]}",
                f"pred: {CLASSES[int(preds[j])]}",
                "✅ correct" if preds[j] == y_true[j] else "❌ wrong",
            ]
            if image_uris is not None:
                parts.append(f"<br><img src='{image_uris[j]}' width='80'>")
            hover_html.append("<br>".join(parts))
        misclass = preds[sub_idx] != y_true[sub_idx]
        marker_line_color = np.where(misclass, "#000000", _PALETTE[c])
        marker_line_width = np.where(misclass, 2.0, 0.5)
        fig.add_trace(go.Scatter(
            x=coords[sub_idx, 0], y=coords[sub_idx, 1],
            mode="markers",
            name=CLASSES[c],
            marker=dict(
                size=10, color=_PALETTE[c],
                line=dict(color=marker_line_color.tolist(),
                          width=marker_line_width.tolist()),
                opacity=0.85,
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_html,
        ))

    if show_misclassified:
        wrong = preds != y_true
        n_wrong = int(wrong.sum())
        if n_wrong > 0:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.99, y=0.99,
                text=f"misclassified: {n_wrong}/{coords.shape[0]}",
                showarrow=False, align="right",
                font=dict(size=10, color="#475569"),
                bgcolor="rgba(255,255,255,0.7)", borderpad=2,
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="dim 1", yaxis_title="dim 2",
        margin=dict(l=30, r=10, t=30, b=30), height=height,
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, x=1, xanchor="right",
                    font=dict(size=10)),
    )
    return fig


# --------- side-by-side small multiples ------------------------------------

def small_multiples(coords_per_attack: dict[str, np.ndarray],
                    y_true_per_attack: dict[str, np.ndarray],
                    preds_per_attack: dict[str, np.ndarray],
                    labels: dict[str, str], *, reducer_name: str,
                    height: int = 720) -> go.Figure:
    """Grid of attacks, one scatter each, axes intentionally NOT shared.

    The axes intentionally aren't shared because each reducer produces its own
    coordinate frame per attack (especially t-SNE / UMAP); aligning them would
    be misleading. To compare we look at structure (clusters, drift towards
    other classes), not absolute positions.
    """
    keys = list(coords_per_attack.keys())
    n = len(keys)
    if n == 0:
        return go.Figure()
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[labels.get(k, k) for k in keys],
        horizontal_spacing=0.07, vertical_spacing=0.12,
    )

    for i, key in enumerate(keys):
        row = i // n_cols + 1
        col = i % n_cols + 1
        coords = coords_per_attack[key]
        y_true = y_true_per_attack[key]
        preds = preds_per_attack[key]
        for c in range(len(CLASSES)):
            mask = y_true == c
            if not mask.any():
                continue
            sub = np.where(mask)[0]
            misclass = preds[sub] != y_true[sub]
            marker_line_color = np.where(misclass, "#000", _PALETTE[c])
            marker_line_width = np.where(misclass, 1.5, 0.3)
            fig.add_trace(
                go.Scatter(
                    x=coords[sub, 0], y=coords[sub, 1],
                    mode="markers", name=CLASSES[c],
                    marker=dict(size=8, color=_PALETTE[c],
                                line=dict(color=marker_line_color.tolist(),
                                          width=marker_line_width.tolist()),
                                opacity=0.85),
                    showlegend=(i == 0),
                    hovertemplate=(
                        f"<b>{CLASSES[c]}</b><br>"
                        f"pred: %{{customdata}}<extra></extra>"),
                    customdata=[CLASSES[int(p)] for p in preds[sub]],
                ),
                row=row, col=col,
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"Embeddings ({reducer_name}) — per attack",
                   font=dict(size=14)),
        height=height,
        margin=dict(l=30, r=10, t=60, b=20),
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, x=1, xanchor="right",
                    font=dict(size=10)),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# --------- summary tables --------------------------------------------------

def metrics_table(bundle_per_attack: dict[str, dict]) -> go.Figure:
    """A small plotly table summarising accuracy + attack success for every attack."""
    from embeddings import per_attack_accuracy
    rows = []
    for key, bundle in bundle_per_attack.items():
        acc, success = per_attack_accuracy(bundle)
        rows.append([
            bundle["spec"].label,
            f"{acc * 100:.1f}%",
            f"{success * 100:.1f}%",
            bundle["features"].shape[1],
        ])
    if not rows:
        return go.Figure()
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Attack", "Acc on target", "Attack success",
                            "Feature dim"],
                    font=dict(size=11, color="#0f172a"),
                    fill_color="#f1f5f9", align="left"),
        cells=dict(values=list(zip(*rows)),
                   font=dict(size=11), align="left",
                   fill_color="white"),
    )])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=160 + 24 * len(rows),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def class_drift_bars(clean_bundle: dict, adv_bundle: dict) -> go.Figure:
    """Per-class drift: how the prediction distribution shifts under attack."""
    n_cls = len(CLASSES)
    clean_dist = np.zeros((n_cls, n_cls))
    adv_dist = np.zeros((n_cls, n_cls))
    for c in range(n_cls):
        mask = clean_bundle["y_true"] == c
        if mask.any():
            for p in clean_bundle["preds"][mask]:
                clean_dist[c, int(p)] += 1
            clean_dist[c] /= clean_dist[c].sum()
        mask = adv_bundle["y_true"] == c
        if mask.any():
            for p in adv_bundle["preds"][mask]:
                adv_dist[c, int(p)] += 1
            adv_dist[c] /= adv_dist[c].sum()

    drift = adv_dist - clean_dist
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=drift, x=CLASSES, y=CLASSES, colorscale="RdBu", zmid=0,
        text=[[f"{v:+.2f}" for v in row] for row in drift],
        texttemplate="%{text}", hoverongaps=False,
        hovertemplate="true=%{y}<br>pred=%{x}<br>drift=%{z:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Prediction drift (adv − clean) by true class",
                   font=dict(size=13)),
        xaxis_title="Predicted class", yaxis_title="True class",
        margin=dict(l=40, r=10, t=30, b=30), height=300,
        yaxis=dict(autorange="reversed"),
    )
    return fig
