"""
Regenerate the static images used by the GitHub Pages site.

Run from repo root:
    python scripts/render_docs_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ui"))

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402

import pipeline as P                    # noqa: E402
import embeddings as E                  # noqa: E402
import embedding_reducers as R          # noqa: E402

OUT = REPO / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = P.CLASSES
PALETTE = ["#4F8DFD", "#F9A825", "#26A69A", "#EF5350"]


def render_dataset_preview(out: Path = OUT / "dataset_preview.png"):
    xs, ys = P.load_raw_dataset()
    n = min(80, len(xs))
    rng = np.random.default_rng(0)
    sel = []
    for c in range(len(CLASSES)):
        idx = np.where(ys == c)[0]
        sel.append(rng.choice(idx, size=20, replace=False))
    sel = np.concatenate(sel)
    fig, axes = plt.subplots(4, 20, figsize=(20 * 0.55, 4 * 0.55))
    for ax, i in zip(axes.ravel(), sel):
        ax.imshow(np.clip(xs[i], 0, 1))
        ax.axis("off")
    for r in range(4):
        axes[r, 0].set_ylabel(CLASSES[r], rotation=0, ha="right",
                              va="center", fontsize=9)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
    fig.suptitle("dataset.npz — 200 RGB 32×32 images, 4 classes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def render_embedding_grid(out: Path = OUT / "embedding_grid.png"):
    dev = P.get_device("cpu")
    specs = E.default_attack_specs(eps_n=8, n_iter=30)
    keys = ["clean", "pgd_u", "ens_u"]
    bundles = {}
    coords = {}
    for spec in specs:
        if spec.key not in keys:
            continue
        job = E.EmbeddingJob(spec=spec, target_model_id=0, layer_name="fc2")
        f, y, ya, p, im = job.run(dev)
        bundles[spec.key] = dict(spec=spec, features=f, y_true=y, preds=p)
        coords[spec.key] = R.cached_fit_transform("pca", f, seed=0)

    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.0),
                             sharex=False, sharey=False)
    for ax, key in zip(axes, keys):
        b = bundles[key]
        for c in range(len(CLASSES)):
            m = b["y_true"] == c
            ax.scatter(coords[key][m, 0], coords[key][m, 1],
                       s=22, color=PALETTE[c], label=CLASSES[c],
                       edgecolor="white", linewidths=0.4)
        wrong = b["preds"] != b["y_true"]
        ax.scatter(coords[key][wrong, 0], coords[key][wrong, 1],
                   s=42, facecolor="none", edgecolor="black", linewidths=1.0,
                   label="misclassified")
        ax.set_title(b["spec"].label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.3)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("SimpleCNN-0 · fc2 features · PCA(2)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def render_attack_strip(out: Path = OUT / "attack_strip.png"):
    """Clean / adversarial / amplified perturbation for one image."""
    dev = P.get_device("cpu")
    import torch
    xs, ys = P.load_raw_dataset()
    idx = 3
    x = P.to_tensor(xs[idx], dev)
    y = torch.tensor([int(ys[idx])], device=dev)
    out_pgd = P.run_pgd(P.load_model(0, str(dev)), x, y,
                        eps=8 / 255., alpha=1 / 255., n_iter=50,
                        rand_init=True, early_stop=True, targeted=False)
    clean = xs[idx]
    adv = P.to_numpy_image(out_pgd.x_adv)
    delta = adv - clean
    perturb = np.clip(delta * 16 + 0.5, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
    titles = [f"clean — {CLASSES[int(ys[idx])]}",
              f"adversarial (ε=8/255)", "perturbation ×16"]
    for ax, img, t in zip(axes, [clean, adv, perturb], titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_dataset_preview()
    render_attack_strip()
    render_embedding_grid()
