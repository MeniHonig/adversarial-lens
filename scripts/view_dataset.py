"""
View images stored in dataset.npz.

Usage:
    python view_dataset.py                  # save a grid of all 200 images to dataset_preview.jpg
    python view_dataset.py --interactive    # open an interactive matplotlib window instead
    python view_dataset.py --per-class 10   # 4x10 grid: 10 random images per class
"""

import argparse
import gzip

import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["airplane", "car", "ship", "truck"]


def load(fpath="dataset.npz"):
    with gzip.open(fpath, "rb") as fin:
        data = np.load(fin, allow_pickle=True)
    xs = np.stack([data[i][0] for i in range(len(data))])  # (N, 32, 32, 3) float32 in [0,1]
    ys = np.array([int(data[i][1]) for i in range(len(data))])
    return xs, ys


def plot_grid(xs, ys, rows, cols, title=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.atleast_2d(axes)
    for ax, img, lab in zip(axes.ravel(), xs, ys):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(CLASSES[int(lab)], fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(xs):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interactive", action="store_true", help="open a window instead of saving")
    p.add_argument("--per-class", type=int, default=0,
                   help="if >0, show this many random images per class (4 rows)")
    p.add_argument("--out", default="dataset_preview.jpg")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    xs, ys = load()
    print(f"Loaded {len(xs)} samples, shape={xs.shape}, dtype={xs.dtype}, "
          f"min={xs.min():.3f}, max={xs.max():.3f}")

    rng = np.random.default_rng(args.seed)
    if args.per_class > 0:
        k = args.per_class
        sel_x, sel_y = [], []
        for c in range(len(CLASSES)):
            idx = np.where(ys == c)[0]
            pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
            sel_x.append(xs[pick])
            sel_y.append(ys[pick])
        sel_x = np.concatenate(sel_x)
        sel_y = np.concatenate(sel_y)
        fig = plot_grid(sel_x, sel_y, rows=len(CLASSES), cols=k,
                        title=f"{k} random images per class")
    else:
        rows, cols = 10, 20
        fig = plot_grid(xs[: rows * cols], ys[: rows * cols], rows, cols,
                        title="dataset.npz (200 images)")

    if args.interactive:
        plt.show()
    else:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
