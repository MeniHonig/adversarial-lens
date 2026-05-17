# adversarial-lens

An interactive playground for **studying how adversarial attacks reshape a CNN's
representation space.** Built on top of a small CIFAR-style 4-class dataset
(airplane / car / ship / truck, 200 images) and three pretrained `SimpleCNN`
models.

| | |
|--|--|
| **Models** | `SimpleCNN-0`, `SimpleCNN-1`, `SimpleCNN-2` (different depth/width) |
| **Attacks** | PGD (white-box), NES (black-box, with momentum), PGD ensemble |
| **Probes** | Benign accuracy, per-class accuracy, confusion matrix, transferability matrix, random bit-flip RAD sweep |
| **NEW** | Embedding viewer: pick a layer → dump features for all 200 images (clean + every attack) → reduce with PCA / t-SNE / UMAP / Isomap / MDS → side-by-side 2-D scatter |

The full pipeline + visualisations are wrapped in a single Streamlit app.

```bash
pip install -r requirements.txt
python ui/run.py            # opens http://localhost:8501
```

## Repository layout

```
adversarial-lens/
├── attacks.py        # PGDAttack, NESBBoxPGDAttack, PGDEnsembleAttack
├── utils.py          # dataset loader, accuracy, attack runners, bit-flip helpers
├── models.py         # SimpleCNN-{0,1,2}
├── consts.py         # SEED / BATCH_SIZE / BF_PER_LAYER
├── dataset.npz       # 200 RGB 32x32 images, 4 classes
├── trained-models/   # simple-cnn-{0,1,2} checkpoints
├── main_a.py main_b.py main_c.py    # original assignment driver scripts
├── ui/               # Streamlit app + reusable pipelines
│   ├── app.py                # tabs: Inference · Attack · Batch · Transfer · Bit-flip · Embeddings
│   ├── pipeline.py           # classifier + attack runner facade
│   ├── viz.py                # plotly + image helpers
│   ├── embeddings.py         # feature extraction + cache for any layer / attack
│   ├── embedding_reducers.py # pluggable 2D reducers (PCA / t-SNE / UMAP / Isomap / MDS / ...)
│   ├── embedding_viz.py      # interactive scatter, side-by-side comparison
│   ├── run.py run.sh         # launchers
│   └── README.md
├── scripts/
│   └── view_dataset.py       # quick preview grid of the 200-image dataset
├── writeup/          # original PDF writeup + LaTeX source + result figures
├── docs/             # GitHub Pages site
└── cache/            # generated embeddings / reductions (git-ignored)
```

## Why "adversarial-lens"?

The new embedding tab is a **lens**: you choose what to look through (model,
layer, reducer) and **what to look at** (clean images, PGD adversarials, NES,
ensemble PGD, ...). Same images, very different shadows.

## Status

This started as the answer to a *Trustworthy ML* homework
([writeup](writeup/writeup.pdf)) and is being grown into a small research
playground. Adding a new attack or a new reducer is intentionally tiny:

```python
# ui/embedding_reducers.py
@register("trimap")
class TrimapReducer(BaseReducer):
    def fit_transform(self, X):
        import trimap
        return trimap.TRIMAP().fit_transform(X)
```

…and it shows up automatically as a dropdown option in the Embeddings tab.

## Credits

Course assignment by Mahmood Sharif (TAU, *Trustworthy ML* 2025).
Solutions, UI scaffolding and embedding viewer by the repo author.
