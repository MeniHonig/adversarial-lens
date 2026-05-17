# adversarial-lens — UI

Interactive Streamlit UI for the adversarial-ML pipeline. Six tabs, each one
a self-contained probe over `SimpleCNN-{0,1,2}`:

- **🔍 Inference** — pick a model, pick a dataset image, see prediction +
  full softmax distribution.
- **⚔️ Adversarial attack** — craft PGD (white-box), NES (black-box), or PGD
  ensemble adversarials. View the clean image, the adversarial image, the
  amplified perturbation, and the before/after probability shift.
- **📊 Batch metrics** — accuracy, per-class accuracy, and a confusion matrix
  for any subset of the three pretrained CNNs.
- **🔄 Transferability** — 3×3 success-rate matrix (source × target model),
  plus the ensemble bonus.
- **🧮 Bit-flip** — random bit-flip experiments; RAD distribution boxplot per
  bit index; ranked list of the most damaging flips.
- **🌌 Embeddings** — (new) feature dump from any layer of any model for the
  clean dataset *and* every attack you tick on. Reduce to 2D with PCA, t-SNE,
  UMAP, Isomap, MDS, Kernel-PCA or random projection. Render as a single
  scatter (with hover image preview) or as a grid of small multiples,
  side-by-side with a prediction-drift heatmap.

## Quick start

Any of these work, from any working directory:

```bash
python run.py                          # top-level shim (recommended)
python ui/run.py                       # the real launcher
bash   ui/run.sh                       # shell wrapper around ui/run.py
python -m streamlit run ui/app.py      # raw streamlit invocation
```

The UI opens at <http://localhost:8501> (auto-bumps to 8502, 8503, ... if
busy).

`ui/run.py` performs a preflight before launch:

- verifies the repo files (`dataset.npz`, `trained-models/simple-cnn-{0,1,2}`,
  every UI module) are present;
- checks every dependency in `requirements.txt` is installed at a high enough
  version, and offers to `pip install` what's missing;
- imports every UI module so syntax errors surface *now*, not after Streamlit
  hides them behind its server log;
- picks a free port if `8501` is taken;
- sets `PYTHONPATH` so `import attacks / utils / models / pipeline` resolves
  consistently across platforms.

Useful flags (`python run.py --help` for the full list):

| flag | what it does |
|---|---|
| `--port N` | preferred port (auto-bump if busy) |
| `--no-browser` | don't open the system browser |
| `--no-install` | fail fast on missing deps instead of pip-installing |
| `--include-optional` | also warn about missing optional deps (e.g. `umap-learn`) |
| `--check-only` | run preflight, print results, exit |
| `-- --server.maxUploadSize 50` | forward args to Streamlit (after `--`) |

Install deps manually:

```bash
pip install -r requirements.txt        # repo-wide deps (incl. UI)
# or only the UI extras:
pip install -r ui/requirements.txt
```

## How it talks to the rest of the repo

The UI imports `attacks.py`, `utils.py`, and `models.py` directly through a
thin facade (`pipeline.py`):

- `pipeline.run_pgd / run_nes / run_ensemble` instantiate
  `PGDAttack`, `NESBBoxPGDAttack`, and `PGDEnsembleAttack` and call
  `.execute(...)`.
- If `execute(...)` returns `None`, the UI shows an `AttackNotImplemented`
  message in place of the result — it never substitutes a reference solution.
- After editing `attacks.py` / `utils.py` / `models.py`, click **Clear
  caches** in the sidebar to force a reload.

The metric helpers in `pipeline.py` are independent of `utils.py`, so the
UI is fully usable for visualisation even before you finish a homework
implementation.

## Adding a new dimensionality reducer

Open `ui/embedding_reducers.py` and decorate a new class:

```python
@register("trimap", needs="trimap", label="TriMap")
class TrimapReducer(BaseReducer):
    @classmethod
    def default_params(cls):
        return {"n_inliers": 10, "n_outliers": 5}

    def fit_transform(self, X):
        import trimap
        return trimap.TRIMAP(
            n_inliers=int(self.extra.get("n_inliers", 10)),
            n_outliers=int(self.extra.get("n_outliers", 5)),
            n_dims=self.n_components,
        ).fit_transform(X)
```

It will appear in the **Embeddings** tab's reducer dropdown automatically.
If `trimap` is not installed it stays greyed-out.

## Adding a new attack

Open `ui/embeddings.py` and extend `default_attack_specs(...)` with a new
`AttackSpec(...)`. A new checkbox appears in the UI on the next reload.

## File layout

```
adversarial-lens/
├── run.py                    # top-level shim that calls ui/run.py
└── ui/
    ├── app.py                    # Streamlit app (six tabs)
    ├── pipeline.py               # Reusable classifier pipeline + attack wrappers
    ├── viz.py                    # Plotly + image-rendering helpers
    ├── embeddings.py             # NEW — feature extraction + cached per-attack bundles
    ├── embedding_reducers.py     # NEW — pluggable 2D reducers (PCA / t-SNE / UMAP / ...)
    ├── embedding_viz.py          # NEW — scatter, small multiples, drift heatmap
    ├── requirements.txt          # streamlit, plotly, pillow
    ├── run.py                    # Robust launcher (preflight, port handling, ...)
    ├── run.sh                    # Shell wrapper around run.py
    └── README.md
```
