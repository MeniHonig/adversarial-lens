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

Three equivalent ways to launch (run from the repo root):

```bash
python ui/run.py                       # one-click; auto-installs missing deps
bash   ui/run.sh                       # equivalent shell launcher
python -m streamlit run ui/app.py      # raw streamlit invocation
```

The UI opens at <http://localhost:8501>.

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
ui/
├── app.py                    # Streamlit app (six tabs)
├── pipeline.py               # Reusable classifier pipeline + attack wrappers
├── viz.py                    # Plotly + image-rendering helpers
├── embeddings.py             # NEW — feature extraction + cached per-attack bundles
├── embedding_reducers.py     # NEW — pluggable 2D reducers (PCA / t-SNE / UMAP / ...)
├── embedding_viz.py          # NEW — scatter, small multiples, drift heatmap
├── requirements.txt          # streamlit, plotly, pillow
├── run.py                    # Python launcher (press ▶ in your IDE)
├── run.sh                    # Equivalent shell launcher
└── README.md
```
