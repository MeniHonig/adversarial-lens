# GitHub Pages site

This folder is served by GitHub Pages from the `main` branch's `/docs`
directory. The theme is the GitHub-built `jekyll-theme-cayman`, so no extra
build step is needed — push and it's live.

## Local preview

```bash
gem install bundler jekyll
cd docs
bundle init                       # only once
bundle add jekyll-theme-cayman
bundle exec jekyll serve --source . --destination _site
```

Then open <http://127.0.0.1:4000>.

## Updating the figures

```bash
cd ..    # repo root
python scripts/render_docs_assets.py
```

This regenerates `docs/assets/dataset_preview.png`, `attack_strip.png`, and
`embedding_grid.png` from the live pipeline.
