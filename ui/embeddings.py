"""
Feature-extraction + adversarial-perturbation pipeline for the Embeddings tab.

Conceptually:

    raw images  ---attack(model, eps, ...)---->  perturbed images
        |                                              |
        +-------- pick model / pick layer -------------+
                                |
                                v
                   forward pass with hook
                                |
                                v
                       (N, D) feature matrix
                                |
                                v
                   reducer (PCA / t-SNE / UMAP / ...)
                                |
                                v
                  (N, 2) coords -> plotted in UI

Designed so we can scale up later (e.g. CIFAR-10 instead of 200 images) — the
hard work is cached on disk under `cache/embeddings/`.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

import pipeline as P

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------- attack catalogue ------------------------------------------------
# Pure config so the UI can show check-boxes and we can add new variants by
# adding one dict entry.

@dataclass
class AttackSpec:
    """How to craft adversarial examples for a given column of the embedding view."""
    key: str                                  # short id, used in cache filename
    label: str                                # display name
    kind: str                                 # 'clean' | 'pgd' | 'nes' | 'ensemble'
    params: dict = field(default_factory=dict)

    def cache_token(self) -> str:
        payload = json.dumps({"k": self.kind, "p": self.params}, sort_keys=True)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def default_attack_specs(eps_n: int = 8, n_iter: int = 50) -> list[AttackSpec]:
    """A reasonable default set, mirroring the homework's three attacks."""
    eps = eps_n / 255.0
    alpha = 1 / 255.0
    return [
        AttackSpec("clean", "Clean", "clean"),
        AttackSpec(
            "pgd_u", "PGD untargeted", "pgd",
            dict(eps=eps, alpha=alpha, n_iter=n_iter, targeted=False,
                 rand_init=True, early_stop=True),
        ),
        AttackSpec(
            "pgd_t", "PGD targeted", "pgd",
            dict(eps=eps, alpha=alpha, n_iter=n_iter, targeted=True,
                 rand_init=True, early_stop=True),
        ),
        AttackSpec(
            "nes_u", "NES untargeted", "nes",
            dict(eps=eps, alpha=alpha, n_iter=n_iter, k=200,
                 sigma=1 / 255., momentum=0.0, targeted=False,
                 rand_init=True, early_stop=True),
        ),
        AttackSpec(
            "ens_u", "Ensemble PGD (1+2)", "ensemble",
            dict(eps=eps, alpha=alpha, n_iter=n_iter, targeted=False,
                 rand_init=True, early_stop=True,
                 source_models=[1, 2]),
        ),
    ]


# --------- adversarial example generation ----------------------------------

def craft_adversarials(spec: AttackSpec, target_model_id: int,
                       device: torch.device,
                       indices: Optional[list[int]] = None,
                       progress_cb: Optional[Callable[[float, str], None]] = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_adv, y_used) tensors on `device`."""
    x, y = P.stack_dataset_tensors(device, indices=indices)
    if spec.kind == "clean":
        return x, y
    p = spec.params
    if spec.kind == "pgd":
        out = P.run_pgd(
            P.load_model(target_model_id, str(device)), x, y,
            eps=p["eps"], alpha=p["alpha"], n_iter=p["n_iter"],
            rand_init=p["rand_init"], early_stop=p["early_stop"],
            targeted=p["targeted"], target_class=p.get("target_class"),
        )
    elif spec.kind == "nes":
        out = P.run_nes(
            P.load_model(target_model_id, str(device)), x, y,
            eps=p["eps"], alpha=p["alpha"], n_iter=p["n_iter"],
            k=p["k"], sigma=p["sigma"], momentum=p["momentum"],
            rand_init=p["rand_init"], early_stop=p["early_stop"],
            targeted=p["targeted"], target_class=p.get("target_class"),
        )
    elif spec.kind == "ensemble":
        models = [P.load_model(i, str(device)) for i in p["source_models"]]
        out = P.run_ensemble(
            models, x, y,
            eps=p["eps"], alpha=p["alpha"], n_iter=p["n_iter"],
            rand_init=p["rand_init"], early_stop=p["early_stop"],
            targeted=p["targeted"], target_class=p.get("target_class"),
        )
    else:
        raise ValueError(f"Unknown attack kind {spec.kind!r}")
    if progress_cb is not None:
        progress_cb(1.0, f"{spec.label} crafted")
    return out.x_adv.detach(), out.y_used.detach()


# --------- feature extraction ---------------------------------------------

def list_feature_layers(model: nn.Module) -> list[str]:
    """All layers we consider sensible to hook for features.

    We deliberately include everything that has a `forward` and is a leaf
    Linear/Conv/Activation, plus `'logits'` as a synonym for the model output.
    """
    out = []
    for name, m in model.named_modules():
        if name == "":
            continue
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d,
                          nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.BatchNorm2d)):
            out.append(name)
    out.append("logits")
    return out


def extract_features(model: nn.Module, x: torch.Tensor, layer_name: str,
                     device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Run x through model, capture activations at `layer_name`, return (N, D)."""
    model.eval()
    feats: list[torch.Tensor] = []

    if layer_name == "logits":
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                feats.append(model(x[i:i + batch_size].to(device)).detach().cpu())
        return torch.cat(feats, dim=0).view(x.shape[0], -1).numpy()

    layer = dict(model.named_modules())[layer_name]
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        captured.append(out.detach())

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                captured.clear()
                model(x[i:i + batch_size].to(device))
                feats.append(captured[0].cpu().reshape(captured[0].shape[0], -1))
    finally:
        handle.remove()
    return torch.cat(feats, dim=0).numpy()


# --------- cached "embedding job" ------------------------------------------

@dataclass
class EmbeddingJob:
    """One row in the embedding table: a single (attack, feature_layer) snapshot."""
    spec: AttackSpec
    target_model_id: int
    layer_name: str

    def cache_path(self) -> Path:
        token = (f"m{self.target_model_id}_layer-{self.layer_name.replace('.', '_')}"
                 f"_{self.spec.key}_{self.spec.cache_token()}.npz")
        return CACHE_DIR / token

    def run(self, device: torch.device,
            progress_cb: Optional[Callable[[float, str], None]] = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (features (N, D), y_true (N,), y_adv (N,), preds (N,), adv_images (N, H, W, 3))."""
        cache = self.cache_path()
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                return (data["features"], data["y_true"], data["y_adv"],
                        data["preds"], data["images"])

        if progress_cb:
            progress_cb(0.05, f"Crafting {self.spec.label}...")
        x_adv, y_used = craft_adversarials(self.spec, self.target_model_id, device,
                                           progress_cb=progress_cb)
        # y_true vs. y_used: for targeted attacks y_used is the requested target,
        # but we want to colour by the *real* class so re-fetch ground truth.
        _, y_true_t = P.stack_dataset_tensors(device)
        if progress_cb:
            progress_cb(0.4, f"Extracting features ({self.layer_name})...")
        model = P.load_model(self.target_model_id, str(device))
        feats = extract_features(model, x_adv, self.layer_name, device)
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=-1).cpu().numpy()
        images = np.transpose(x_adv.detach().cpu().numpy(), (0, 2, 3, 1))
        images = np.clip(images, 0.0, 1.0).astype(np.float32)
        np.savez(
            cache,
            features=feats.astype(np.float32),
            y_true=y_true_t.cpu().numpy().astype(np.int64),
            y_adv=y_used.cpu().numpy().astype(np.int64),
            preds=preds.astype(np.int64),
            images=images,
        )
        if progress_cb:
            progress_cb(1.0, f"{self.spec.label} cached")
        return (feats.astype(np.float32), y_true_t.cpu().numpy().astype(np.int64),
                y_used.cpu().numpy().astype(np.int64), preds.astype(np.int64), images)


def run_all_jobs(specs: Iterable[AttackSpec], target_model_id: int,
                 layer_name: str, device: torch.device,
                 progress_cb: Optional[Callable[[float, str], None]] = None
                 ) -> dict[str, dict]:
    """Run all jobs and return a dict keyed by spec.key with feature/img/pred bundles."""
    specs = list(specs)
    n = max(1, len(specs))
    out: dict[str, dict] = {}
    for i, spec in enumerate(specs):
        def cb(frac, msg, i=i, n=n):
            if progress_cb:
                progress_cb((i + frac) / n, msg)
        job = EmbeddingJob(spec=spec, target_model_id=target_model_id,
                           layer_name=layer_name)
        features, y_true, y_adv, preds, images = job.run(device, progress_cb=cb)
        out[spec.key] = dict(
            spec=spec, features=features, y_true=y_true,
            y_adv=y_adv, preds=preds, images=images,
        )
    return out


# --------- small helpers ---------------------------------------------------

def images_to_data_uris(images: np.ndarray, scale: int = 2) -> list[str]:
    """Convert (N, H, W, 3) float images in [0,1] to PNG data: URIs for plotly hovers."""
    from PIL import Image
    import base64
    uris: list[str] = []
    for arr in images:
        img = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        pil = Image.fromarray(img, mode="RGB")
        if scale > 1:
            pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/png;base64,{b64}")
    return uris


def per_attack_accuracy(bundle: dict) -> tuple[float, float]:
    """Return (accuracy, attack_success_rate). Defined on the bundles from run_all_jobs."""
    y_true = bundle["y_true"]
    preds = bundle["preds"]
    spec: AttackSpec = bundle["spec"]
    acc = float(np.mean(preds == y_true))
    if spec.kind == "clean":
        return acc, 0.0
    if spec.params.get("targeted"):
        success = float(np.mean(preds == bundle["y_adv"]))
    else:
        success = float(np.mean(preds != y_true))
    return acc, success


def list_cached_jobs() -> list[Path]:
    return sorted(CACHE_DIR.glob("*.npz"))


def clear_cache() -> int:
    """Delete every cached embedding bundle and return how many files were removed."""
    files = list_cached_jobs()
    for f in files:
        f.unlink()
    return len(files)
