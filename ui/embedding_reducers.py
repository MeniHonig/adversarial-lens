"""
Pluggable dimensionality reducers for the embedding viewer.

Adding a new reducer:

    @register("trimap", needs="trimap")
    class TrimapReducer(BaseReducer):
        def fit_transform(self, X, *, n_components=2):
            import trimap
            return trimap.TRIMAP(n_dims=n_components).fit_transform(X)

It will appear automatically in the UI dropdown.

Every reducer:
- Operates on a (N, D) float matrix.
- Returns a (N, n_components) float matrix.
- Reads a hyper-parameter dict (see `default_params` for each class).
- Receives an optional `seed` so runs are reproducible.

We keep an LRU cache keyed by (reducer_name, hyperparams, X-hash). For the
canonical 200-image dataset this means re-selecting an already-tried reducer is
instantaneous.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# --------- registry ---------------------------------------------------------

_REGISTRY: dict[str, "BaseReducer"] = {}


def register(name: str, *, needs: Optional[str] = None,
             label: Optional[str] = None) -> Callable:
    """Class decorator that registers a reducer under `name`."""
    def wrap(cls):
        cls.name = name
        cls.label = label or name.upper()
        cls.optional_dep = needs
        _REGISTRY[name] = cls
        return cls
    return wrap


def available_reducers() -> list[str]:
    """Return the names of all reducers whose optional dependency is importable."""
    out = []
    for name, cls in _REGISTRY.items():
        if cls.optional_dep is None:
            out.append(name)
            continue
        try:
            __import__(cls.optional_dep)
            out.append(name)
        except ImportError:
            continue
    return out


def all_reducers() -> dict[str, type]:
    """All registered reducers, even those whose optional deps are missing."""
    return dict(_REGISTRY)


def get_reducer(name: str, **params) -> "BaseReducer":
    if name not in _REGISTRY:
        raise KeyError(f"Unknown reducer {name!r}. "
                       f"Registered: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls(**params)


# --------- base + cache -----------------------------------------------------

@dataclass
class BaseReducer:
    """Tiny base class. Subclasses override `fit_transform`."""
    n_components: int = 2
    seed: int = 0
    extra: dict = field(default_factory=dict)

    name: str = ""              # filled by @register
    label: str = ""
    optional_dep: Optional[str] = None

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Hyper-parameters (other than n_components / seed) the UI should expose."""
        return {}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _hash_inputs(name: str, X: np.ndarray, params: dict) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(name.encode())
    h.update(X.shape.__repr__().encode())
    h.update(X.dtype.str.encode())
    # only hash a digest of X, not all the bytes — keeps things fast and is
    # plenty for the 200-image regime.
    h.update(np.ascontiguousarray(X).tobytes()[: 64 * 1024])
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()


_RESULT_CACHE: dict[str, np.ndarray] = {}


def cached_fit_transform(name: str, X: np.ndarray,
                         *, n_components: int = 2, seed: int = 0,
                         **params) -> np.ndarray:
    """Memoised entry point used by the UI."""
    key = _hash_inputs(name, X, dict(params, n_components=n_components, seed=seed))
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]
    reducer = get_reducer(name, n_components=n_components, seed=seed, extra=params)
    out = reducer.fit_transform(X)
    _RESULT_CACHE[key] = out.astype(np.float32, copy=False)
    return _RESULT_CACHE[key]


def clear_cache() -> None:
    _RESULT_CACHE.clear()


# --------- built-in reducers ------------------------------------------------

@register("pca", label="PCA (linear)")
class PCAReducer(BaseReducer):
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"whiten": False}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components,
                  whiten=bool(self.extra.get("whiten", False)),
                  random_state=self.seed)
        return pca.fit_transform(X)


@register("tsne", label="t-SNE")
class TSNEReducer(BaseReducer):
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"perplexity": 30.0, "learning_rate": "auto", "metric": "euclidean"}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        perp = float(self.extra.get("perplexity", 30.0))
        perp = min(perp, max(5.0, (X.shape[0] - 1) / 3.0))
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=perp,
            learning_rate=self.extra.get("learning_rate", "auto"),
            metric=self.extra.get("metric", "euclidean"),
            init="pca", random_state=self.seed,
        )
        return tsne.fit_transform(X)


@register("umap", label="UMAP", needs="umap")
class UMAPReducer(BaseReducer):
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import umap
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=int(self.extra.get("n_neighbors", 15)),
            min_dist=float(self.extra.get("min_dist", 0.1)),
            metric=self.extra.get("metric", "euclidean"),
            random_state=self.seed,
        )
        return reducer.fit_transform(X)


@register("isomap", label="Isomap")
class IsomapReducer(BaseReducer):
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"n_neighbors": 8}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import Isomap
        n_neighbors = int(self.extra.get("n_neighbors", 8))
        n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
        iso = Isomap(n_neighbors=n_neighbors, n_components=self.n_components)
        return iso.fit_transform(X)


@register("mds", label="MDS")
class MDSReducer(BaseReducer):
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"metric_mds": True}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import MDS
        mds = MDS(
            n_components=self.n_components,
            metric=bool(self.extra.get("metric_mds", True)),
            random_state=self.seed,
            normalized_stress="auto",
        )
        return mds.fit_transform(X)


@register("kernel_pca", label="Kernel PCA (RBF)")
class KernelPCAReducer(BaseReducer):
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"gamma": 0.0}

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import KernelPCA
        gamma = float(self.extra.get("gamma", 0.0)) or None
        kp = KernelPCA(n_components=self.n_components, kernel="rbf",
                       gamma=gamma, random_state=self.seed)
        return kp.fit_transform(X)


@register("random_proj", label="Random Gaussian Projection")
class RandomProjectionReducer(BaseReducer):
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.random_projection import GaussianRandomProjection
        rp = GaussianRandomProjection(n_components=self.n_components,
                                      random_state=self.seed)
        return rp.fit_transform(X)
