"""
Reusable classifier pipeline for HW1 (and future research).

Goals:
- Self-contained: works even before the student fills in utils.py / attacks.py.
- Calls the student's attacks.py for attack execution (so the UI grows with their
  implementation), but never silently substitutes a reference solution.
- Stable, type-friendly API the Streamlit app talks to.
"""

from __future__ import annotations

import functools
import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

HW_DIR = Path(__file__).resolve().parent.parent
if str(HW_DIR) not in sys.path:
    sys.path.insert(0, str(HW_DIR))

CLASSES = ["airplane", "car", "ship", "truck"]
CLASS_EMOJIS = {"airplane": "\u2708\ufe0f", "car": "\U0001f697",
                "ship": "\U0001f6a2", "truck": "\U0001f69b"}
N_CLASSES = len(CLASSES)


# ----- environment / device --------------------------------------------------

def get_device(prefer: str = "auto") -> torch.device:
    """Pick a torch device. prefer in {auto, cuda, mps, cpu}."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----- lazy imports of student modules ---------------------------------------
# Lazy + reload-friendly so the UI keeps working as the student edits files.

def _reload(name: str) -> types.ModuleType:
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def get_models_module():
    return _reload("models")


def get_utils_module():
    return _reload("utils")


def get_attacks_module():
    return _reload("attacks")


# ----- model + dataset loading -----------------------------------------------

@functools.lru_cache(maxsize=8)
def load_model(cnn_id: int, device_str: str) -> torch.nn.Module:
    """Load and cache a pretrained CNN by id (0/1/2)."""
    utils_mod = get_utils_module()
    model = utils_mod.load_pretrained_cnn(
        cnn_id, models_dir=str(HW_DIR / "trained-models")
    )
    model.to(torch.device(device_str))
    model.eval()
    return model


def clear_model_cache() -> None:
    load_model.cache_clear()


@functools.lru_cache(maxsize=1)
def load_raw_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) where xs is (N, 32, 32, 3) float32 in [0,1] and ys is (N,)."""
    import gzip
    with gzip.open(str(HW_DIR / "dataset.npz"), "rb") as fin:
        data = np.load(fin, allow_pickle=True)
    xs = np.stack([np.asarray(data[i][0], dtype=np.float32) for i in range(len(data))])
    ys = np.array([int(data[i][1]) for i in range(len(data))], dtype=np.int64)
    return xs, ys


def dataset_size() -> int:
    return int(load_raw_dataset()[0].shape[0])


def get_image(idx: int) -> tuple[np.ndarray, int]:
    """Return (HxWxC float32 in [0,1], label)."""
    xs, ys = load_raw_dataset()
    return xs[idx].copy(), int(ys[idx])


def indices_for_class(class_id: int) -> list[int]:
    _, ys = load_raw_dataset()
    return np.where(ys == class_id)[0].tolist()


def to_tensor(image_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W, C) [0,1] float to (1, C, H, W) tensor on device."""
    x = torch.from_numpy(image_hwc).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device).float()


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) or (1, C, H, W) tensor to (H, W, C) numpy in [0,1]."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def stack_dataset_tensors(device: torch.device,
                          indices: Optional[list[int]] = None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = load_raw_dataset()
    if indices is None:
        indices = list(range(len(xs)))
    sub_x = torch.from_numpy(xs[indices]).permute(0, 3, 1, 2).contiguous().float().to(device)
    sub_y = torch.from_numpy(ys[indices]).long().to(device)
    return sub_x, sub_y


# ----- inference + metrics ---------------------------------------------------

@dataclass
class Prediction:
    logits: np.ndarray            # (n_classes,)
    probs: np.ndarray             # (n_classes,)
    pred_class: int
    pred_label: str
    confidence: float

    @classmethod
    def from_logits(cls, logits_t: torch.Tensor) -> "Prediction":
        if logits_t.dim() == 2:
            logits_t = logits_t[0]
        probs = F.softmax(logits_t, dim=-1).detach().cpu().numpy()
        pred = int(np.argmax(probs))
        return cls(
            logits=logits_t.detach().cpu().numpy(),
            probs=probs,
            pred_class=pred,
            pred_label=CLASSES[pred],
            confidence=float(probs[pred]),
        )


def predict_one(model: torch.nn.Module, image_hwc: np.ndarray,
                device: torch.device) -> Prediction:
    x = to_tensor(image_hwc, device)
    with torch.no_grad():
        logits = model(x)
    return Prediction.from_logits(logits)


def predict_batch(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Return predicted classes (np array)."""
    with torch.no_grad():
        return model(x).argmax(dim=-1).cpu().numpy()


def benign_accuracy(model: torch.nn.Module, device: torch.device,
                    batch_size: int = 32) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute (acc, per_class_acc, confusion_matrix) on the full dataset."""
    x, y = stack_dataset_tensors(device)
    correct = 0
    n = x.shape[0]
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size]
            yb = y[i:i + batch_size]
            preds = model(xb).argmax(dim=-1)
            correct += int((preds == yb).sum().item())
            for t, p in zip(yb.cpu().numpy(), preds.cpu().numpy()):
                cm[int(t), int(p)] += 1
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return correct / n, per_class.astype(np.float32), cm


# ----- attack wrappers -------------------------------------------------------

class AttackNotImplemented(RuntimeError):
    """Raised when the student's attack returns None / hasn't been filled in."""


def _check_adv(x_adv: Any, name: str) -> torch.Tensor:
    if x_adv is None:
        raise AttackNotImplemented(
            f"`{name}.execute(...)` returned `None`. "
            f"Implement it in attacks.py to use this feature."
        )
    return x_adv


@dataclass
class AttackOutput:
    x: torch.Tensor                  # original tensor (B, C, H, W)
    x_adv: torch.Tensor              # adversarial tensor
    y_true: torch.Tensor             # original labels
    y_used: torch.Tensor             # labels used by the attack (true or target)
    targeted: bool
    n_queries: Optional[torch.Tensor] = None  # only for blackbox
    attack_name: str = ""
    config: dict = field(default_factory=dict)

    @property
    def delta(self) -> torch.Tensor:
        return (self.x_adv - self.x).detach()

    @property
    def linf(self) -> float:
        return float(self.delta.abs().max().item())

    @property
    def l2_per_sample(self) -> float:
        return float(self.delta.flatten(1).norm(dim=1).mean().item())

    @property
    def mean_abs(self) -> float:
        return float(self.delta.abs().mean().item())

    def attack_success(self, model: torch.nn.Module) -> float:
        with torch.no_grad():
            preds_adv = model(self.x_adv).argmax(dim=-1)
        if self.targeted:
            success = (preds_adv == self.y_used).float().mean().item()
        else:
            success = (preds_adv != self.y_true).float().mean().item()
        return float(success)


def _resolve_targets(y_true: torch.Tensor, targeted: bool,
                     target_class: Optional[int]) -> torch.Tensor:
    """Compute label tensor passed to attack.execute()."""
    if not targeted:
        return y_true
    if target_class is None:
        # default rule from utils: t = (c + randint(1, n_classes)) % n_classes
        offsets = torch.randint(1, N_CLASSES, y_true.shape, device=y_true.device)
        return (y_true + offsets) % N_CLASSES
    return torch.full_like(y_true, int(target_class))


def run_pgd(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
            *, eps: float, alpha: float, n_iter: int,
            rand_init: bool, early_stop: bool,
            targeted: bool, target_class: Optional[int] = None
            ) -> AttackOutput:
    attacks_mod = get_attacks_module()
    attack = attacks_mod.PGDAttack(
        model=model, eps=eps, n=n_iter, alpha=alpha,
        rand_init=rand_init, early_stop=early_stop,
    )
    y_used = _resolve_targets(y, targeted, target_class)
    x_adv = _check_adv(attack.execute(x, y_used, targeted=targeted), "PGDAttack")
    return AttackOutput(
        x=x.detach(), x_adv=x_adv.detach(),
        y_true=y, y_used=y_used, targeted=targeted,
        attack_name="PGD",
        config=dict(eps=eps, alpha=alpha, n_iter=n_iter,
                    rand_init=rand_init, early_stop=early_stop,
                    targeted=targeted, target_class=target_class),
    )


def run_nes(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
            *, eps: float, alpha: float, n_iter: int, k: int, sigma: float,
            momentum: float, rand_init: bool, early_stop: bool,
            targeted: bool, target_class: Optional[int] = None
            ) -> AttackOutput:
    attacks_mod = get_attacks_module()
    attack = attacks_mod.NESBBoxPGDAttack(
        model=model, eps=eps, n=n_iter, alpha=alpha,
        momentum=momentum, k=k, sigma=sigma,
        rand_init=rand_init, early_stop=early_stop,
    )
    y_used = _resolve_targets(y, targeted, target_class)
    result = attack.execute(x, y_used, targeted=targeted)
    if result is None or not isinstance(result, tuple) or len(result) != 2:
        raise AttackNotImplemented(
            "`NESBBoxPGDAttack.execute(...)` should return (x_adv, n_queries). "
            "Implement it in attacks.py."
        )
    x_adv, n_queries = result
    return AttackOutput(
        x=x.detach(), x_adv=x_adv.detach(),
        y_true=y, y_used=y_used, targeted=targeted,
        n_queries=n_queries.detach() if torch.is_tensor(n_queries) else torch.as_tensor(n_queries),
        attack_name="NES (black-box)",
        config=dict(eps=eps, alpha=alpha, n_iter=n_iter, k=k, sigma=sigma,
                    momentum=momentum, rand_init=rand_init, early_stop=early_stop,
                    targeted=targeted, target_class=target_class),
    )


def run_ensemble(models: list[torch.nn.Module], x: torch.Tensor, y: torch.Tensor,
                 *, eps: float, alpha: float, n_iter: int,
                 rand_init: bool, early_stop: bool,
                 targeted: bool, target_class: Optional[int] = None
                 ) -> AttackOutput:
    attacks_mod = get_attacks_module()
    attack = attacks_mod.PGDEnsembleAttack(
        models=models, eps=eps, n=n_iter, alpha=alpha,
        rand_init=rand_init, early_stop=early_stop,
    )
    y_used = _resolve_targets(y, targeted, target_class)
    x_adv = _check_adv(attack.execute(x, y_used, targeted=targeted), "PGDEnsembleAttack")
    return AttackOutput(
        x=x.detach(), x_adv=x_adv.detach(),
        y_true=y, y_used=y_used, targeted=targeted,
        attack_name="PGD ensemble",
        config=dict(eps=eps, alpha=alpha, n_iter=n_iter,
                    rand_init=rand_init, early_stop=early_stop,
                    targeted=targeted, target_class=target_class,
                    n_source_models=len(models)),
    )


# ----- bit-flip attack helpers (research utilities, not HW solutions) --------

def float_to_bits(w: float) -> str:
    """Return 32-bit big-endian IEEE-754 representation of a float32."""
    import struct
    [packed] = struct.unpack(">I", struct.pack(">f", float(w)))
    return format(packed, "032b")


def bits_to_float(bits: str) -> float:
    import struct
    packed = int(bits, 2).to_bytes(4, "big")
    return struct.unpack(">f", packed)[0]


def flip_bit(w: float, bit_idx: int) -> float:
    """Flip a specific bit (0 = MSB / sign) of a float32 weight."""
    bits = float_to_bits(w)
    flipped = list(bits)
    flipped[bit_idx] = "1" if bits[bit_idx] == "0" else "0"
    return bits_to_float("".join(flipped))


@dataclass
class BitFlipResult:
    layer_name: str
    weight_index: tuple
    bit_idx: int
    original_value: float
    flipped_value: float
    acc_before: float
    acc_after: float

    @property
    def rad(self) -> float:
        if self.acc_before <= 0:
            return float("nan")
        return (self.acc_before - self.acc_after) / self.acc_before


def random_bit_flip_experiment(model: torch.nn.Module, layer_name: str,
                               device: torch.device,
                               acc_before: Optional[float] = None,
                               rng: Optional[np.random.Generator] = None
                               ) -> BitFlipResult:
    """Flip one random bit in a random weight of `layer_name`, measure RAD,
    and restore the weight."""
    if rng is None:
        rng = np.random.default_rng()
    layer = dict(model.named_modules())[layer_name]
    if not hasattr(layer, "weight") or layer.weight is None:
        raise ValueError(f"Layer {layer_name!r} has no weight tensor.")

    if acc_before is None:
        acc_before, _, _ = benign_accuracy(model, device)

    with torch.no_grad():
        W = layer.weight
        flat_idx = int(rng.integers(0, W.numel()))
        bit_idx = int(rng.integers(0, 32))
        idx = np.unravel_index(flat_idx, W.shape)
        original_value = float(W.view(-1)[flat_idx].item())
        flipped_value = flip_bit(original_value, bit_idx)
        W.view(-1)[flat_idx] = float(flipped_value)
        acc_after, _, _ = benign_accuracy(model, device)
        W.view(-1)[flat_idx] = float(original_value)

    return BitFlipResult(
        layer_name=layer_name,
        weight_index=tuple(int(i) for i in idx),
        bit_idx=bit_idx,
        original_value=original_value,
        flipped_value=float(flipped_value),
        acc_before=acc_before,
        acc_after=acc_after,
    )


def list_flippable_layers(model: torch.nn.Module) -> list[str]:
    return [name for name, m in model.named_modules()
            if hasattr(m, "weight") and m.weight is not None
            and isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
