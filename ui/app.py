"""
HW1 Adversarial ML Playground.

A Streamlit UI for visualising:
- Model inference on the HW1 dataset.
- Adversarial attacks (PGD white-box, NES black-box, PGD ensemble).
- Batch metrics, transferability, and bit-flip vulnerability.

Run with:
    streamlit run ui/app.py
from the `hw1-release/` folder, or use the provided `ui/run.sh` script.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pipeline as P  # noqa: E402
import viz as V  # noqa: E402
import embeddings as E  # noqa: E402
import embedding_reducers as R  # noqa: E402
import embedding_viz as EV  # noqa: E402

st.set_page_config(
    page_title="adversarial-lens",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Compact page chrome so everything fits in one viewport */
    .block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important;
                       max-width: 100% !important; }
    header[data-testid="stHeader"] { height: 0 !important; }
    h1 { font-size: 1.4rem !important; margin: 0 !important; padding: 0 !important; }
    h2 { font-size: 1.05rem !important; margin: 0.4rem 0 0.2rem 0 !important; }
    h3 { font-size: 0.95rem !important; margin: 0.3rem 0 0.2rem 0 !important; }
    h4 { font-size: 0.85rem !important; margin: 0.2rem 0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] { padding: 6px 10px; font-size: 0.85rem; }
    /* Tighter widget spacing */
    div[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.6rem !important; }
    label[data-testid="stWidgetLabel"] { font-size: 0.75rem !important;
                                          margin-bottom: 0 !important; }
    div[data-testid="stExpander"] details > summary { padding: 4px 8px !important;
                                                       font-size: 0.8rem !important; }
    /* Smaller plotly chart container margins */
    .js-plotly-plot { margin: 0 !important; }
    /* Metric cards */
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 6px 10px; margin-bottom: 0;
    }
    .metric-card h4 { margin: 0; color: #475569; font-weight: 500;
                      font-size: 11px; line-height: 1.1; }
    .metric-card .value { font-size: 15px; font-weight: 600; color: #0f172a;
                           line-height: 1.15; }
    /* Pills */
    .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
            font-size: 11px; font-weight: 600; margin-right: 4px; }
    .pill-true { background: #FEF3C7; color: #92400E; }
    .pill-pred { background: #DBEAFE; color: #1E40AF; }
    .pill-correct { background: #DCFCE7; color: #166534; }
    .pill-wrong { background: #FEE2E2; color: #991B1B; }
    .small-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 11px; color: #475569; }
    /* Compact button + image */
    .stButton > button { padding: 4px 12px; font-size: 0.85rem; }
    .stImage > img { border-radius: 6px; }
    /* Status alert boxes shorter */
    div[data-testid="stAlert"] { padding: 6px 12px !important; margin: 4px 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- session state defaults ------------------------------------------

def _init_state():
    ss = st.session_state
    ss.setdefault("device_pref", "auto")
    ss.setdefault("img_idx", 0)
    ss.setdefault("rng_seed", 4321)
    ss.setdefault("attack_result", None)
    ss.setdefault("attack_clean_pred", None)
    ss.setdefault("attack_adv_pred", None)
    ss.setdefault("attack_image_idx", None)
    ss.setdefault("attack_model_id", None)
    ss.setdefault("batch_results", {})
    ss.setdefault("transfer_matrix", None)
    ss.setdefault("transfer_targeted", False)
    ss.setdefault("emb_bundles", None)
    ss.setdefault("emb_coords", None)
    ss.setdefault("emb_meta", None)


_init_state()


# ---------- helpers ----------------------------------------------------------

def _device() -> torch.device:
    return P.get_device(st.session_state.device_pref)


def metric_card(col, label: str, value: str, hint: str | None = None):
    html = f"""
    <div class="metric-card">
      <h4>{label}</h4>
      <div class="value">{value}</div>
      {f'<div class="small-mono">{hint}</div>' if hint else ''}
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)


def model_picker(key: str, default: int = 0,
                 label: str = "Model") -> int:
    return st.selectbox(
        label, options=[0, 1, 2],
        format_func=lambda i: f"SimpleCNN-{i}",
        index=default, key=key,
    )


def class_picker(key: str, default: int = 0, label: str = "Class") -> int:
    return st.selectbox(
        label, options=list(range(P.N_CLASSES)),
        format_func=lambda i: f"{P.CLASS_EMOJIS[P.CLASSES[i]]} {P.CLASSES[i]}",
        index=default, key=key,
    )


def image_chooser(key_prefix: str) -> int:
    """Reusable image-chooser widget. Returns the dataset index."""
    n = P.dataset_size()
    mode = st.radio(
        "Pick image by",
        options=["Random", "Index", "Class"],
        horizontal=True, key=f"{key_prefix}_mode",
    )
    if mode == "Random":
        idx_default = int(st.session_state.setdefault(f"{key_prefix}_idx", 0))
        cols = st.columns([3, 1])
        cols[0].markdown(
            f"<div class='small-mono' style='padding-top:8px'>"
            f"image #{idx_default} \u00b7 "
            f"label = {P.CLASSES[P.get_image(idx_default)[1]]}</div>",
            unsafe_allow_html=True,
        )
        if cols[1].button("\U0001f3b2 Roll", key=f"{key_prefix}_roll",
                          use_container_width=True):
            st.session_state[f"{key_prefix}_idx"] = int(np.random.randint(0, n))
            st.rerun()
    elif mode == "Index":
        idx = st.number_input(
            "Dataset index", min_value=0, max_value=n - 1,
            value=int(st.session_state.get(f"{key_prefix}_idx", 0)),
            key=f"{key_prefix}_idx_input",
        )
        st.session_state[f"{key_prefix}_idx"] = int(idx)
    else:
        cols = st.columns([3, 1])
        c = cols[0].selectbox(
            "Class", options=list(range(P.N_CLASSES)),
            format_func=lambda i: f"{P.CLASS_EMOJIS[P.CLASSES[i]]} {P.CLASSES[i]}",
            key=f"{key_prefix}_class",
        )
        in_class = P.indices_for_class(c)
        if cols[1].button("\U0001f3b2 Roll", key=f"{key_prefix}_roll_class",
                          use_container_width=True):
            st.session_state[f"{key_prefix}_idx"] = int(np.random.choice(in_class))
            st.rerun()
        idx = st.session_state.setdefault(f"{key_prefix}_idx", in_class[0])
        if idx not in in_class:
            idx = in_class[0]
            st.session_state[f"{key_prefix}_idx"] = idx
    return int(st.session_state[f"{key_prefix}_idx"])


def render_image_card(image_hwc: np.ndarray, caption: str, *, scale: int = 8):
    pil = V.hwc_to_pil(image_hwc, scale=scale)
    st.image(pil, caption=caption, use_container_width=False)


def status_pills(true_class: int, pred_class: int) -> str:
    correct = pred_class == true_class
    pred_pill_cls = "pill-correct" if correct else "pill-wrong"
    pred_label = "correct" if correct else "wrong"
    return (
        f"<span class='pill pill-true'>true: {P.CLASSES[true_class]}</span>"
        f"<span class='pill pill-pred'>pred: {P.CLASSES[pred_class]}</span>"
        f"<span class='pill {pred_pill_cls}'>{pred_label}</span>"
    )


# ---------- sidebar ----------------------------------------------------------

with st.sidebar:
    st.markdown("## \U0001f9e0 adversarial-lens")
    st.caption("Attacks, transferability, bit-flip + embedding viewer "
               "for SimpleCNN-{0,1,2}.")

    st.session_state.device_pref = st.selectbox(
        "Device",
        options=["auto", "cuda", "mps", "cpu"],
        format_func=lambda x: x.upper() if x != "auto" else "Auto",
        index=["auto", "cuda", "mps", "cpu"].index(st.session_state.device_pref),
        help="Device used for inference & attacks.",
    )
    dev = _device()
    st.markdown(f"<span class='small-mono'>device: {dev}</span>", unsafe_allow_html=True)

    st.session_state.rng_seed = int(st.number_input(
        "RNG seed", min_value=0, max_value=2**31 - 1,
        value=int(st.session_state.rng_seed),
        help="Used for random image picks, attack random init, etc.",
    ))
    np.random.seed(st.session_state.rng_seed)
    torch.manual_seed(st.session_state.rng_seed)

    if st.button("Clear caches", use_container_width=True,
                 help="Reload models & dataset (use after editing models.py / utils.py)."):
        P.clear_model_cache()
        P.load_raw_dataset.cache_clear()
        st.toast("Caches cleared.", icon="\u267b\ufe0f")

    with st.expander("Status", expanded=False):
        try:
            P.get_attacks_module()
            attack_status = "\u2705 attacks.py imports cleanly"
        except Exception as e:
            attack_status = f"\u26a0\ufe0f attacks.py error: {e}"
        try:
            P.get_utils_module()
            utils_status = "\u2705 utils.py imports cleanly"
        except Exception as e:
            utils_status = f"\u26a0\ufe0f utils.py error: {e}"
        st.write(attack_status)
        st.write(utils_status)
        st.write(f"Dataset: {P.dataset_size()} images")

    with st.expander("Tips", expanded=False):
        st.markdown(
            "- Attack tabs call your `attacks.py` directly. If you see an "
            "_AttackNotImplemented_ message, fill in the corresponding "
            "`execute(...)` method and rerun.\n"
            "- After editing source files, click **Clear caches**.\n"
            "- $\\epsilon = 8/255 \\approx 0.0314$ is the standard L-inf budget."
        )


_title_cols = st.columns([3, 1])
with _title_cols[0]:
    st.markdown(
        "### \U0001f9e0 adversarial-lens "
        "<span class='small-mono'>&nbsp;\u00b7 inference \u00b7 attacks \u00b7 "
        "transferability \u00b7 bit-flip \u00b7 embeddings</span>",
        unsafe_allow_html=True,
    )
with _title_cols[1]:
    st.markdown(
        f"<div style='text-align:right; padding-top:4px;'>"
        f"<span class='small-mono'>device: {_device()} \u00b7 "
        f"dataset: {P.dataset_size()} imgs \u00b7 seed: "
        f"{st.session_state.rng_seed}</span></div>",
        unsafe_allow_html=True,
    )


# ---------- tab definitions --------------------------------------------------

(tab_inference, tab_attack, tab_batch, tab_transfer,
 tab_bitflip, tab_embed) = st.tabs(
    [
        "\U0001f50d Inference",
        "\u2694\ufe0f Adversarial attack",
        "\U0001f4ca Batch metrics",
        "\U0001f500 Transferability",
        "\U0001f9ee Bit-flip",
        "\U0001f30c Embeddings",
    ]
)


# ===== Tab 1: Inference =====================================================

with tab_inference:
    cfg, view = st.columns([1, 2.4], gap="medium")

    with cfg:
        cnn_id = model_picker("inf_model", default=0)
        idx = image_chooser("inf")
        run_inf = st.button("Run inference", type="primary",
                            use_container_width=True, key="inf_run")
        try:
            _, _label = P.get_image(idx)
            st.markdown(
                f"<div class='small-mono'>idx={idx} \u00b7 "
                f"label={P.CLASSES[_label]} {P.CLASS_EMOJIS[P.CLASSES[_label]]} "
                f"\u00b7 shape=32\u00d732\u00d73</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    with view:
        try:
            model = P.load_model(cnn_id, str(_device()))
            image_hwc, label = P.get_image(idx)
            pred = P.predict_one(model, image_hwc, _device())

            cols = st.columns([1, 2])
            with cols[0]:
                render_image_card(image_hwc,
                                  caption=f"#{idx} \u00b7 truth: {P.CLASSES[label]}",
                                  scale=6)
                st.markdown(status_pills(label, pred.pred_class), unsafe_allow_html=True)
                st.markdown(
                    f"<div class='small-mono'>confidence = {pred.confidence:.4f}</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.plotly_chart(
                    V.probability_bars(pred.probs, true_class=label,
                                       pred_class=pred.pred_class),
                    use_container_width=True,
                )
                with st.expander("Raw logits", expanded=False):
                    st.dataframe(
                        {"class": P.CLASSES,
                         "logit": [f"{v:.4f}" for v in pred.logits],
                         "prob":  [f"{v:.4f}" for v in pred.probs]},
                        use_container_width=True, hide_index=True,
                    )
            if run_inf:
                st.toast("Inference complete.", icon="\u2705")
        except Exception as e:
            st.error(f"Failed to run inference: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ===== Tab 2: Adversarial Attack ===========================================

with tab_attack:
    cfg, view = st.columns([1, 2], gap="medium")

    with cfg:
        attack_type = st.selectbox(
            "Attack",
            options=["PGD (white-box)", "NES (black-box)", "PGD ensemble"],
            help="Whitebox uses gradients. Blackbox queries only. "
                 "Ensemble attacks several models simultaneously to improve transfer.",
        )

        if attack_type == "PGD ensemble":
            ensemble_ids = st.multiselect(
                "Source models (crafted on)",
                options=[0, 1, 2], default=[1, 2],
                format_func=lambda i: f"SimpleCNN-{i}",
            )
            target_id = model_picker("atk_target", default=0,
                                     label="Target model (evaluated on)")
        else:
            ensemble_ids = None
            target_id = model_picker("atk_target", default=0, label="Model")

        # Targeted toggle on its own row; target class appears beside it when active.
        tg_cols = st.columns([1.4, 2])
        targeted = tg_cols[0].toggle(
            "Targeted", value=False, key="atk_tg",
            help="Untargeted: any wrong class. Targeted: a specific wrong class.",
        )
        if targeted:
            target_class = tg_cols[1].selectbox(
                "Target class", options=list(range(P.N_CLASSES)),
                format_func=lambda i: f"{P.CLASS_EMOJIS[P.CLASSES[i]]} {P.CLASSES[i]}",
                key="atk_target_class", label_visibility="collapsed",
            )
        else:
            target_class = None
            tg_cols[1].markdown(
                "<div class='small-mono' style='padding-top:8px'>"
                "untargeted</div>",
                unsafe_allow_html=True,
            )

        idx = image_chooser("atk")

        # Hyperparameters in a tight 2-col grid
        h1, h2 = st.columns(2)
        eps_n = h1.slider("ε (×1/255)", 1, 32, 8, help="L-inf budget")
        alpha_n = h2.slider("α (×1/255)", 1, 8, 1, help="step size")
        n_iter = h1.slider("Iterations", 5, 200, 50, step=5)
        c1, c2 = st.columns(2)
        rand_init = c1.checkbox("Random init", value=True)
        early_stop = c2.checkbox("Early stop", value=True)

        if attack_type == "NES (black-box)":
            n1, n2 = st.columns(2)
            k = n1.slider("k", 50, 500, 200, step=50, help="2k queries / iter")
            sigma_n = n2.slider("σ (×1/255)", 1, 8, 1)
            momentum = st.slider("Momentum", 0.0, 0.99, 0.0, step=0.05)

        run_atk = st.button("Run attack", type="primary", use_container_width=True)

        with st.popover("\u2139\ufe0f Cheatsheet", use_container_width=True):
            st.markdown(
                "**ε / 255** controls how much each pixel can move; "
                "8/255 is the standard CIFAR budget and is invisible to the eye.\n\n"
                "**Targeted** attacks aim at a specific wrong class — much harder, "
                "lower success rate.\n\n"
                "**Random init** + **early stop** match the HW spec in `attacks.py`.\n\n"
                "**NES** estimates gradients with 2k forward passes per step. "
                "Higher k = better gradients but more queries.\n\n"
                "**Ensemble** PGD optimises perturbations against several models "
                "at once — adversarial examples transfer better."
            )

    with view:
        try:
            target_model = P.load_model(target_id, str(_device()))
            image_hwc, label = P.get_image(idx)

            if run_atk:
                eps = eps_n / 255.
                alpha = alpha_n / 255.
                x = P.to_tensor(image_hwc, _device())
                y = torch.tensor([label], device=_device(), dtype=torch.long)

                with st.spinner(f"Running {attack_type}..."):
                    t0 = time.time()
                    if attack_type == "PGD (white-box)":
                        result = P.run_pgd(
                            target_model, x, y,
                            eps=eps, alpha=alpha, n_iter=n_iter,
                            rand_init=rand_init, early_stop=early_stop,
                            targeted=targeted, target_class=target_class,
                        )
                    elif attack_type == "NES (black-box)":
                        result = P.run_nes(
                            target_model, x, y,
                            eps=eps, alpha=alpha, n_iter=n_iter,
                            k=k, sigma=sigma_n / 255.,
                            momentum=momentum,
                            rand_init=rand_init, early_stop=early_stop,
                            targeted=targeted, target_class=target_class,
                        )
                    else:
                        if not ensemble_ids:
                            raise ValueError("Pick at least one source model.")
                        ensemble_models = [
                            P.load_model(i, str(_device())) for i in ensemble_ids
                        ]
                        result = P.run_ensemble(
                            ensemble_models, x, y,
                            eps=eps, alpha=alpha, n_iter=n_iter,
                            rand_init=rand_init, early_stop=early_stop,
                            targeted=targeted, target_class=target_class,
                        )
                    elapsed = time.time() - t0

                clean_pred = P.predict_one(target_model, image_hwc, _device())
                adv_pred = P.Prediction.from_logits(target_model(result.x_adv))

                st.session_state.attack_result = result
                st.session_state.attack_clean_pred = clean_pred
                st.session_state.attack_adv_pred = adv_pred
                st.session_state.attack_image_idx = idx
                st.session_state.attack_model_id = target_id
                st.session_state.attack_elapsed = elapsed
                st.toast("Attack finished.", icon="\u2694\ufe0f")

            result = st.session_state.attack_result
            if result is None:
                st.info("Configure an attack on the left and press **Run attack**.")
            else:
                clean_pred = st.session_state.attack_clean_pred
                adv_pred = st.session_state.attack_adv_pred
                adv_image = P.to_numpy_image(result.x_adv)
                clean_image = P.to_numpy_image(result.x)
                delta = P.to_numpy_image(result.delta + 0.5) - 0.5

                ic1, ic2, ic3 = st.columns(3, gap="small")
                with ic1:
                    st.image(V.hwc_to_pil(clean_image, scale=6),
                             caption=f"Clean: {clean_pred.pred_label} "
                                     f"({clean_pred.confidence * 100:.0f}%)")
                with ic2:
                    correct = adv_pred.pred_class == result.y_true.item()
                    badge = "\u2705 correct" if correct else "\u274c fooled"
                    st.image(V.hwc_to_pil(adv_image, scale=6),
                             caption=f"Adv: {adv_pred.pred_label} "
                                     f"({adv_pred.confidence * 100:.0f}%) {badge}")
                with ic3:
                    st.image(V.perturbation_visual(delta, amplify=16, scale=6),
                             caption="Perturbation \u00d716")

                m1, m2, m3, m4 = st.columns(4)
                metric_card(m1, "L-inf budget",
                            f"{result.config['eps'] * 255:.1f}/255",
                            f"{result.config['eps']:.4f}")
                metric_card(m2, "Achieved L-inf",
                            f"{result.linf * 255:.2f}/255",
                            f"{result.linf:.4f}")
                metric_card(m3, "Mean |δ|",
                            f"{result.mean_abs * 255:.2f}/255",
                            f"L2: {result.l2_per_sample:.3f}")
                if result.n_queries is not None:
                    qval = int(result.n_queries.float().mean().item())
                    metric_card(m4, "Queries used", f"{qval}",
                                f"runtime: {st.session_state.get('attack_elapsed', 0):.1f}s")
                else:
                    metric_card(m4, "Runtime",
                                f"{st.session_state.get('attack_elapsed', 0):.2f} s",
                                f"iters: {result.config['n_iter']}")

                succeeded = (
                    (result.targeted and adv_pred.pred_class == result.y_used.item()) or
                    (not result.targeted and adv_pred.pred_class != result.y_true.item())
                )
                if succeeded:
                    st.success(
                        f"\u2705 Fooled \u2014 `{P.CLASSES[result.y_true.item()]}` "
                        f"\u2192 `{adv_pred.pred_label}`",
                        icon=None,
                    )
                else:
                    st.warning(
                        "Attack failed (try increasing \u03b5 or iterations).",
                        icon=None,
                    )

                st.plotly_chart(
                    V.probability_compare(clean_pred.probs, adv_pred.probs,
                                          true_class=result.y_true.item()),
                    use_container_width=True,
                )

                with st.expander("Perturbation diagnostics", expanded=False):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.image(V.diff_heatmap(delta, scale=6),
                                 caption="Per-pixel L-inf magnitude")
                    with d2:
                        ok = "\u2705"
                        bad = "\u274c"
                        in_ball = ok if result.linf <= result.config['eps'] + 1e-6 else bad
                        in_range = ok if (
                            result.x_adv.min() >= -1e-6
                            and result.x_adv.max() <= 1 + 1e-6
                        ) else bad
                        target_str = (
                            P.CLASSES[result.y_used.item()] if result.targeted else "—"
                        )
                        st.markdown(
                            f"**Attack:** {result.attack_name}  \n"
                            f"**Targeted:** {result.targeted}  \n"
                            f"**Target class:** {target_str}  \n"
                            f"**Within ε-ball:** {in_ball}  \n"
                            f"**Pixels in [0,1]:** {in_range}"
                        )

        except P.AttackNotImplemented as e:
            st.warning(f"\U0001f6e0\ufe0f {e}")
            st.caption(
                "The UI calls your `attacks.py` directly. "
                "Once you implement the attack and click **Run attack** again, "
                "results will appear here."
            )
        except Exception as e:
            st.error(f"Attack failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ===== Tab 3: Batch metrics =================================================

with tab_batch:
    cfg, view = st.columns([1, 2.4], gap="medium")

    with cfg:
        cnn_ids = st.multiselect(
            "Models", options=[0, 1, 2], default=[0, 1, 2],
            format_func=lambda i: f"SimpleCNN-{i}",
        )
        run_eval = st.button("Evaluate on full dataset",
                             type="primary", use_container_width=True)
        st.caption("Computes accuracy + per-class accuracy + confusion matrix.")

    with view:
        if run_eval:
            for cnn_id in cnn_ids:
                with st.spinner(f"Evaluating SimpleCNN-{cnn_id}..."):
                    model = P.load_model(cnn_id, str(_device()))
                    acc, per_class, cm = P.benign_accuracy(model, _device())
                    st.session_state.batch_results[cnn_id] = (acc, per_class, cm)
            st.toast("Batch evaluation complete.", icon="\u2705")

        if not st.session_state.batch_results:
            st.info("Pick models and press **Evaluate on full dataset**.")
        else:
            keys = sorted(st.session_state.batch_results.keys())
            sub_tabs = st.tabs([f"CNN-{k}" for k in keys])
            for tab_, cnn_id in zip(sub_tabs, keys):
                with tab_:
                    acc, per_class, cm = st.session_state.batch_results[cnn_id]
                    m1, m2, m3 = st.columns(3)
                    metric_card(m1, "Accuracy", f"{acc * 100:.2f}%",
                                f"{int(acc * P.dataset_size())}/{P.dataset_size()}")
                    metric_card(m2, "Best",
                                f"{P.CLASSES[int(per_class.argmax())]} "
                                f"({per_class.max() * 100:.0f}%)")
                    metric_card(m3, "Worst",
                                f"{P.CLASSES[int(per_class.argmin())]} "
                                f"({per_class.min() * 100:.0f}%)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(V.per_class_accuracy_bars(per_class),
                                        use_container_width=True)
                    with c2:
                        st.plotly_chart(V.confusion_matrix_fig(cm),
                                        use_container_width=True)


# ===== Tab 4: Transferability ==============================================

with tab_transfer:
    cfg, view = st.columns([1, 2.4], gap="medium")

    with cfg:
        targeted = st.toggle("Targeted", value=False, key="tx_targeted")
        h1, h2 = st.columns(2)
        eps_n = h1.slider("ε (×1/255)", 1, 32, 8, key="tx_eps")
        alpha_n = h2.slider("α (×1/255)", 1, 8, 1, key="tx_alpha")
        n_iter = h1.slider("Iterations", 5, 200, 50, step=5, key="tx_iter")
        sample_size = h2.slider(
            "Sample size", 16, P.dataset_size(),
            min(64, P.dataset_size()), step=16,
            help="Subset of the dataset used to keep this responsive.",
        )
        c1, c2 = st.columns(2)
        early_stop = c1.checkbox("Early stop", value=False, key="tx_es")
        rand_init = c2.checkbox("Random init", value=True, key="tx_ri")
        include_ensemble = st.checkbox(
            "Include ensemble attack (CNN 1+2 \u2192 0)", value=True,
            help="Q2.2: craft adversarials on models 1+2 and evaluate against model 0.",
        )
        run_tx = st.button("Run transferability sweep",
                           type="primary", use_container_width=True)
        st.caption("Row = source (crafted on), Column = target (evaluated on).")

    with view:
        if run_tx:
            try:
                rng = np.random.default_rng(st.session_state.rng_seed)
                indices = rng.choice(P.dataset_size(), size=sample_size, replace=False).tolist()
                x, y = P.stack_dataset_tensors(_device(), indices=indices)

                models = [P.load_model(i, str(_device())) for i in range(3)]
                eps = eps_n / 255.
                alpha = alpha_n / 255.

                matrix = np.zeros((3, 3), dtype=np.float32)
                progress = st.progress(0.0, "Crafting attacks...")
                for src in range(3):
                    res = P.run_pgd(
                        models[src], x, y,
                        eps=eps, alpha=alpha, n_iter=n_iter,
                        rand_init=rand_init, early_stop=early_stop,
                        targeted=targeted,
                    )
                    for tgt in range(3):
                        matrix[src, tgt] = res.attack_success(models[tgt])
                    progress.progress((src + 1) / 3, f"Crafting on CNN-{src}")
                progress.empty()
                st.session_state.transfer_matrix = matrix
                st.session_state.transfer_targeted = targeted

                ensemble_row = None
                if include_ensemble:
                    res = P.run_ensemble(
                        [models[1], models[2]], x, y,
                        eps=eps, alpha=alpha, n_iter=n_iter,
                        rand_init=rand_init, early_stop=early_stop,
                        targeted=targeted,
                    )
                    ensemble_row = np.array(
                        [res.attack_success(models[i]) for i in range(3)],
                        dtype=np.float32,
                    )
                st.session_state.transfer_ensemble = ensemble_row
                st.toast("Transferability sweep done.", icon="\U0001f500")
            except P.AttackNotImplemented as e:
                st.warning(f"\U0001f6e0\ufe0f {e}")
            except Exception as e:
                st.error(f"Failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

        matrix = st.session_state.transfer_matrix
        if matrix is None:
            st.info("Configure the sweep on the left and press "
                    "**Run transferability sweep**.")
        else:
            label = "targeted" if st.session_state.transfer_targeted else "untargeted"
            st.plotly_chart(
                V.transferability_heatmap(matrix,
                                          title=f"PGD transferability ({label})"),
                use_container_width=True,
            )
            d = matrix.diagonal().mean()
            off = (matrix.sum() - matrix.diagonal().sum()) / 6.0
            m1, m2 = st.columns(2)
            metric_card(m1, "Mean white-box success", f"{d * 100:.1f}%",
                        "diagonal: attack \u2192 same model")
            metric_card(m2, "Mean cross-model success", f"{off * 100:.1f}%",
                        "off-diagonal: attack on src, eval on different target")
            ensemble_row = st.session_state.get("transfer_ensemble")
            if ensemble_row is not None:
                st.markdown("#### Ensemble vs. single-source attack")
                e1, e2, e3 = st.columns(3)
                metric_card(e1, "Ensemble (1+2) \u2192 CNN-0",
                            f"{ensemble_row[0] * 100:.1f}%",
                            "single CNN-1\u2192CNN-0: "
                            f"{matrix[1, 0] * 100:.1f}%")
                metric_card(e2, "Ensemble (1+2) \u2192 CNN-1",
                            f"{ensemble_row[1] * 100:.1f}%", "white-box")
                metric_card(e3, "Ensemble (1+2) \u2192 CNN-2",
                            f"{ensemble_row[2] * 100:.1f}%", "white-box")


# ===== Tab 5: Bit-flip =======================================================

with tab_bitflip:
    cfg, view = st.columns([1, 2.4], gap="medium")

    with cfg:
        cnn_id = model_picker("bf_model", default=1)
        try:
            model = P.load_model(cnn_id, str(_device()))
            layer_options = P.list_flippable_layers(model)
        except Exception as e:
            st.error(f"Couldn't load SimpleCNN-{cnn_id}: {e}")
            model = None
            layer_options = []
        layers = st.multiselect(
            "Layers", options=layer_options, default=layer_options,
            help="Each selected layer gets `flips per layer` bit flips.",
        )
        flips_per_layer = st.slider("Flips per layer", 5, 500, 50, step=5)
        run_bf = st.button("Run random bit-flip sweep",
                           type="primary", use_container_width=True)
        st.caption("Flip one random bit in one random weight, measure RAD, restore.")

    with view:
        if run_bf and model is None:
            st.error("Cannot run: model failed to load.")
        elif run_bf:
            try:
                rng = np.random.default_rng(st.session_state.rng_seed)
                rad_per_bit: dict[int, list[float]] = {b: [] for b in range(32)}
                all_results: list[P.BitFlipResult] = []
                acc_orig, _, _ = P.benign_accuracy(model, _device())
                progress = st.progress(0.0, "Flipping bits...")
                total = max(1, len(layers) * flips_per_layer)
                done = 0
                for layer_name in layers:
                    for _ in range(flips_per_layer):
                        r = P.random_bit_flip_experiment(
                            model, layer_name, _device(),
                            acc_before=acc_orig, rng=rng,
                        )
                        all_results.append(r)
                        rad_per_bit[r.bit_idx].append(r.rad)
                        done += 1
                        if done % max(1, total // 30) == 0:
                            progress.progress(done / total,
                                              f"{done}/{total} flips")
                progress.empty()

                rads = np.array([r.rad for r in all_results])
                m0, m1, m2, m3 = st.columns(4)
                metric_card(m0, "Original acc", f"{acc_orig * 100:.2f}%")
                metric_card(m1, "Total flips", str(len(rads)))
                metric_card(m2, "Max RAD", f"{rads.max() * 100:.2f}%")
                metric_card(m3, "RAD > 15%",
                            f"{(rads > 0.15).mean() * 100:.2f}%")

                st.plotly_chart(V.rad_vs_bit_idx_box(rad_per_bit),
                                use_container_width=True)

                worst = sorted(all_results, key=lambda r: r.rad, reverse=True)[:10]
                st.markdown("#### Top-10 most damaging flips")
                st.dataframe(
                    [{
                        "layer": r.layer_name,
                        "weight idx": str(r.weight_index),
                        "bit": r.bit_idx,
                        "before": f"{r.original_value:+.3e}",
                        "after": f"{r.flipped_value:+.3e}",
                        "RAD": f"{r.rad * 100:.2f}%",
                    } for r in worst],
                    use_container_width=True, hide_index=True,
                )
            except Exception as e:
                st.error(f"Bit-flip sweep failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
        else:
            st.info("Pick layers and press **Run random bit-flip sweep**.")


# ===== Tab 6: Embeddings =====================================================

with tab_embed:
    st.markdown(
        "<div class='small-mono'>"
        "Pick a model + layer, choose which attacks to craft, run the pipeline, "
        "then reduce to 2D with any registered reducer. "
        "All adversarial bundles are cached in <code>cache/embeddings/</code>."
        "</div>", unsafe_allow_html=True,
    )

    cfg, view = st.columns([1, 2.6], gap="medium")

    with cfg:
        emb_model_id = model_picker("emb_model", default=0,
                                    label="Target model (features extracted from)")
        try:
            _m = P.load_model(emb_model_id, str(_device()))
            layer_options = E.list_feature_layers(_m)
        except Exception as e:
            st.error(f"Couldn't load SimpleCNN-{emb_model_id}: {e}")
            layer_options = ["logits"]
        default_layer = layer_options[-2] if len(layer_options) > 1 else layer_options[0]
        layer_name = st.selectbox(
            "Feature layer",
            options=layer_options,
            index=layer_options.index(default_layer),
            help="Activations are captured with a forward hook on this layer. "
                 "`logits` is the network output.",
        )

        st.markdown("**Attack budget**")
        h1, h2 = st.columns(2)
        emb_eps_n = h1.slider("ε (×1/255)", 1, 32, 8, key="emb_eps")
        emb_n_iter = h2.slider("Iterations", 5, 200, 50, step=5, key="emb_iter")

        # build default specs but let the user toggle each one
        default_specs = E.default_attack_specs(eps_n=emb_eps_n, n_iter=emb_n_iter)
        st.markdown("**Attacks to include**")
        enabled_keys = []
        spec_by_key = {s.key: s for s in default_specs}
        for spec in default_specs:
            checked = st.checkbox(
                spec.label, value=spec.key in ("clean", "pgd_u", "ens_u"),
                key=f"emb_spec_{spec.key}",
                help=(f"kind={spec.kind} | "
                      f"{', '.join(f'{k}={v}' for k, v in spec.params.items()) or '—'}"),
            )
            if checked:
                enabled_keys.append(spec.key)

        st.markdown("**Reducer**")
        avail = R.available_reducers()
        all_known = R.all_reducers()
        missing = [n for n in all_known if n not in avail]
        if missing:
            st.caption(
                "Disabled (missing dep): "
                + ", ".join(f"`{n}` (needs `{all_known[n].optional_dep}`)"
                            for n in missing)
            )
        reducer_name = st.selectbox(
            "Method",
            options=avail,
            index=avail.index("pca") if "pca" in avail else 0,
            format_func=lambda n: all_known[n].label,
        )
        reducer_cls = all_known[reducer_name]
        reducer_params: dict = {}
        for pname, pval in reducer_cls.default_params().items():
            if isinstance(pval, bool):
                reducer_params[pname] = st.checkbox(pname, value=pval,
                                                    key=f"emb_red_{pname}")
            elif isinstance(pval, int):
                reducer_params[pname] = st.number_input(
                    pname, value=int(pval), step=1, key=f"emb_red_{pname}",
                )
            elif isinstance(pval, float):
                reducer_params[pname] = st.number_input(
                    pname, value=float(pval), step=0.1, format="%.3f",
                    key=f"emb_red_{pname}",
                )
            elif isinstance(pval, str):
                reducer_params[pname] = st.text_input(pname, value=pval,
                                                      key=f"emb_red_{pname}")
            else:
                # fall through: render as JSON to keep API simple
                reducer_params[pname] = pval

        layout_mode = st.radio(
            "Layout",
            options=["small_multiples", "single"], horizontal=True,
            format_func=lambda x: {"small_multiples": "Grid (per attack)",
                                   "single": "Single attack"}[x],
            key="emb_layout",
        )
        if layout_mode == "single":
            single_key = st.selectbox(
                "Show attack", options=enabled_keys,
                format_func=lambda k: spec_by_key[k].label,
                key="emb_single",
            ) if enabled_keys else None
        else:
            single_key = None

        with_image_hover = st.checkbox(
            "Image previews on hover", value=True,
            help="Encodes every image as a base64 PNG (slower for larger datasets).",
        )

        run_emb = st.button(
            "Run embedding pipeline", type="primary",
            use_container_width=True, key="emb_run",
            disabled=not enabled_keys,
        )

        cache_files = E.list_cached_jobs()
        with st.expander(f"Cache ({len(cache_files)} files)", expanded=False):
            if cache_files:
                st.code("\n".join(f.name for f in cache_files[-12:]))
            if st.button("Clear embedding cache", key="emb_clear",
                         use_container_width=True):
                removed = E.clear_cache()
                R.clear_cache()
                st.toast(f"Removed {removed} cached bundles", icon="\u267b\ufe0f")

    with view:
        if run_emb:
            try:
                specs = [spec_by_key[k] for k in enabled_keys]
                bundles: dict[str, dict] = {}
                progress = st.progress(0.0, "Starting...")
                total = max(1, len(specs))
                for i, spec in enumerate(specs):
                    def cb(frac, msg, i=i, total=total):
                        progress.progress((i + frac) / total, msg)
                    job = E.EmbeddingJob(spec=spec,
                                         target_model_id=emb_model_id,
                                         layer_name=layer_name)
                    f, y_true, y_adv, preds, images = job.run(_device(),
                                                              progress_cb=cb)
                    bundles[spec.key] = dict(
                        spec=spec, features=f, y_true=y_true,
                        y_adv=y_adv, preds=preds, images=images,
                    )
                progress.progress(1.0, "Reducing...")

                coords_per_attack: dict[str, np.ndarray] = {}
                for key, bundle in bundles.items():
                    coords = R.cached_fit_transform(
                        reducer_name, bundle["features"],
                        n_components=2, seed=st.session_state.rng_seed,
                        **reducer_params,
                    )
                    coords_per_attack[key] = coords
                progress.empty()
                st.session_state.emb_bundles = bundles
                st.session_state.emb_coords = coords_per_attack
                st.session_state.emb_meta = dict(
                    model_id=emb_model_id, layer=layer_name,
                    reducer=reducer_name, reducer_params=reducer_params,
                    with_image_hover=with_image_hover,
                )
                st.toast("Embedding pipeline finished.", icon="\U0001f30c")
            except Exception as e:
                st.error(f"Embedding pipeline failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

        bundles = st.session_state.emb_bundles
        coords_per_attack = st.session_state.emb_coords
        meta = st.session_state.emb_meta

        if bundles is None or coords_per_attack is None:
            st.info(
                "Pick attacks + reducer on the left and press "
                "**Run embedding pipeline**. The clean pass is cheap; "
                "PGD and NES take a few seconds each on CPU."
            )
        else:
            st.plotly_chart(EV.metrics_table(bundles),
                            use_container_width=True)

            label_map = {k: b["spec"].label for k, b in bundles.items()}
            if layout_mode == "small_multiples":
                fig = EV.small_multiples(
                    coords_per_attack,
                    {k: b["y_true"] for k, b in bundles.items()},
                    {k: b["preds"] for k, b in bundles.items()},
                    label_map,
                    reducer_name=R.all_reducers()[meta["reducer"]].label,
                    height=720 if len(bundles) > 3 else 420,
                )
                st.plotly_chart(fig, use_container_width=True)
            elif single_key is not None and single_key in bundles:
                bundle = bundles[single_key]
                coords = coords_per_attack[single_key]
                image_uris = (E.images_to_data_uris(bundle["images"])
                              if meta.get("with_image_hover") else None)
                st.plotly_chart(
                    EV.scatter_2d(
                        coords, bundle["y_true"], bundle["preds"],
                        title=f"{bundle['spec'].label} — "
                              f"{R.all_reducers()[meta['reducer']].label}",
                        image_uris=image_uris,
                        height=520,
                    ),
                    use_container_width=True,
                )

            # Drift heatmap (clean vs. each adversarial attack) — only when both are present.
            if "clean" in bundles:
                others = [k for k in bundles if k != "clean"]
                if others:
                    st.markdown("#### Prediction drift")
                    per_row = 3
                    for row_start in range(0, len(others), per_row):
                        row_keys = others[row_start:row_start + per_row]
                        row_cols = st.columns(len(row_keys))
                        for col, key in zip(row_cols, row_keys):
                            with col:
                                st.plotly_chart(
                                    EV.class_drift_bars(bundles["clean"],
                                                        bundles[key]),
                                    use_container_width=True,
                                )

            with st.expander("Pipeline metadata", expanded=False):
                st.json({
                    "model": f"SimpleCNN-{meta['model_id']}",
                    "layer": meta["layer"],
                    "reducer": meta["reducer"],
                    "reducer_params": meta["reducer_params"],
                    "attacks": list(bundles.keys()),
                    "feature_shape": list(next(iter(bundles.values()))["features"].shape),
                })
