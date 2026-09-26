"""Architecture-agnostic port of the default loop in looped_qwen3_moe.py, for other MoE models.

Enabled per model with hf_overrides={"gloop": {"start": a, "end": b, ...}} (no architecture
override: the stock vLLM class is used and its window layers are patched after __init__, so
the loop is in place before CUDA graph capture). Keys:

    start, end      window layers [start, end)                         (required)
    K               evaluations of each layer                          6
    beta            anchor weight on the natural pass                  0.5
    step            damped-Euler step                                  1/K
    freeze          replay the natural pass's router output            True
    norm_interp     rescale iterates to the interpolated norm          True

K=1 is the baseline: the same full-state path with the loop off.

Per window layer, with x the full residual stream (hidden + residual):
    natural = L(x)                                  # routing captured at the gate
    h = rescale((1-s) x + s natural, N_1)           # N_k = |x| + (|natural| - |x|) k/K
    for k = 1..K-1:
        h = (1-s) h + s L(h; frozen routing)
        if k < K-1: h = rescale(h, N_{k+1})
    L(x) once more                                  # rewrites the natural K/V the
                                                    # iterations overwrote in the cache
    out = beta natural + (1-beta) h

Requirements on the model: decoder layers called as layer(positions, hidden, residual) or
layer(hidden, positions, residual) returning (hidden, residual) with a fused add-norm
(vLLM's Llama-style layers), and an MoE block at layer.mlp exposing its router as .gate or
.router (dense layers are looped without freezing). Checked: ERNIE-4.5-MoE, GLM-4.5, gpt-oss,
DeepSeek-V2/V3 (incl. Kimi-VL), Qwen3-MoE, dense Qwen3.
"""
import functools
import importlib
import inspect

import torch

DEFAULT_CFG = dict(K=6, beta=0.5, step=None, freeze=True, norm_interp=True)

# (module, class): model classes whose __init__ gets the post-construction hook
TARGETS = [
    ("vllm.model_executor.models.ernie45_moe", "Ernie4_5_MoeForCausalLM"),
    ("vllm.model_executor.models.glm4_moe", "Glm4MoeForCausalLM"),
    ("vllm.model_executor.models.gpt_oss", "GptOssForCausalLM"),
    ("vllm.model_executor.models.deepseek_v2", "DeepseekV2ForCausalLM"),
    ("vllm.model_executor.models.kimi_vl", "KimiVLForConditionalGeneration"),
    ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeForCausalLM"),
    ("vllm.model_executor.models.qwen3", "Qwen3ForCausalLM"),     # dense: no router to freeze
]


def _rescale(a, n):
    cur = a.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (a.float() * (n / cur)).to(a.dtype)


def _router(layer):
    blk = getattr(layer, "mlp", None)
    if blk is None or not hasattr(blk, "experts"):
        return None                        # dense layer: nothing to freeze
    for name in ("gate", "router"):
        m = getattr(blk, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def _install_layer(layer, cfg):
    if getattr(layer, "_gloop", False):
        return
    layer._gloop = True
    orig = layer.forward
    sig = inspect.signature(orig)
    slot = {"mode": None, "value": None}

    router = _router(layer)
    if router is not None and cfg["freeze"]:
        r_orig = router.forward

        def r_forward(*args, **kwargs):
            if slot["mode"] == "frozen":
                return slot["value"]
            out = r_orig(*args, **kwargs)
            if slot["mode"] == "capture":
                slot["value"] = out
            return out
        router.forward = r_forward

    K, beta = int(cfg["K"]), float(cfg["beta"])
    s = float(cfg["step"]) if cfg.get("step") is not None else 1.0 / K

    def forward(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = dict(bound.arguments)
        hidden, residual = a["hidden_states"], a["residual"]
        x = hidden if residual is None else hidden + residual

        def run(state):
            # residual=None makes the layer take `state` as the whole stream; its fused
            # add-norm then writes into that tensor, so hand it a copy
            a2 = dict(a, hidden_states=state.clone(), residual=None)
            h, r = orig(**a2)
            return h if r is None else h + r

        slot["mode"] = "capture"
        natural = run(x)
        if K > 1:
            slot["mode"] = "frozen"
            n_in = x.float().norm(dim=-1, keepdim=True)
            n_tgt = natural.float().norm(dim=-1, keepdim=True)
            h = (1.0 - s) * x + s * natural
            if cfg["norm_interp"]:
                h = _rescale(h, n_in + (n_tgt - n_in) * (1 / K))
            for k in range(1, K):
                h = (1.0 - s) * h + s * run(h)
                if k < K - 1 and cfg["norm_interp"]:
                    h = _rescale(h, n_in + (n_tgt - n_in) * ((k + 1) / K))
            run(x)                         # natural K/V back into this layer's cache slots
            out = beta * natural + (1.0 - beta) * h
        else:
            out = natural
        slot["mode"], slot["value"] = None, None
        return out, None

    layer.forward = forward
    return router is not None


def _decoder_layers(model):
    for path in ("model.layers", "language_model.model.layers", "language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError(f"gloop: no decoder layers found on {type(model).__name__}")


def install(model, user_cfg):
    cfg = {**DEFAULT_CFG, **dict(user_cfg)}
    layers = _decoder_layers(model)
    start, end = int(cfg["start"]), int(cfg["end"])
    assert 0 < start < end < len(layers), f"gloop window [{start}, {end}) vs {len(layers)} layers"
    frozen = [_install_layer(layers[i], cfg) for i in range(start, end)]
    print(f"GLOOP {type(model).__name__}: layers [{start}, {end}) of {len(layers)}, "
          f"K={cfg['K']} beta={cfg['beta']} routers frozen {sum(map(bool, frozen))}/{end - start}",
          flush=True)


def _wrap(cls):
    if getattr(cls, "_gloop_wrapped", False):
        return
    orig_init = cls.__init__

    @functools.wraps(orig_init)            # vLLM reads this signature to pick the constructor call
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        vcfg = kwargs.get("vllm_config", args[0] if args else None)
        user = getattr(getattr(getattr(vcfg, "model_config", None), "hf_config", None), "gloop", None)
        if user:
            install(self, user)

    cls.__init__ = __init__
    cls._gloop_wrapped = True


def register():
    for mod, name in TARGETS:
        try:
            _wrap(getattr(importlib.import_module(mod), name))
        except Exception as e:                  # an architecture missing from this vLLM
            print(f"gloop: skip {name}: {type(e).__name__}: {e}", flush=True)
