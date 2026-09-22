"""vLLM port of loop_qwen3.py's `layer_anchored_frozen` for Qwen3-MoE.

Registered as architecture `LoopedQwen3MoeForCausalLM`; the loop is configured via
hf_overrides={"architectures": ["LoopedQwen3MoeForCausalLM"], "loop_cfg": {...}}:

    start, end      window layers [start, end)            (22, 26)
    K               evaluations of each layer              6
    beta            anchor weight on the natural pass      0.5
    step            damped-Euler step                      1/K
    freeze          reuse natural-pass router logits       True
    norm_interp     rescale iterates to interpolated norm  True
    anchor_rescale  rescale the blended output to |natural| False
    res_scale       scale the layer's total update by a    1.0  (out = x + a (out - x))
    mode            "layer" (per-layer loop) | "block" (iterate the window as one map)
    beta_ramp       [b_first, b_last]: anchor weight varies linearly across window layers
    out_mode        "last" (default) | "mean" (blend the anchor with the mean of the iterates)
    beta_head       path to a train_beta.py checkpoint: per-token learned anchor weight
    topk/topk_range MoE experts per token in [start, end) layers (default: the model's own)
    blend           anchor mixing scheme: "linear" | "renorm" | "slerp" | "orth" | "novelty"
    step_blend      same choices for the damped-Euler steps (default "linear")
    recirc_iters    number of recirculation passes (paper's k; default 1)
    iter_mode       "full" (re-attend each iteration) | "mlp" (reuse the natural attention
                    output, iterate only the MoE; no KV writes during iterations)
    windows         optional [[start, end], ...] to loop several windows (overrides start/end)
    kv              "natural" (restore natural-pass K/V) | "last" (keep last iteration's K/V)
    windows=[]      no loop windows at all (e.g. recirculation only)
    recirc_*        recirculation (arXiv:2608.17981): recirc_alpha, recirc_src, recirc_dst,
                    recirc_convex -- see _recirculate. recirc_loop=True also loops the
                    window layers on the recirculation pass (default False keeps the
                    original behaviour, where that pass runs them once)
    pos_min/pos_max optional token-position gate (layer mode): loop only where
                    pos_min <= position < pos_max, natural pass elsewhere
    gate_tau        divergence gate: keep the looped output only for tokens whose
                    ||h_K - natural|| / ||natural|| >= tau, natural pass elsewhere.
                    LOOP_LOG_DIVERGENCE=1 prints that ratio's percentiles to pick tau.
                    Measured on AIME prompts: p10 ~.015-.021, p50 ~.020-.027, p90 ~.026-.035.
    gate_tau_max    the same gate inverted: loop only where that ratio is <= tau_max.
    pre_rescale     "first" | "all": rescale the damped-Euler step's first operand to the
                    second's per-token norm before mixing, so the step weight s acts on
                    directions rather than on magnitude-weighted vectors.
    step_sched      "inv": the k-th iteration uses step 1/(k + step_sched_offset)
                    instead of the constant `step`; with step_blend="slerp" that is a
                    rotation of theta/(k+off) toward the layer's output.
    chain_ens       M: run M damped-Euler chains whose steps are spread over [s/2, 2s]
                    and average their endpoints -- sequential structure plus ensembling.
    ensemble_random "rand" (default; torch.rand per token) | "hash" (derived from |x|,
                    so it also varies per decode step and survives cudagraph replay) |
                    "grid" (one fixed s per member, shared by every token).
    ensemble_jitter with `ensemble`: shift each token's grid of s by a golden-ratio
                    offset of its position, so members differ per token (stratified
                    sampling) rather than every token getting the same fixed s.
    ensemble        N: replace the damped-Euler chain with N parallel evaluations of the
                    layer at s_k = k/(N+1) along [natural, x], averaged. Same compute as
                    K=N; tests whether the gain is variance reduction, not a fixed point.

Per window layer (x = full hidden state, i.e. hidden + residual):
    natural = L(x)                       # writes this layer's KV, yields router logits
    h = rescale((1-s) x + s natural, N_1)
    for k = 1..K-1:
        h = (1-s) h + s L(h; logits)     # attention overwrites the token's KV slot ...
        if k < K-1: h = rescale(h, N_{k+1})
    restore natural K/V into the slots   # ... so the natural entry is written back
    out = beta natural + (1-beta) h      # optionally rescaled to |natural|
Every generated token runs the loop (HF `decode_mode="full"`). Prefill iterations attend
to the chunk's iterated K/V and decode iterations to past natural K/V plus their own,
matching the HF implementation. Written for eager execution (enforce_eager=True).
"""
import os
import types

import torch

DEBUG_NAN = os.environ.get("LOOP_DEBUG_NAN") == "1"
LOG_DIVERGENCE = os.environ.get("LOOP_LOG_DIVERGENCE") == "1"
_div_logged = [0]

from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

DEFAULT_CFG = dict(start=22, end=26, K=6, beta=0.5, step=None, freeze=True,
                   norm_interp=True, anchor_rescale=False, res_scale=1.0, mode="layer",
                   kv="natural", recirc_alpha=0.0, recirc_src=None, recirc_dst=None,
                   recirc_convex=True)


def _blend(anchor, other, w, mode):
    """Mix `other` into `anchor` with weight w, using one of the paper's schemes (Table B.2).

    linear  : (1-w) a + w b                      (plain convex mix; shrinks the norm)
    renorm  : linear, then rescaled to |a|
    slerp   : rotate a toward b by w*theta, magnitude fixed at |a|
    orth    : a + w * (component of b orthogonal to a)   -- keeps a intact, adds novelty
    novelty : a + w * (|a|/|b|) * b / cos(theta)         -- novelty-scaled source
    """
    if mode == "linear" or w == 0.0:
        # In the operands' dtype (bf16), as in the runs behind the README numbers. An fp32
        # upcast here is a mathematical no-op, yet it changed every one of the 480
        # generations of a seed-matched rerun -- keep it out of the default path.
        return ((1.0 - w) * anchor + w * other).to(anchor.dtype)
    a, b = anchor.float(), other.float()
    na = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    nb = b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if mode == "renorm":
        m = (1.0 - w) * a + w * b
        return (m * (na / m.norm(dim=-1, keepdim=True).clamp_min(1e-6))).to(anchor.dtype)
    cos = ((a * b).sum(-1, keepdim=True) / (na * nb)).clamp(-1.0, 1.0)
    if mode == "orth":
        return (a + w * (b - (a * b).sum(-1, keepdim=True) / na.pow(2) * a)).to(anchor.dtype)
    if mode == "novelty":
        return (a + w * (na / nb) * b / cos.clamp_min(0.1)).to(anchor.dtype)
    if mode == "slerp":
        theta = torch.arccos(cos)
        sin = torch.sin(theta)
        out = (torch.sin((1.0 - w) * theta) * a + torch.sin(w * theta) * (na / nb) * b) / sin.clamp_min(1e-4)
        lin = (1.0 - w) * a + w * (na / nb) * b          # theta ~ 0: the two are parallel
        return torch.where(sin > 1e-4, out, lin).to(anchor.dtype)
    raise ValueError(f"unknown blend mode {mode!r}")


def _rescale(a, n):
    cur = a.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (a.float() * (n / cur)).to(a.dtype)


def _layer_full(layer, positions, x, frozen_logits=None):
    """One decoder layer on the full hidden state x; returns (output, router_logits, k, v)."""
    attn = layer.self_attn
    h = layer.input_layernorm(x)
    qkv, _ = attn.qkv_proj(h)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.view(*q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim)).view(q.shape)
    k = attn.k_norm(k.view(*k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim)).view(k.shape)
    q, k = attn.rotary_emb(positions, q, k)
    a = attn.attn(q, k, v)
    a, _ = attn.o_proj(a)
    r = x + a
    hn = layer.post_attention_layernorm(r)
    moe = layer.mlp
    logits = frozen_logits if frozen_logits is not None else moe.gate(hn)[0]
    y = moe.experts(hidden_states=hn, router_logits=logits)
    return r + y, logits, k, v, a


def _layer_mlp_only(layer, x, a_nat, logits):
    """Iteration step with the natural pass's attention output reused: only the MoE runs.

    Keeps cross-token mixing fixed (no re-attention, no KV writes) and refines the
    token's own features: r = x + a_nat; out = r + MoE(norm(r)).
    """
    r = x + a_nat
    hn = layer.post_attention_layernorm(r)
    moe = layer.mlp
    y = moe.experts(hidden_states=hn,
                    router_logits=logits if logits is not None else moe.gate(hn)[0])
    return r + y


def _loop_layer(layer, positions, x, cfg, depth=None):
    K, beta = int(cfg["K"]), float(cfg["beta"])
    if cfg.get("beta_ramp") and depth is not None:
        b0, b1 = (float(v) for v in cfg["beta_ramp"])   # linear beta across the window layers
        beta = b0 if depth[1] < 2 else b0 + (b1 - b0) * depth[0] / (depth[1] - 1)
    s = float(cfg["step"]) if cfg.get("step") is not None else 1.0 / K
    natural, logits, k_nat, v_nat, a_nat = _layer_full(layer, positions, x)
    mlp_only = cfg.get("iter_mode") == "mlp"
    if K > 1:
        n_in = x.float().norm(dim=-1, keepdim=True)
        n_tgt = natural.float().norm(dim=-1, keepdim=True)

        def target(k):
            return n_in + (n_tgt - n_in) * (k / K)

        def run_layer(hh):
            if mlp_only:
                return _layer_mlp_only(layer, hh, a_nat, logits if cfg["freeze"] else None)
            return _layer_full(layer, positions, hh, logits if cfg["freeze"] else None)[0]

        ens = int(cfg.get("ensemble") or 0)
        if ens:
            # Is the gain an ensemble effect rather than a fixed-point one? Same compute,
            # no chain: evaluate the layer at `ens` points spread along the segment
            # [natural, x] and average the outputs, instead of iterating damped Euler.
            # The ODE view predicts this is worse; the ensemble view predicts a tie.
            # ensemble_jitter: give each token its own offset into the grid, so the
            # members differ per token instead of every token seeing the same fixed s.
            # That is stratified sampling -- the variance reduction a fixed grid cannot
            # give -- and a golden-ratio offset off `positions` keeps it deterministic,
            # which torch.rand would not be under cudagraph replay.
            jit = None
            # Sampling is the default: a fixed grid makes every token see the same s,
            # which is a quadrature of the segment, not an ensemble. "grid" opts back in.
            mode = cfg.get("ensemble_random", "rand")
            if mode == "rand":
                # true per-token sampling. Under FULL_DECODE_ONLY the decode step is a
                # captured graph; torch advances the philox offset per replay, but if a
                # capture freezes it this degenerates to a fixed-per-position s -- which
                # is why "hash" below exists as the control.
                jit = torch.rand(x.shape[0], 1, device=x.device, dtype=torch.float32)
            elif mode == "hash":
                # content-derived: varies per token AND per decode step (|x| moves every
                # step), pure arithmetic, so a graph replay cannot freeze it.
                jit = torch.frac(1e4 * x.float().norm(dim=-1, keepdim=True))
            elif cfg.get("ensemble_jitter"):
                jit = torch.frac(0.6180339887498949 * positions.float()).unsqueeze(-1)
            if jit is not None and LOG_DIVERGENCE and _div_logged[0] < 12 \
                    and not torch.cuda.is_current_stream_capturing():
                _div_logged[0] += 1
                qs = torch.tensor([.0, .25, .5, .75, 1.], device=jit.device)
                v = torch.quantile(jit.flatten(), qs).tolist()
                print(f"LOOP_JITTER n={jit.numel()} min={v[0]:.4f} p25={v[1]:.4f} "
                      f"p50={v[2]:.4f} p75={v[3]:.4f} max={v[4]:.4f}", flush=True)
            acc_e = None
            for i in range(1, ens + 1):
                if jit is None:
                    sk = i / (ens + 1.0)                  # "grid": same s for every token
                else:
                    sk = (i - 1.0 + jit) / ens            # per-token shifted grid
                hk = (sk * x.float() + (1.0 - sk) * natural.float()).to(x.dtype)
                if cfg["norm_interp"]:
                    hk = _rescale(hk, n_in + (n_tgt - n_in) * (1.0 - sk))
                yk = run_layer(hk)
                acc_e = yk.float() if acc_e is None else acc_e + yk.float()
            h = (acc_e / ens).to(x.dtype)
        else:
            # pre_rescale: match the two operands' norms before mixing, so s weights their
            # *directions* instead of being skewed by |natural|/|x| (measured median 1.04).
            # "first" does it only for the x + natural step, "all" for every Euler step.
            pre = cfg.get("pre_rescale")

            def run_chain(step):
                h = _blend(_rescale(x, n_tgt) if pre else x, natural, step,
                           cfg.get("step_blend", "linear"))
                if cfg["norm_interp"]:
                    h = _rescale(h, target(1))
                acc = h.float() if cfg.get("out_mode") == "mean" else None
                sched = cfg.get("step_sched")
                off = float(cfg.get("step_sched_offset", 0.0))
                for k in range(1, K):
                    y = run_layer(h)
                    # Robbins-Monro: a step that decays as 1/k instead of the constant 1/K.
                    # With step_blend="slerp" this rotates h toward y by theta/(k+off), the
                    # norm-preserving form of cos(phi) h + sin(phi) y (which only preserves
                    # the norm when h and y are orthogonal -- angle measured at ~12 degrees).
                    sk = 1.0 / (k + off) if sched == "inv" else step
                    hb = _rescale(h, y.float().norm(dim=-1, keepdim=True)) if pre == "all" else h
                    h = _blend(hb, y, sk, cfg.get("step_blend", "linear"))
                    if k < K - 1 and cfg["norm_interp"]:
                        h = _rescale(h, target(k + 1))
                    if acc is not None:
                        acc = acc + h.float()
                if acc is not None:
                    h = (acc / K).to(h.dtype)  # mean of the iterates h_1..h_K
                return h

            ce = int(cfg.get("chain_ens") or 0)
            if ce:
                # M chains with different step sizes, averaged at the end: keeps the
                # sequential structure and adds the ensembling on top of it. Spread the
                # steps over [0.5s, 2s] so the chains actually differ.
                acc_c = None
                for m in range(ce):
                    hm = run_chain(s * (0.5 + 1.5 * m / max(ce - 1, 1)))
                    acc_c = hm.float() if acc_c is None else acc_c + hm.float()
                h = (acc_c / ce).to(x.dtype)
            else:
                h = run_chain(s)
        if DEBUG_NAN and not torch.isfinite(h).all():
            n = int((~torch.isfinite(h)).any(-1).sum())
            print(f"LOOP_NONFINITE layer={layer.self_attn.attn.layer_name} tokens={n}/{h.shape[0]}", flush=True)
        # mlp-only iterations never touch the cache, so the natural K/V is already in place
        if cfg["kv"] == "natural" and not mlp_only:
            a = layer.self_attn.attn
            unified_kv_cache_update(k_nat.view(-1, a.num_kv_heads, a.head_size),
                                    v_nat.view(-1, a.num_kv_heads, a.head_size_v), a.layer_name)
        # kv == "last": keep the K/V the final iteration wrote, so later tokens attend to
        # the iterated (refined) keys/values instead of the natural-pass ones
        head = getattr(layer, "_loop_beta_head", None)
        if head is not None:      # learned per-token beta (train_beta.py); scalar beta unused
            hw, hb = head         # plain tensors: an nn.Module here registers params that
            beta = torch.sigmoid(torch.nn.functional.linear(   # vLLM's loader then rejects
                torch.cat([x, natural], -1).float(), hw, hb)).to(x.dtype)
            out = beta * natural + (1.0 - beta) * h
        else:
            out = _blend(natural, h, 1.0 - beta, cfg.get("blend", "linear"))
        if cfg["anchor_rescale"]:
            out = _rescale(out, n_tgt)
        tau = cfg.get("gate_tau")
        if tau or LOG_DIVERGENCE:
            # How far iteration moved this token, relative to its own scale. The measured
            # gain comes from branch points and every attempt to perturb *more* scored
            # worse, so gate_tau spends the perturbation only where iteration found
            # something: below the threshold the token keeps its natural output.
            rel = ((h.float() - natural.float()).norm(dim=-1)
                   / natural.float().norm(dim=-1).clamp_min(1e-6))
            # vLLM's memory profiling and cudagraph capture run dummy all-zero batches,
            # whose divergence is identically 0 -- skip those or they eat the log budget.
            # rel.max() is a device sync, which is illegal while a graph is being captured.
            if (LOG_DIVERGENCE and _div_logged[0] < 12
                    and not torch.cuda.is_current_stream_capturing()
                    and float(rel.max()) > 0):
                _div_logged[0] += 1
                qs = torch.tensor([.1, .25, .5, .75, .9, .99], device=rel.device)
                v = torch.quantile(rel, qs).tolist()
                print(f"LOOP_DIVERGENCE n={rel.numel()} "
                      f"p10={v[0]:.4f} p25={v[1]:.4f} p50={v[2]:.4f} "
                      f"p75={v[3]:.4f} p90={v[4]:.4f} p99={v[5]:.4f}", flush=True)
            if tau:
                out = torch.where((rel >= float(tau)).unsqueeze(-1), out, natural)
            tmax = cfg.get("gate_tau_max")
            if tmax:
                # the opposite bet: every measured attempt to perturb *more* scored worse,
                # so keep the loop only where it barely moved the representation
                out = torch.where((rel <= float(tmax)).unsqueeze(-1), out, natural)
    else:
        out = natural
    if cfg["res_scale"] != 1.0:
        # scale this layer's whole update: x + a * (out - x); a = 1 leaves `out` untouched
        out = x + cfg["res_scale"] * (out - x)
    return _gate(out, natural, positions, cfg)


def _gate(out, natural, positions, cfg):
    """Keep the looped result only for tokens with pos_min <= position < pos_max."""
    lo, hi = cfg.get("pos_min"), cfg.get("pos_max")
    if lo is None and hi is None:
        return out
    keep = torch.ones_like(positions, dtype=torch.bool)
    if lo is not None:
        keep &= positions >= int(lo)
    if hi is not None:
        keep &= positions < int(hi)
    return torch.where(keep.unsqueeze(-1), out, natural)


def _block_loop(layers, positions, x, cfg):
    """Iterate the whole window g = L_end-1 ∘ … ∘ L_start as one map (HF block_anchored),
    optionally with per-layer routing frozen to the natural path and norm interpolation."""
    K, beta = int(cfg["K"]), float(cfg["beta"])
    s = float(cfg["step"]) if cfg.get("step") is not None else 1.0 / K
    natural, logits, kvs = x, [], []
    for layer in layers:
        natural, lg, k, v, _ = _layer_full(layer, positions, natural)
        logits.append(lg); kvs.append((k, v))
    if K <= 1:
        return natural

    def g(h):
        for layer, lg in zip(layers, logits):
            h, _, _, _, _ = _layer_full(layer, positions, h, lg if cfg["freeze"] else None)
        return h

    n_in = x.float().norm(dim=-1, keepdim=True)
    n_tgt = natural.float().norm(dim=-1, keepdim=True)

    def target(k):
        return n_in + (n_tgt - n_in) * (k / K)

    h = (1.0 - s) * x + s * natural
    if cfg["norm_interp"]:
        h = _rescale(h, target(1))
    for k in range(1, K):
        h = (1.0 - s) * h + s * g(h)
        if k < K - 1 and cfg["norm_interp"]:
            h = _rescale(h, target(k + 1))
    for layer, (k_nat, v_nat) in zip(layers, kvs):
        a = layer.self_attn.attn
        unified_kv_cache_update(k_nat.view(-1, a.num_kv_heads, a.head_size),
                                v_nat.view(-1, a.num_kv_heads, a.head_size_v), a.layer_name)
    out = beta * natural + (1.0 - beta) * h
    if cfg["anchor_rescale"]:
        out = _rescale(out, n_tgt)
    return out


def _windows(cfg):
    ws = cfg["windows"] if cfg.get("windows") is not None else [[cfg["start"], cfg["end"]]]
    return sorted((int(a), int(b)) for a, b in ws)


def _recirculate(self, positions, split_d, z_d, z_s, cfg, loop_layers, depth_of):
    """Recirculation (Mozer et al., arXiv:2608.17981): second pass for the current tokens.

    The destination layer's output is mixed with the norm-matched source-layer output,
        convex:    d' = (1-a) d + a (|d|/|s|) s        nonconvex: d' = d + a (|d|/|s|) s
    and layers dst+1.. are recomputed on it. The pass only rewrites those layers' KV for
    these tokens (later tokens attend to the recirculated state); its output is discarded,
    since the prediction is read out after the first pass. The mix is applied as a delta
    on the saved (hidden, residual) split so that a = 0 reproduces the first pass exactly.
    """
    a = float(cfg["recirc_alpha"])
    zd, zs = z_d.float(), z_s.float()
    # note: with recirc_iters > 1 the pass is repeated, each time re-reading the source
    # from the freshly recomputed stack (the paper's k-iteration variant)
    src = zs * (zd.norm(dim=-1, keepdim=True) / zs.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    mix = (1.0 - a) * zd + a * src if cfg.get("recirc_convex", True) else zd + a * src
    dst, src = int(cfg["recirc_dst"]), int(cfg["recirc_src"])
    for _ in range(int(cfg.get("recirc_iters", 1))):
        # clone: the layers' fused add-norm mutates these in place, so a second
        # iteration must not reuse the tensors the previous pass overwrote
        hidden = split_d[0].clone() + (mix - zd).to(split_d[0].dtype)
        residual = None if split_d[1] is None else split_d[1].clone()
        new_src = None
        for j in range(dst + 1, self.end_layer):
            layer = self.layers[j]
            if j in loop_layers:
                # window layers may have had their MoE gate detached (freeze): full-state path
                x = hidden if residual is None else hidden + residual
                if cfg.get("recirc_loop"):
                    # re-apply the loop on the second pass too. Without this the pass
                    # rewrites the K/V of every layer above dst from a *non-looped* state,
                    # which silently undoes the loop for those layers -- the reason
                    # loop+recirculation looked non-additive.
                    x = _loop_layer(layer, positions, x, cfg, depth_of[j])
                else:
                    x, _, _, _, _ = _layer_full(layer, positions, x)
                hidden, residual = x, None
            else:
                hidden, residual = layer(positions, hidden, residual)
            if j == src:
                new_src = (hidden if residual is None else hidden + residual).float()
        if new_src is None:
            break                                   # source is above the destination only
        src2 = new_src * (zd.norm(dim=-1, keepdim=True) / new_src.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        mix = (1.0 - a) * zd + a * src2 if cfg.get("recirc_convex", True) else zd + a * src2


def _looped_model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    cfg = self._loop_cfg
    hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    residual = None
    recirc = bool(cfg.get("recirc_alpha"))
    dst, src = (int(cfg["recirc_dst"]), int(cfg["recirc_src"])) if recirc else (-1, -1)
    full, split = {}, None       # full states of dst/src; (hidden, residual) split after dst

    def keep(j, h, r):
        nonlocal split
        if j == dst:
            split = (h.clone(), None if r is None else r.clone())
        if j in (dst, src):
            full[j] = h if r is None else h + r

    windows = _windows(cfg)
    loop_layers = {j for a, b in windows for j in range(a, b)}
    depth_of = {j: (j - a, b - a) for a, b in windows for j in range(a, b)}
    i = self.start_layer
    for start, end in windows:
        for j in range(i, start):
            hidden, residual = self.layers[j](positions, hidden, residual)
            keep(j, hidden, residual)
        x = hidden if residual is None else hidden + residual
        if cfg["mode"] == "block":
            x = _block_loop([self.layers[j] for j in range(start, end)], positions, x, cfg)
            keep(end - 1, x, None)
        else:
            for j in range(start, end):
                x = _loop_layer(self.layers[j], positions, x, cfg, (j - start, end - start))
                keep(j, x, None)
        hidden, residual, i = x, None, end
    for j in range(i, self.end_layer):
        hidden, residual = self.layers[j](positions, hidden, residual)
        keep(j, hidden, residual)
    out, _ = self.norm(hidden, residual)
    if recirc:
        _recirculate(self, positions, split, full[dst], full[src], cfg, loop_layers, depth_of)
    return out


class LoopedQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        user = getattr(vllm_config.model_config.hf_config, "loop_cfg", None) or {}
        self._loop_cfg = {**DEFAULT_CFG, **dict(user)} if user.get("enabled", True) else None
        if self._loop_cfg is not None:
            self.model._loop_cfg = self._loop_cfg
            self.model.forward = types.MethodType(_looped_model_forward, self.model)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        cfg = self._loop_cfg or {}
        if cfg.get("beta_head"):
            ckpt = torch.load(cfg["beta_head"], map_location="cpu")
            for n, li in enumerate(ckpt["window"]):
                dev = self.model.layers[li].mlp.gate.weight.device
                self.model.layers[li]._loop_beta_head = (
                    ckpt["state"][f"{n}.weight"].float().to(dev),
                    ckpt["state"][f"{n}.bias"].float().to(dev))
        if cfg.get("topk"):
            # widen the MoE: more experts per token in the given layer range (training-free
            # capacity increase, orthogonal to looping). Routing math is unchanged otherwise.
            a, b = cfg.get("topk_range") or (0, len(self.model.layers))
            for i in range(int(a), int(b)):
                mlp = self.model.layers[i].mlp
                if hasattr(mlp, "experts") and hasattr(mlp.experts, "router"):
                    mlp.experts.router.top_k = int(cfg["topk"])
        if self._loop_cfg is not None and self._loop_cfg["freeze"]:
            # Let the window layers' MoE runners take router logits from the caller
            # (natural pass or frozen) instead of recomputing them from the input.
            for start, end in _windows(self._loop_cfg):
                for i in range(start, end):
                    self.model.layers[i].mlp.experts.gate = None
        return loaded


def register():
    from vllm import ModelRegistry
    if "LoopedQwen3MoeForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("LoopedQwen3MoeForCausalLM",
                                     "looped_qwen3_moe:LoopedQwen3MoeForCausalLM")
