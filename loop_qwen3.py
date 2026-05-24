"""
Training-free layer looping for Qwen3.

Given an unmodified Qwen3 checkpoint, run the chosen contiguous block of
middle layers K times, with one of several blending strategies, then resume
the remaining layers as normal.

This patches `Qwen3Model.forward` at the instance level. KV caching is
disabled inside the custom forward (fine for loglikelihood-style eval).

Strategy taxonomy
-----------------
The looped block defines a map g(x) = block(x) (=running the contiguous
loop window once). Looping K times is then a fixed-point iteration on g.

  * naive            : x_{k+1} = g(x_k)                        (Picard)
  * residual_scaled  : x_K = x_0 + (g^K(x_0) - x_0) / K
  * ema / euler      : x_{k+1} = (1-α) x_k + α g(x_k)          (damped Picard)
                       (ema with α and Euler with h=α are algebraically identical)
  * midpoint/heun/rk4: classical RK on f(x) = g(x) - x         (no acceleration in
                       practice — block is not a smooth ODE field)
  * ema_sched        : as ema but with per-step α schedule
  * heavy_ball       : x_{k+1} = x_k + α(g(x_k)-x_k) + β(x_k - x_{k-1})
                       (Polyak momentum on the residual)
  * anderson         : Walker–Ni Type-II Anderson(m) acceleration of damped
                       Picard with damping β. m=2,3 typical.
  * aitken           : Per-coordinate Aitken Δ² extrapolation; uses 2 g calls
                       per Aitken step, so K must be even.
"""

from __future__ import annotations

from types import MethodType
from typing import List, Literal, Optional, Sequence

import torch
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)


Strategy = Literal[
    "naive", "residual_scaled", "ema", "euler",
    "midpoint", "heun", "rk4",
    "ema_sched", "heavy_ball", "anderson", "aitken",
    "norm_stab", "poly_blend", "uniform", "per_layer_anchored",
    "block_anchored",
]
CacheStrategy = Literal["last", "first", "none"]
LoopMode = Literal["block", "layer"]
# block: output = g^K(x) with g = L_b∘…∘L_a (the whole window iterated jointly).
# layer: output = L_b^K ∘ … ∘ L_a^K (x) — each layer iterated K times before
#        passing to the next.

DecodeMode = Literal["bypass", "full", "first_n"]
# bypass : seq_len=1 (incremental decode) skips the loop entirely (current default).
# full   : every decode token also goes through the K-loop.
# first_n: only the first `decode_first_n` decode tokens after each prefill
#          go through the loop; subsequent ones bypass.


def _iterate_loop(
    x_init: torch.Tensor,
    run_block_once,
    *,
    strategy: str,
    K: int,
    ema_alpha: float = 0.5,
    momentum_beta: float = 0.3,
    anderson_m: int = 2,
    anderson_beta: float = 1.0,
    per_step_alphas: Optional[Sequence[float]] = None,
    halt_tau: Optional[float] = None,
):
    """Apply the chosen fixed-point iteration K times starting from x_init.

    `run_block_once(x)` should compute one application of the loop window,
    returning a tensor of the same shape as x.
    """
    eps = 1e-6

    def _halted(x_new, x_old):
        if halt_tau is None:
            return False
        with torch.no_grad():
            delta = (x_new - x_old).float().norm()
            denom = x_old.float().norm().clamp_min(eps)
            return float(delta / denom) < float(halt_tau)

    if strategy == "naive":
        x = x_init
        for _ in range(K):
            x = run_block_once(x)
        return x

    if strategy == "residual_scaled":
        y = x_init
        for _ in range(K):
            y = run_block_once(y)
        return x_init + (y - x_init) / K

    if strategy == "ema":
        x = x_init
        for _ in range(K):
            y = run_block_once(x)
            x_new = ema_alpha * y + (1.0 - ema_alpha) * x
            if _halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "ema_sched":
        assert per_step_alphas is not None and len(per_step_alphas) == K, (
            f"ema_sched needs per_step_alphas of length K={K}, got "
            f"{None if per_step_alphas is None else len(per_step_alphas)}"
        )
        x = x_init
        for k in range(K):
            a = float(per_step_alphas[k])
            y = run_block_once(x)
            x_new = a * y + (1.0 - a) * x
            if _halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "heavy_ball":
        a = float(ema_alpha)
        b = float(momentum_beta)
        x = x_init
        x_prev = x_init
        for _ in range(K):
            G = run_block_once(x)
            x_new = x + a * (G - x) + b * (x - x_prev)
            if _halted(x_new, x):
                return x_new
            x_prev = x
            x = x_new
        return x

    if strategy == "anderson":
        m_max = max(1, int(anderson_m))
        beta = float(anderson_beta)
        reg = 1e-6

        x = x_init
        hist_x: List[torch.Tensor] = []
        hist_g: List[torch.Tensor] = []

        for _ in range(K):
            G = run_block_once(x)
            hist_x.append(x)
            hist_g.append(G)
            if len(hist_x) > m_max + 1:
                hist_x = hist_x[-(m_max + 1):]
                hist_g = hist_g[-(m_max + 1):]

            m_eff = len(hist_x) - 1
            if m_eff == 0:
                # First iteration: damped Picard step (no history yet).
                x_new = (1.0 - beta) * x + beta * G
            else:
                # Walker–Ni Type-II Anderson:
                #   γ = argmin || f_t - ΔF γ ||
                #   x_{t+1} = (1-β)(x_t - ΔX γ) + β(G_t - ΔF γ)
                # where f_i = g_i - x_i, ΔF[:,i] = f_{i+1} - f_i, ΔX similarly.
                # Use fp32 for the LS to avoid bf16 noise.
                dF_list = [
                    (hist_g[i + 1] - hist_x[i + 1]) - (hist_g[i] - hist_x[i])
                    for i in range(m_eff)
                ]
                dX_list = [hist_x[i + 1] - hist_x[i] for i in range(m_eff)]
                dF = torch.stack(dF_list, dim=-1).float()  # (..., m_eff)
                dX = torch.stack(dX_list, dim=-1).float()
                f_last = (G - x).float()

                m = m_eff
                dF_flat = dF.reshape(-1, m)
                f_flat = f_last.reshape(-1)

                # Normal equations with adaptive Tikhonov regularization.
                A = dF_flat.T @ dF_flat  # (m, m)
                diag_max = A.diagonal().abs().max().clamp_min(1e-8)
                A = A + reg * diag_max * torch.eye(m, device=A.device, dtype=A.dtype)
                b_vec = dF_flat.T @ f_flat
                try:
                    gamma = torch.linalg.solve(A, b_vec)
                except Exception:
                    gamma = torch.zeros(m, device=A.device, dtype=A.dtype)

                gamma_view = gamma.view(*([1] * (dX.ndim - 1)), m)
                upd_X = (dX * gamma_view).sum(dim=-1).to(x.dtype)
                upd_F = (dF * gamma_view).sum(dim=-1).to(x.dtype)
                x_new = (1.0 - beta) * (x - upd_X) + beta * (G - upd_F)

            if _halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "aitken":
        # Per-coordinate Aitken Δ² (Steffensen) with safeguards.
        # Done in fp32 since d2 = g2 - 2g1 + x is a small second difference and
        # bf16 noise dominates without it. Per-coordinate update is clipped to
        # |Δ| ≤ |d1| so we never take a bigger step than naive Picard.
        assert K % 2 == 0, "aitken requires even K (uses 2 g calls per step)"
        x = x_init
        for _ in range(K // 2):
            g1 = run_block_once(x)
            g2 = run_block_once(g1)
            d1 = (g1 - x).float()
            d2 = (g2 - 2.0 * g1 + x).float()
            # Safeguard: only accelerate where the second difference is
            # meaningful relative to the first; otherwise just take g1.
            mask = d2.abs() > (1e-3 * d1.abs() + 1e-4)
            update = torch.where(mask, (d1 * d1) / d2.where(mask, torch.ones_like(d2)),
                                 -d1)  # no Aitken → fall back to x ← g1 (i.e., x - (-d1))
            # Clip per-coordinate magnitude to |d1|.
            clip = d1.abs()
            update = update.clamp(min=-clip, max=clip)
            x_new = (x.float() - update).to(x.dtype)
            if _halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "norm_stab":
        # Norm-stabilized damped Picard. After each blended step, rescale per
        # token to preserve the L2 norm of the loop-window entry state. This
        # holds the post-loop hidden state in the same norm regime the coda
        # layers were trained on, which Phase 6 suggests is the binding
        # constraint on K>2.
        a = float(ema_alpha)
        x = x_init
        target_norm = x_init.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        for _ in range(K):
            y = run_block_once(x)
            x_prop = a * y + (1.0 - a) * x
            cur_norm = x_prop.float().norm(dim=-1, keepdim=True).clamp_min(eps)
            x_new = (x_prop.float() * (target_norm / cur_norm)).to(x.dtype)
            if _halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "poly_blend":
        # Polynomial blend: output = Σ_{k=0..K} w_k · g^k(x_0), Σ w_k = 1.
        # Distinct from EMA: EMA blends the *blended* state at each step
        # (compositional), poly_blend blends pure powers of g. Lets us search
        # the convex hull of {x_0, g(x_0), g²(x_0), ...} directly.
        assert per_step_alphas is not None and len(per_step_alphas) == K + 1, (
            f"poly_blend needs per_step_alphas of length K+1={K+1} (weights w_0..w_K), "
            f"got {len(per_step_alphas) if per_step_alphas is not None else None}"
        )
        weights = [float(w) for w in per_step_alphas]
        g_iter = x_init
        out = weights[0] * g_iter
        for k in range(1, K + 1):
            g_iter = run_block_once(g_iter)
            out = out + weights[k] * g_iter
        return out

    if strategy == "input_anchor":
        # Parcae-style input-anchored damping. Unlike `ema` (which blends g(x)
        # with the previous iterate x), this blends g(x) with the *entry*
        # state x_init that never changes:
        #   x_{t+1} = α · g(x_t) + (1 − α) · x_init
        # The fixed point is unchanged from naive Picard (any x* with g(x*)=x*
        # remains a fixed point only if x* = x_init, otherwise the bias toward
        # x_init shifts it). What this DOES buy is drift-resistance: the
        # iterate stays in the convex hull of {x_init, g(x_init), g²(x_init),
        # …}, and as α → 1 it recovers naive. Useful for K beyond the trained
        # depth, where naive starts to drift but the model still expects the
        # output to live near the training-time iterate manifold.
        a = float(ema_alpha)
        x = x_init
        for _ in range(K):
            y = run_block_once(x)
            x_new = a * y + (1.0 - a) * x_init
            if _halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy in ("euler", "midpoint", "heun", "rk4"):
        h = 1.0 / K

        def _f(z):
            return run_block_once(z) - z

        x = x_init
        if strategy == "euler":
            for _ in range(K):
                x_new = x + h * _f(x)
                if _halted(x_new, x):
                    return x_new
                x = x_new
        elif strategy == "midpoint":
            for _ in range(K):
                k1 = _f(x)
                k2 = _f(x + (h / 2.0) * k1)
                x = x + h * k2
        elif strategy == "heun":
            for _ in range(K):
                k1 = _f(x)
                k2 = _f(x + h * k1)
                x = x + (h / 2.0) * (k1 + k2)
        elif strategy == "rk4":
            for _ in range(K):
                k1 = _f(x)
                k2 = _f(x + (h / 2.0) * k1)
                k3 = _f(x + (h / 2.0) * k2)
                k4 = _f(x + h * k3)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    raise ValueError(f"unknown strategy: {strategy!r}")


def _gather_loop_kwargs(self):
    return dict(
        strategy=self._loop_strategy,
        K=self._loop_K,
        ema_alpha=getattr(self, "_loop_ema_alpha", 0.5),
        momentum_beta=getattr(self, "_loop_momentum_beta", 0.3),
        anderson_m=getattr(self, "_loop_anderson_m", 2),
        anderson_beta=getattr(self, "_loop_anderson_beta", 1.0),
        per_step_alphas=getattr(self, "_loop_per_step_alphas", None),
        halt_tau=getattr(self, "_loop_halt_tau", None),
    )


def _apply_loop(hidden_states, run_block_once, run_one_layer, loop_start, loop_end,
                loop_mode, kw):
    """Dispatch between block-mode (one g^K call) and layer-mode (per-layer K)."""
    if loop_mode == "block":
        return _iterate_loop(hidden_states, run_block_once, **kw)
    # layer: iterate each layer K times before moving on
    x = hidden_states
    for li in range(loop_start, loop_end):
        f = run_one_layer(li)
        x = _iterate_loop(x, f, **kw)
    return x


def _looped_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask=None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    loop_indices: List[int] = self._loop_indices

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        from transformers.cache_utils import DynamicCache
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(self, "has_sliding_layers", False):
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    n_layers = self.config.num_hidden_layers
    loop_indices_sorted = sorted(loop_indices)
    loop_start = loop_indices_sorted[0]
    loop_end = loop_indices_sorted[-1] + 1
    assert loop_indices_sorted == list(range(loop_start, loop_end)), (
        "loop_indices must be contiguous"
    )

    seq_len = inputs_embeds.shape[1]
    is_incremental = (past_key_values is not None) and (seq_len == 1)

    # Decode counter: reset on prefill, increment on each decode step.
    if not is_incremental:
        self._loop_decode_count = 0
    else:
        self._loop_decode_count = getattr(self, "_loop_decode_count", 0) + 1

    decode_mode = getattr(self, "_loop_decode_mode", "bypass")
    if not is_incremental:
        do_loop = True
    elif decode_mode == "bypass":
        do_loop = False
    elif decode_mode == "full":
        do_loop = True
    elif decode_mode == "first_n":
        do_loop = self._loop_decode_count <= getattr(self, "_loop_decode_first_n", 0)
    else:
        raise ValueError(f"unknown decode_mode: {decode_mode!r}")

    def _call_layer(layer, x, *, pkv, uc):
        return layer(
            x,
            attention_mask=causal_mask_mapping[layer.attention_type],
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=uc,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def _call_no_cache(layer, x):
        return _call_layer(layer, x, pkv=None, uc=False)

    def _call_with_cache(layer, x):
        return _call_layer(layer, x, pkv=past_key_values, uc=use_cache)

    # Loop body during prefill: pkv=None (no past to attend to within the window;
    # tokens attend to each other within the seq).
    def _run_block_once_prefill(x):
        for li in range(loop_start, loop_end):
            x = _call_no_cache(self.layers[li], x)
        return x

    def _run_one_layer_prefill(li):
        layer = self.layers[li]
        return lambda z: _call_no_cache(layer, z)

    # Loop body during decode (seq_len=1, has past KV): we MUST attend to past
    # KV for the new token to be correct, but DynamicCache.update always appends,
    # so naive K iterations would write K extra entries per loop layer at the
    # same logical decode position. Snapshot cache references before each
    # iteration and restore after, so the loop body has no net cache effect.
    def _run_block_once_decode(x):
        # Snapshot pre-loop seq lengths in the loop region; restore via crop
        # so the loop body has no net cache effect.
        saved_lens = [past_key_values.layers[li].get_seq_length()
                      for li in range(loop_start, loop_end)]
        for li in range(loop_start, loop_end):
            x = _call_with_cache(self.layers[li], x)
        for off, li in enumerate(range(loop_start, loop_end)):
            past_key_values.layers[li].crop(saved_lens[off])
        return x

    def _run_one_layer_decode(li):
        layer = self.layers[li]
        def f(z):
            saved_len = past_key_values.layers[li].get_seq_length()
            z = _call_with_cache(layer, z)
            past_key_values.layers[li].crop(saved_len)
            return z
        return f

    for idx in range(0, loop_start):
        hidden_states = _call_with_cache(self.layers[idx], hidden_states)

    if not do_loop:
        # Bypass: just stream through loop layers + downstream as a normal forward.
        for li in range(loop_start, loop_end):
            hidden_states = _call_with_cache(self.layers[li], hidden_states)
        for idx in range(loop_end, n_layers):
            hidden_states = _call_with_cache(self.layers[idx], hidden_states)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    if is_incremental:
        run_block_once = _run_block_once_decode
        run_one_layer = _run_one_layer_decode
    else:
        run_block_once = _run_block_once_prefill
        run_one_layer = _run_one_layer_prefill

    x_in_for_cache = hidden_states
    loop_mode = getattr(self, "_loop_mode", "block")
    strategy = getattr(self, "_loop_strategy", "naive")
    if strategy == "uniform":
        # looped-transformer "uniform" interpolation: full-window iteration K times,
        # but at each layer position within the window, the hidden state passed to
        # the next layer is the running mean of that layer's outputs across all
        # iterations so far. O(n_loop_layers) memory via incremental running sum.
        K = int(getattr(self, "_loop_K", 1))
        n_loop_layers = loop_end - loop_start
        running_sums = [None] * n_loop_layers
        counts = [0] * n_loop_layers
        x = hidden_states
        for _t in range(K):
            if is_incremental:
                saved_lens = [past_key_values.layers[li].get_seq_length()
                              for li in range(loop_start, loop_end)]
            for j in range(n_loop_layers):
                li = loop_start + j
                if is_incremental:
                    x = _call_with_cache(self.layers[li], x)
                else:
                    x = _call_no_cache(self.layers[li], x)
                if running_sums[j] is None:
                    running_sums[j] = x
                else:
                    running_sums[j] = running_sums[j] + x
                counts[j] += 1
                x = running_sums[j] / counts[j]
            if is_incremental:
                for off, li in enumerate(range(loop_start, loop_end)):
                    past_key_values.layers[li].crop(saved_lens[off])
        hidden_states = x
    elif strategy == "block_anchored":
        # Block-mode damped Euler with anchor at window exit:
        #   natural = g(h_in)                              # one full window pass
        #   h_iter = (1-1/K) h_in + (1/K) natural          # step 1 (reuses natural)
        #   for t = 2..K: h_iter = (1-1/K) h_iter + (1/K) g(h_iter)
        #   h_out = β · natural + (1-β) · h_iter           # anchor only at window exit
        # β=0 reduces to pure block Euler. Single anchor at window-level rather
        # than per-layer (compare per_layer_anchored which anchors each layer).
        K = int(getattr(self, "_loop_K", 1))
        beta = float(getattr(self, "_loop_anchor_beta", 0.0))
        # Natural single-pass through window
        if is_incremental:
            saved_lens = [past_key_values.layers[li].get_seq_length()
                          for li in range(loop_start, loop_end)]
            natural = hidden_states
            for li in range(loop_start, loop_end):
                natural = _call_with_cache(self.layers[li], natural)
            for off, li in enumerate(range(loop_start, loop_end)):
                past_key_values.layers[li].crop(saved_lens[off])
        else:
            natural = hidden_states
            for li in range(loop_start, loop_end):
                natural = _call_no_cache(self.layers[li], natural)
        h_step = 1.0 / K
        h_iter = hidden_states + h_step * (natural - hidden_states)
        for _t in range(1, K):
            if is_incremental:
                saved_lens = [past_key_values.layers[li].get_seq_length()
                              for li in range(loop_start, loop_end)]
                g_h = h_iter
                for li in range(loop_start, loop_end):
                    g_h = _call_with_cache(self.layers[li], g_h)
                for off, li in enumerate(range(loop_start, loop_end)):
                    past_key_values.layers[li].crop(saved_lens[off])
            else:
                g_h = h_iter
                for li in range(loop_start, loop_end):
                    g_h = _call_no_cache(self.layers[li], g_h)
            h_iter = h_iter + h_step * (g_h - h_iter)
        hidden_states = beta * natural + (1.0 - beta) * h_iter
    elif strategy == "per_layer_anchored":
        # Per-layer damped Euler with per-layer output anchor:
        #   for each layer ℓ in window:
        #     natural = L_ℓ(h_in)                 # iter-0 (in-distribution)
        #     h_iter  = (1-1/K) h_in + (1/K) natural
        #     repeat K-1 times: h_iter = (1-1/K) h_iter + (1/K) L_ℓ(h_iter)
        #     h_out = β · natural + (1-β) · h_iter   # anchor blend
        # β=0  → pure per-layer Euler (recovers K=2 +0.749 winner)
        # β=1  → baseline (no loop)
        # Breaks the layer-by-layer drift cascade: every layer's output is
        # re-anchored to its in-distribution iter-0 application before the
        # next layer sees it.
        K = int(getattr(self, "_loop_K", 1))
        beta = float(getattr(self, "_loop_anchor_beta", 0.0))
        h = hidden_states
        for li in range(loop_start, loop_end):
            if is_incremental:
                saved_len = past_key_values.layers[li].get_seq_length()
                natural = _call_with_cache(self.layers[li], h)
                past_key_values.layers[li].crop(saved_len)
            else:
                natural = _call_no_cache(self.layers[li], h)
            h_iter = (1.0 - 1.0 / K) * h + (1.0 / K) * natural
            for _t in range(1, K):
                if is_incremental:
                    y = _call_with_cache(self.layers[li], h_iter)
                    past_key_values.layers[li].crop(saved_len)
                else:
                    y = _call_no_cache(self.layers[li], h_iter)
                h_iter = (1.0 - 1.0 / K) * h_iter + (1.0 / K) * y
            h = beta * natural + (1.0 - beta) * h_iter
        hidden_states = h
    else:
        hidden_states = _apply_loop(
            hidden_states, run_block_once, run_one_layer,
            loop_start, loop_end, loop_mode, _gather_loop_kwargs(self),
        )

    cache_strategy = getattr(self, "_loop_cache_strategy", "last")
    if use_cache and past_key_values is not None and cache_strategy != "none":
        _stash = hidden_states if cache_strategy == "last" else x_in_for_cache
        for li in range(loop_start, loop_end):
            _stash = _call_with_cache(self.layers[li], _stash)

    for idx in range(loop_end, n_layers):
        hidden_states = _call_with_cache(self.layers[idx], hidden_states)

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def _looped_forward_moe(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask=None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> MoeModelOutputWithPast:
    """Looped forward for Qwen3MoeModel (single causal mask, MoE outputs)."""
    loop_indices: List[int] = self._loop_indices

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        from transformers.cache_utils import DynamicCache
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = (
        create_causal_mask if self.config.sliding_window is None
        else create_sliding_window_causal_mask
    )
    causal_mask = mask_function(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    n_layers = self.config.num_hidden_layers
    loop_indices_sorted = sorted(loop_indices)
    loop_start = loop_indices_sorted[0]
    loop_end = loop_indices_sorted[-1] + 1
    assert loop_indices_sorted == list(range(loop_start, loop_end)), (
        "loop_indices must be contiguous"
    )

    seq_len = inputs_embeds.shape[1]
    is_incremental = (past_key_values is not None) and (seq_len == 1)

    if not is_incremental:
        self._loop_decode_count = 0
    else:
        self._loop_decode_count = getattr(self, "_loop_decode_count", 0) + 1

    decode_mode = getattr(self, "_loop_decode_mode", "bypass")
    if not is_incremental:
        do_loop = True
    elif decode_mode == "bypass":
        do_loop = False
    elif decode_mode == "full":
        do_loop = True
    elif decode_mode == "first_n":
        do_loop = self._loop_decode_count <= getattr(self, "_loop_decode_first_n", 0)
    else:
        raise ValueError(f"unknown decode_mode: {decode_mode!r}")

    def _call_layer(layer, x, *, pkv, uc):
        return layer(
            x,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=uc,
            cache_position=cache_position,
            **kwargs,
        )

    def _call_no_cache(layer, x):
        return _call_layer(layer, x, pkv=None, uc=False)

    def _call_with_cache(layer, x):
        return _call_layer(layer, x, pkv=past_key_values, uc=use_cache)

    def _run_block_once_prefill(x):
        for li in range(loop_start, loop_end):
            x = _call_no_cache(self.layers[li], x)
        return x

    def _run_one_layer_prefill(li):
        layer = self.layers[li]
        return lambda z: _call_no_cache(layer, z)

    def _run_block_once_decode(x):
        # Snapshot pre-loop seq lengths in the loop region; restore via crop
        # so the loop body has no net cache effect.
        saved_lens = [past_key_values.layers[li].get_seq_length()
                      for li in range(loop_start, loop_end)]
        for li in range(loop_start, loop_end):
            x = _call_with_cache(self.layers[li], x)
        for off, li in enumerate(range(loop_start, loop_end)):
            past_key_values.layers[li].crop(saved_lens[off])
        return x

    def _run_one_layer_decode(li):
        layer = self.layers[li]
        def f(z):
            saved_len = past_key_values.layers[li].get_seq_length()
            z = _call_with_cache(layer, z)
            past_key_values.layers[li].crop(saved_len)
            return z
        return f

    for idx in range(0, loop_start):
        hidden_states = _call_with_cache(self.layers[idx], hidden_states)

    if not do_loop:
        for li in range(loop_start, loop_end):
            hidden_states = _call_with_cache(self.layers[li], hidden_states)
        for idx in range(loop_end, n_layers):
            hidden_states = _call_with_cache(self.layers[idx], hidden_states)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    if is_incremental:
        run_block_once = _run_block_once_decode
        run_one_layer = _run_one_layer_decode
    else:
        run_block_once = _run_block_once_prefill
        run_one_layer = _run_one_layer_prefill

    x_in_for_cache = hidden_states
    loop_mode = getattr(self, "_loop_mode", "block")
    strategy = getattr(self, "_loop_strategy", "naive")
    if strategy == "block_anchored":
        K = int(getattr(self, "_loop_K", 1))
        beta = float(getattr(self, "_loop_anchor_beta", 0.0))
        if is_incremental:
            saved_lens = [past_key_values.layers[li].get_seq_length()
                          for li in range(loop_start, loop_end)]
            natural = hidden_states
            for li in range(loop_start, loop_end):
                natural = _call_with_cache(self.layers[li], natural)
            for off, li in enumerate(range(loop_start, loop_end)):
                past_key_values.layers[li].crop(saved_lens[off])
        else:
            natural = hidden_states
            for li in range(loop_start, loop_end):
                natural = _call_no_cache(self.layers[li], natural)
        h_step = 1.0 / K
        h_iter = hidden_states + h_step * (natural - hidden_states)
        for _t in range(1, K):
            if is_incremental:
                saved_lens = [past_key_values.layers[li].get_seq_length()
                              for li in range(loop_start, loop_end)]
                g_h = h_iter
                for li in range(loop_start, loop_end):
                    g_h = _call_with_cache(self.layers[li], g_h)
                for off, li in enumerate(range(loop_start, loop_end)):
                    past_key_values.layers[li].crop(saved_lens[off])
            else:
                g_h = h_iter
                for li in range(loop_start, loop_end):
                    g_h = _call_no_cache(self.layers[li], g_h)
            h_iter = h_iter + h_step * (g_h - h_iter)
        hidden_states = beta * natural + (1.0 - beta) * h_iter
    else:
        hidden_states = _apply_loop(
            hidden_states, run_block_once, run_one_layer,
            loop_start, loop_end, loop_mode, _gather_loop_kwargs(self),
        )

    cache_strategy = getattr(self, "_loop_cache_strategy", "last")
    if use_cache and past_key_values is not None and cache_strategy != "none":
        _stash = hidden_states if cache_strategy == "last" else x_in_for_cache
        for li in range(loop_start, loop_end):
            _stash = _call_with_cache(self.layers[li], _stash)

    for idx in range(loop_end, n_layers):
        hidden_states = _call_with_cache(self.layers[idx], hidden_states)

    hidden_states = self.norm(hidden_states)
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def _looped_forward_qwen2moe(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask=None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> MoeModelOutputWithPast:
    """Looped forward for Qwen2MoeModel (e.g. Qwen1.5-MoE-A2.7B).

    Differences from _looped_forward_moe (Qwen3-MoE):
      * uses self._update_causal_mask (no create_causal_mask in masking_utils)
      * layer.forward returns a tuple (hidden_states, *opts); we take [0]
      * layer takes output_attentions, output_router_logits args
    """
    loop_indices: List[int] = self._loop_indices

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if use_cache and past_key_values is None:
        from transformers.cache_utils import DynamicCache
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, False
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    n_layers = self.config.num_hidden_layers
    loop_indices_sorted = sorted(loop_indices)
    loop_start = loop_indices_sorted[0]
    loop_end = loop_indices_sorted[-1] + 1
    assert loop_indices_sorted == list(range(loop_start, loop_end)), (
        "loop_indices must be contiguous"
    )

    seq_len = inputs_embeds.shape[1]
    is_incremental = (past_key_values is not None) and (seq_len == 1)

    if not is_incremental:
        self._loop_decode_count = 0
    else:
        self._loop_decode_count = getattr(self, "_loop_decode_count", 0) + 1

    decode_mode = getattr(self, "_loop_decode_mode", "bypass")
    if not is_incremental:
        do_loop = True
    elif decode_mode == "bypass":
        do_loop = False
    elif decode_mode == "full":
        do_loop = True
    elif decode_mode == "first_n":
        do_loop = self._loop_decode_count <= getattr(self, "_loop_decode_first_n", 0)
    else:
        raise ValueError(f"unknown decode_mode: {decode_mode!r}")

    def _call_layer(layer, x, *, pkv, uc):
        out = layer(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=uc,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=False,
            output_router_logits=False,
        )
        # layer returns a tuple (hidden_states, ...); we want hidden_states only
        return out[0] if isinstance(out, tuple) else out

    def _call_no_cache(layer, x):
        return _call_layer(layer, x, pkv=None, uc=False)

    def _call_with_cache(layer, x):
        return _call_layer(layer, x, pkv=past_key_values, uc=use_cache)

    def _run_block_once_prefill(x):
        for li in range(loop_start, loop_end):
            x = _call_no_cache(self.layers[li], x)
        return x

    def _run_one_layer_prefill(li):
        layer = self.layers[li]
        return lambda z: _call_no_cache(layer, z)

    def _run_block_once_decode(x):
        saved_lens = [past_key_values.layers[li].get_seq_length()
                      for li in range(loop_start, loop_end)]
        for li in range(loop_start, loop_end):
            x = _call_with_cache(self.layers[li], x)
        for off, li in enumerate(range(loop_start, loop_end)):
            past_key_values.layers[li].crop(saved_lens[off])
        return x

    def _run_one_layer_decode(li):
        layer = self.layers[li]
        def f(z):
            saved_len = past_key_values.layers[li].get_seq_length()
            z = _call_with_cache(layer, z)
            past_key_values.layers[li].crop(saved_len)
            return z
        return f

    for idx in range(0, loop_start):
        hidden_states = _call_with_cache(self.layers[idx], hidden_states)

    if not do_loop:
        for li in range(loop_start, loop_end):
            hidden_states = _call_with_cache(self.layers[li], hidden_states)
        for idx in range(loop_end, n_layers):
            hidden_states = _call_with_cache(self.layers[idx], hidden_states)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    if is_incremental:
        run_block_once = _run_block_once_decode
        run_one_layer = _run_one_layer_decode
    else:
        run_block_once = _run_block_once_prefill
        run_one_layer = _run_one_layer_prefill

    x_in_for_cache = hidden_states
    loop_mode = getattr(self, "_loop_mode", "block")
    strategy = getattr(self, "_loop_strategy", "naive")
    if strategy == "block_anchored":
        K = int(getattr(self, "_loop_K", 1))
        beta = float(getattr(self, "_loop_anchor_beta", 0.0))
        if is_incremental:
            saved_lens = [past_key_values.layers[li].get_seq_length()
                          for li in range(loop_start, loop_end)]
            natural = hidden_states
            for li in range(loop_start, loop_end):
                natural = _call_with_cache(self.layers[li], natural)
            for off, li in enumerate(range(loop_start, loop_end)):
                past_key_values.layers[li].crop(saved_lens[off])
        else:
            natural = hidden_states
            for li in range(loop_start, loop_end):
                natural = _call_no_cache(self.layers[li], natural)
        h_step = 1.0 / K
        h_iter = hidden_states + h_step * (natural - hidden_states)
        for _t in range(1, K):
            if is_incremental:
                saved_lens = [past_key_values.layers[li].get_seq_length()
                              for li in range(loop_start, loop_end)]
                g_h = h_iter
                for li in range(loop_start, loop_end):
                    g_h = _call_with_cache(self.layers[li], g_h)
                for off, li in enumerate(range(loop_start, loop_end)):
                    past_key_values.layers[li].crop(saved_lens[off])
            else:
                g_h = h_iter
                for li in range(loop_start, loop_end):
                    g_h = _call_no_cache(self.layers[li], g_h)
            h_iter = h_iter + h_step * (g_h - h_iter)
        hidden_states = beta * natural + (1.0 - beta) * h_iter
    else:
        hidden_states = _apply_loop(
            hidden_states, run_block_once, run_one_layer,
            loop_start, loop_end, loop_mode, _gather_loop_kwargs(self),
        )

    cache_strategy = getattr(self, "_loop_cache_strategy", "last")
    if use_cache and past_key_values is not None and cache_strategy != "none":
        _stash = hidden_states if cache_strategy == "last" else x_in_for_cache
        for li in range(loop_start, loop_end):
            _stash = _call_with_cache(self.layers[li], _stash)

    for idx in range(loop_end, n_layers):
        hidden_states = _call_with_cache(self.layers[idx], hidden_states)

    hidden_states = self.norm(hidden_states)
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def patch_qwen3_with_loop(
    model,
    loop_indices: List[int],
    K: int,
    strategy: Strategy = "naive",
    ema_alpha: float = 0.5,
    cache_strategy: CacheStrategy = "last",
    momentum_beta: float = 0.3,
    anderson_m: int = 2,
    anderson_beta: float = 1.0,
    per_step_alphas: Optional[Sequence[float]] = None,
    halt_tau: Optional[float] = None,
    loop_mode: LoopMode = "block",
    decode_mode: DecodeMode = "bypass",
    decode_first_n: int = 0,
    anchor_beta: float = 0.0,
):
    """
    Patch a Qwen3{,Moe}ForCausalLM (or {Qwen3,Qwen3Moe}Model) instance so the
    contiguous block of `loop_indices` is applied K times with the chosen
    strategy. Auto-detects qwen3 vs qwen3_moe via config.model_type.

    cache_strategy controls KV-cache writes for the loop region (generation only):
      'last'  | 'first' | 'none'

    decode_mode controls whether incremental decode (seq_len=1) also goes through
    the loop:
      'bypass'  : skip loop on every decode token (default; original behavior).
      'full'    : every decode token also loops K times.
      'first_n' : only the first `decode_first_n` decode tokens after each
                  prefill go through the loop; the rest bypass.

    Strategy-specific extras:
      heavy_ball : ema_alpha (= α), momentum_beta (= β)
      anderson   : anderson_m (history depth m), anderson_beta (damping β)
      ema_sched  : per_step_alphas (length must equal K)
      halt_tau   : if set, all damped/momentum/anderson strategies break early
                   when ||Δx|| / ||x|| < halt_tau (per outer iteration).
    """
    inner = model.model if hasattr(model, "model") else model
    inner._loop_indices = list(loop_indices)
    inner._loop_K = int(K)
    inner._loop_strategy = strategy
    inner._loop_ema_alpha = float(ema_alpha)
    inner._loop_cache_strategy = cache_strategy
    inner._loop_momentum_beta = float(momentum_beta)
    inner._loop_anderson_m = int(anderson_m)
    inner._loop_anderson_beta = float(anderson_beta)
    inner._loop_anchor_beta = float(anchor_beta)
    inner._loop_per_step_alphas = (
        [float(a) for a in per_step_alphas] if per_step_alphas is not None else None
    )
    inner._loop_halt_tau = float(halt_tau) if halt_tau is not None else None
    assert loop_mode in ("block", "layer"), f"loop_mode must be 'block' or 'layer', got {loop_mode!r}"
    inner._loop_mode = loop_mode
    assert decode_mode in ("bypass", "full", "first_n"), (
        f"decode_mode must be 'bypass', 'full', or 'first_n', got {decode_mode!r}"
    )
    inner._loop_decode_mode = decode_mode
    inner._loop_decode_first_n = int(decode_first_n)
    inner._loop_decode_count = 0

    model_type = getattr(inner.config, "model_type", "qwen3")
    if model_type == "qwen3_moe":
        inner.forward = MethodType(_looped_forward_moe, inner)
    elif model_type == "qwen2_moe":
        inner.forward = MethodType(_looped_forward_qwen2moe, inner)
    else:
        inner.forward = MethodType(_looped_forward, inner)
    return model


def describe_loop(model) -> str:
    inner = model.model if hasattr(model, "model") else model
    if not hasattr(inner, "_loop_indices"):
        return "no loop patch"
    parts = [
        f"loop_indices={inner._loop_indices}",
        f"K={inner._loop_K}",
        f"strategy={inner._loop_strategy}",
        f"mode={getattr(inner, '_loop_mode', 'block')}",
        f"decode={getattr(inner, '_loop_decode_mode', 'bypass')}"
        + (f"(N={inner._loop_decode_first_n})"
           if getattr(inner, '_loop_decode_mode', 'bypass') == 'first_n' else ''),
    ]
    s = inner._loop_strategy
    if s in ("ema", "euler", "heavy_ball"):
        parts.append(f"alpha={inner._loop_ema_alpha}")
    if s == "heavy_ball":
        parts.append(f"beta={inner._loop_momentum_beta}")
    if s == "anderson":
        parts.append(f"m={inner._loop_anderson_m} beta={inner._loop_anderson_beta}")
    if s == "ema_sched":
        parts.append(f"alphas={inner._loop_per_step_alphas}")
    if inner._loop_halt_tau is not None:
        parts.append(f"halt_tau={inner._loop_halt_tau}")
    return " ".join(parts)
