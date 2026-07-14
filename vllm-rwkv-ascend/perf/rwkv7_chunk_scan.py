"""Opt-in Cube-friendly RWKV-7 DPLR chunk scan.

The implementation follows the chunkwise DPLR factorization used by FLA's
RWKV-7 operator, but is expressed with ordinary PyTorch matrix multiplies so
Ascend can route the expensive work to Cube units.  It is a performance
prototype, not the default scan backend.
"""

from __future__ import annotations

import math

import torch


def _stage_start(stage_events, name: str):
    if stage_events is None:
        return None
    event = torch.npu.Event(enable_timing=True)
    event.record()
    return event


def _stage_end(stage_events, name: str, started) -> None:
    if stage_events is None:
        return
    ended = torch.npu.Event(enable_timing=True)
    ended.record()
    stage_events.setdefault(name, []).append((started, ended))


def _as_bthn(value: torch.Tensor, heads: int, width: int) -> torch.Tensor:
    if value.ndim == 4:
        if tuple(value.shape[-2:]) != (heads, width):
            raise ValueError(
                f"expected trailing shape {(heads, width)}, got {tuple(value.shape[-2:])}"
            )
        return value
    if value.ndim == 3 and value.shape[-1] == heads * width:
        return value.view(value.shape[0], value.shape[1], heads, width)
    raise ValueError("scan vectors must be [B,T,H,N] or [B,T,H*N]")


def _cube_bmm(
    left: torch.Tensor,
    right: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Run a batched matmul in the requested device compute dtype."""

    if compute_dtype == torch.float32:
        return torch.bmm(left.float(), right.float())
    return torch.bmm(left.to(compute_dtype), right.to(compute_dtype)).float()


def _invert_unit_minus_strict_lower(
    lower: torch.Tensor,
    compute_dtype: torch.dtype,
    identity: torch.Tensor,
) -> torch.Tensor:
    """Invert ``I-L`` using a finite Cube-friendly Neumann product.

    For a CxC strict-lower matrix, ``L**C == 0`` and therefore
    ``(I-L)^-1 = (I+L)(I+L^2)(I+L^4)...``.  Power-of-two chunk sizes turn a
    slow batch of tiny triangular solves into a handful of batched matmuls.
    """

    chunk_size = int(lower.shape[-1])
    if chunk_size > 16:
        split = chunk_size // 2
        lower_11 = lower[:, :split, :split]
        lower_21 = lower[:, split:, :split]
        lower_22 = lower[:, split:, split:]
        inverse_11 = _invert_unit_minus_strict_lower(
            lower_11,
            compute_dtype,
            identity[:split, :split],
        )
        inverse_22 = _invert_unit_minus_strict_lower(
            lower_22,
            compute_dtype,
            identity[split:, split:],
        )
        inverse_21 = _cube_bmm(
            _cube_bmm(inverse_22, lower_21, compute_dtype),
            inverse_11,
            compute_dtype,
        )
        zero = torch.zeros_like(inverse_21)
        return torch.cat(
            (
                torch.cat((inverse_11, zero), dim=-1),
                torch.cat((inverse_21, inverse_22), dim=-1),
            ),
            dim=-2,
        )

    eye = identity.float().expand(lower.shape[0], -1, -1)
    power = lower.float()
    inverse = eye + power
    order = 1
    while order * 2 < chunk_size:
        power = _cube_bmm(power, power, compute_dtype)
        inverse = _cube_bmm(inverse, eye + power, compute_dtype)
        order *= 2
    return inverse


def rwkv7_chunk_scan(
    state: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    *,
    chunk_size: int = 16,
    compute_dtype: torch.dtype | None = None,
    stage_events=None,
    inverse_module=None,
    w_is_log_decay: bool = False,
    dense_chunk_prefix: bool = False,
    dense_prefix_algorithm: str = "hillis",
    inverse_backend: str = "native",
    inverse_base_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the RWKV-7 recurrence through a DPLR chunk factorization.

    ``state`` uses the repository-native VxK layout ``[B,H,N,N]``.  Token
    vectors may use either ``[B,T,H,N]`` or flattened ``[B,T,H*N]`` layout.
    The returned state is fp32 and the output follows ``r.dtype``.
    """

    if state.ndim != 4 or state.shape[-1] != state.shape[-2]:
        raise ValueError("state must be square [B,H,N,N]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch, heads, width, _ = state.shape
    w4 = _as_bthn(w, heads, width)
    tokens = int(w4.shape[1])
    if tokens % chunk_size:
        raise ValueError(
            f"token count {tokens} must be divisible by chunk_size {chunk_size}"
        )
    expected = tuple(w4.shape)
    vectors = {}
    for name, value in (("k", k), ("v", v), ("kk", kk), ("a", a), ("r", r)):
        value4 = _as_bthn(value, heads, width)
        if tuple(value4.shape) != expected:
            raise ValueError(f"{name} shape must match w")
        vectors[name] = value4
    if int(w4.shape[0]) != batch:
        raise ValueError("state and vector batch sizes must match")

    if compute_dtype is None:
        compute_dtype = r.dtype
    if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("compute_dtype must be fp16, bf16, or fp32")
    if inverse_backend not in ("native", "native_blocked"):
        raise ValueError(
            "inverse_backend must be native or native_blocked"
        )
    if inverse_base_size not in (16, 32, 64):
        raise ValueError("inverse_base_size must be 16, 32, or 64")

    stage = _stage_start(stage_events, "input_and_gates")
    chunks = tokens // chunk_size
    groups = batch * chunks * heads
    use_fused_gate_prep = (
        inverse_module is not None
        and w_is_log_decay
        and chunk_size <= 128
        and hasattr(inverse_module, "rwkv7_chunk_gate_prep")
    )

    def chunked(value: torch.Tensor) -> torch.Tensor:
        return (
            value.view(batch, chunks, chunk_size, heads, width)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

    def flat(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(groups, chunk_size, value.shape[-1])

    if use_fused_gate_prep:
        native_bf16_gate_prep = (
            compute_dtype == torch.bfloat16
            and hasattr(inverse_module, "rwkv7_chunk_gate_prep_bf16")
        )
        gate_prep = (
            inverse_module.rwkv7_chunk_gate_prep_bf16
            if native_bf16_gate_prep
            else inverse_module.rwkv7_chunk_gate_prep
        )
        (
            qg_f,
            kg_f,
            ag_f,
            bg_f,
            v_f,
            q_state_f,
            state_keys_f,
            end_decay_f,
            offset_exp_f,
        ) = gate_prep(
            w4.reshape(batch, tokens, heads * width),
            vectors["k"].reshape(batch, tokens, heads * width),
            vectors["v"].reshape(batch, tokens, heads * width),
            vectors["kk"].reshape(batch, tokens, heads * width),
            vectors["a"].reshape(batch, tokens, heads * width),
            vectors["r"].reshape(batch, tokens, heads * width),
            chunk_size,
            heads,
            width,
        )
        if compute_dtype == torch.bfloat16 and not native_bf16_gate_prep:
            qg_f = qg_f.to(compute_dtype)
            kg_f = kg_f.to(compute_dtype)
            ag_f = ag_f.to(compute_dtype)
            bg_f = bg_f.to(compute_dtype)
            v_f = v_f.to(compute_dtype)
            q_state_f = q_state_f.to(compute_dtype)
            state_keys_f = state_keys_f.to(compute_dtype)
        v_c = v_f.reshape(batch, chunks, heads, chunk_size, width)
    else:
        w_c = chunked(w4).float()
        k_c = chunked(vectors["k"])
        v_c = chunked(vectors["v"])
        kk_c = chunked(vectors["kk"])
        a_c = chunked(vectors["a"])
        q_c = chunked(vectors["r"])
        alpha_c = -kk_c
        beta_c = kk_c * a_c

        # Inclusive/exclusive log-decay within each chunk.  Centering keeps
        # low-precision operands finite without changing pairwise products.
        log_decay = (
            w_c if w_is_log_decay else w_c.clamp_min(1.0e-20).log()
        )
        inclusive = log_decay.cumsum(dim=-2)
        exclusive = inclusive - log_decay
        offset = inclusive[..., chunk_size // 2, :]
        qg = q_c.float() * torch.exp(inclusive - offset.unsqueeze(-2))
        kg = k_c.float() * torch.exp(-inclusive + offset.unsqueeze(-2))
        ag = alpha_c.float() * torch.exp(exclusive - offset.unsqueeze(-2))
        bg = beta_c.float() * torch.exp(-inclusive + offset.unsqueeze(-2))
        qg_f, kg_f, ag_f, bg_f = map(flat, (qg, kg, ag, bg))
        v_f = flat(v_c)
    _stage_end(stage_events, "input_and_gates", stage)

    stage = _stage_start(stage_events, "intra_matrices")
    positions = torch.arange(chunk_size, device=state.device)
    lower = positions[:, None] >= positions[None, :]
    strict_lower = positions[:, None] > positions[None, :]
    identity = positions[:, None] == positions[None, :]

    detail = _stage_start(stage_events, "intra_bmm")
    use_native_bf16_intra = (
        compute_dtype == torch.bfloat16
        and inverse_backend in ("native", "native_blocked")
        and inverse_module is not None
        and hasattr(inverse_module, "rwkv7_chunk_mask3_bf16")
        and hasattr(inverse_module, "rwkv7_chunk_inverse_bf16_io")
    )
    if use_native_bf16_intra:
        a_qk = torch.bmm(qg_f, kg_f.transpose(1, 2))
        a_qb = torch.bmm(qg_f, bg_f.transpose(1, 2))
        a_ak = torch.bmm(ag_f, kg_f.transpose(1, 2))
        a_ab = torch.bmm(ag_f, bg_f.transpose(1, 2))
    else:
        a_qk = _cube_bmm(qg_f, kg_f.transpose(1, 2), compute_dtype)
        a_qb = _cube_bmm(qg_f, bg_f.transpose(1, 2), compute_dtype)
        a_ak = _cube_bmm(ag_f, kg_f.transpose(1, 2), compute_dtype)
        a_ab = _cube_bmm(ag_f, bg_f.transpose(1, 2), compute_dtype)
    _stage_end(stage_events, "intra_bmm", detail)
    detail = _stage_start(stage_events, "intra_mask")
    if use_native_bf16_intra:
        a_qk, a_qb, a_ak, _ = inverse_module.rwkv7_chunk_mask3_bf16(
            a_qk.contiguous(),
            a_qb.contiguous(),
            a_ak.contiguous(),
            a_ab.contiguous(),
        )
    elif chunk_size % 8 == 0 and inverse_module is not None and hasattr(
        inverse_module, "rwkv7_chunk_mask3"
    ):
        a_qk, a_qb, a_ak, _ = inverse_module.rwkv7_chunk_mask3(
            a_qk.contiguous(),
            a_qb.contiguous(),
            a_ak.contiguous(),
            a_ab.contiguous(),
        )
    elif chunk_size % 8 == 0 and inverse_module is not None and hasattr(
        inverse_module, "rwkv7_chunk_mask"
    ):
        a_qk, a_qb, a_ak, a_ab = inverse_module.rwkv7_chunk_mask(
            a_qk.contiguous(),
            a_qb.contiguous(),
            a_ak.contiguous(),
            a_ab.contiguous(),
        )
    else:
        a_qk.masked_fill_(~lower, 0.0)
        a_qb.masked_fill_(~lower, 0.0)
        a_ak.masked_fill_(~strict_lower, 0.0)
        a_ab.masked_fill_(~strict_lower, 0.0)
    _stage_end(stage_events, "intra_mask", detail)
    _stage_end(stage_events, "intra_matrices", stage)

    stage = _stage_start(stage_events, "neumann_inverse")
    if (
        inverse_backend in ("native", "native_blocked")
        and inverse_module is not None
        and chunk_size <= 128
    ):
        if use_native_bf16_intra and inverse_backend == "native_blocked":
            def diagonal_inverses(lower: torch.Tensor):
                size = int(lower.shape[-1])
                base_size = min(inverse_base_size, size)
                block_count = size // base_size
                diagonal = torch.cat(
                    [
                        lower[
                            :,
                            block * base_size : (block + 1) * base_size,
                            block * base_size : (block + 1) * base_size,
                        ]
                        for block in range(block_count)
                    ],
                    dim=0,
                ).contiguous()
                base_inverse = (
                    inverse_module.rwkv7_chunk_inverse_bf16_io(diagonal)
                )
                groups = int(lower.shape[0])
                return (
                    base_inverse.reshape(
                        block_count, groups, base_size, base_size
                    )
                    .permute(1, 0, 2, 3)
                    .contiguous(),
                    base_size,
                    block_count,
                )

            def blocked_inverse(lower: torch.Tensor) -> torch.Tensor:
                current, base_size, _ = diagonal_inverses(lower)
                groups = int(lower.shape[0])
                span = base_size
                while current.shape[1] > 1:
                    pairs = int(current.shape[1] // 2)
                    inverse_11 = current[:, 0::2]
                    inverse_22 = current[:, 1::2]
                    lower_21 = torch.stack(
                        [
                            lower[
                                :,
                                pair * 2 * span + span :
                                (pair + 1) * 2 * span,
                                pair * 2 * span : pair * 2 * span + span,
                            ]
                            for pair in range(pairs)
                        ],
                        dim=1,
                    )
                    inverse_21 = _cube_bmm(
                        _cube_bmm(
                            inverse_22.reshape(-1, span, span),
                            lower_21.reshape(-1, span, span),
                            compute_dtype,
                        ),
                        inverse_11.reshape(-1, span, span),
                        compute_dtype,
                    ).to(compute_dtype).reshape(
                        groups, pairs, span, span
                    )
                    zero = torch.zeros_like(inverse_21)
                    current = torch.cat(
                        (
                            torch.cat((inverse_11, zero), dim=-1),
                            torch.cat((inverse_21, inverse_22), dim=-1),
                        ),
                        dim=-2,
                    )
                    span *= 2
                return current[:, 0]

            inverse = blocked_inverse(a_ab)
            inverse_fn = None
        elif use_native_bf16_intra:
            inverse_fn = inverse_module.rwkv7_chunk_inverse_bf16_io
        elif compute_dtype == torch.bfloat16 and hasattr(
            inverse_module, "rwkv7_chunk_inverse_bf16"
        ):
            inverse_fn = inverse_module.rwkv7_chunk_inverse_bf16
        else:
            inverse_fn = inverse_module.rwkv7_chunk_inverse
        if inverse_fn is not None:
            inverse = inverse_fn(a_ab.contiguous())
    else:
        inverse = _invert_unit_minus_strict_lower(
            a_ab,
            torch.float32,
            identity,
        )
    _stage_end(stage_events, "neumann_inverse", stage)
    stage = _stage_start(stage_events, "wy_u_factors")
    wy_centered = _cube_bmm(inverse, ag_f, compute_dtype)
    u = _cube_bmm(
        inverse,
        _cube_bmm(a_ak, v_f, compute_dtype),
        compute_dtype,
    )
    if use_fused_gate_prep:
        wy = wy_centered * offset_exp_f.reshape(groups, 1, width)
    else:
        wy = wy_centered * torch.exp(offset).reshape(groups, 1, width)
    if compute_dtype == torch.bfloat16:
        wy = wy.to(compute_dtype)
    _stage_end(stage_events, "wy_u_factors", stage)

    stage = _stage_start(stage_events, "factor_post")
    def unflat(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, chunks, heads, chunk_size, value.shape[-1])

    a_qk = unflat(a_qk)
    a_qb = unflat(a_qb)
    wy = unflat(wy)
    u = unflat(u)
    if use_fused_gate_prep:
        q_state = q_state_f.reshape(
            batch, chunks, heads, chunk_size, width
        )
        state_keys = state_keys_f.reshape(
            batch, chunks, heads, 2 * chunk_size, width
        )
        end_decay = end_decay_f.reshape(batch, chunks, heads, width)
    else:
        q_state = qg * torch.exp(offset).unsqueeze(-2)
        end_relative = torch.exp(
            inclusive[..., -1, :].unsqueeze(-2) - inclusive
        )
        end_decay = torch.exp(inclusive[..., -1, :])
        state_keys = torch.cat(
            (
                k_c.float() * end_relative,
                beta_c.float() * end_relative,
            ),
            dim=-2,
        )
    _stage_end(stage_events, "factor_post", stage)

    stage = _stage_start(stage_events, "chunk_apply")
    # The repository stores VxK; chunk equations use the transposed KxV state.
    current = state.float().transpose(-1, -2).contiguous()
    state_queries = torch.cat((wy, q_state), dim=-2)
    output_weights = torch.cat((a_qk, a_qb), dim=-1)
    if compute_dtype == torch.bfloat16:
        output_weights = output_weights.to(compute_dtype)
    if dense_chunk_prefix and chunks > 1:
        detail = _stage_start(stage_events, "dense_chunk_summary")
        values_zero_state = torch.cat(
            (v_c, u.to(compute_dtype)), dim=-2
        ).reshape(groups, 2 * chunk_size, width)
        state_keys_flat = state_keys.reshape(
            groups, 2 * chunk_size, width
        )
        wy_flat = wy.reshape(groups, chunk_size, width)
        q_state_flat = q_state.reshape(groups, chunk_size, width)
        output_weights_flat = output_weights.reshape(
            groups, chunk_size, 2 * chunk_size
        )
        additive = _cube_bmm(
            state_keys_flat.transpose(1, 2),
            values_zero_state,
            compute_dtype,
        )
        transition = _cube_bmm(
            state_keys_flat[:, chunk_size:].transpose(1, 2),
            wy_flat,
            compute_dtype,
        )
        transition = transition + torch.diag_embed(
            end_decay.reshape(groups, width)
        )
        output_base = _cube_bmm(
            output_weights_flat,
            values_zero_state,
            compute_dtype,
        )
        output_query = q_state_flat + _cube_bmm(
            a_qb.reshape(groups, chunk_size, chunk_size),
            wy_flat,
            compute_dtype,
        )
        if compute_dtype == torch.bfloat16:
            # Keep affine chunk summaries in Cube-native storage between
            # prefix levels.  Each BMM already rounds to BF16; retaining the
            # immediately widened FP32 tensor only doubles traffic and forces
            # another cast before the next BMM.
            transition = transition.to(compute_dtype)
            additive = additive.to(compute_dtype)
            output_base = output_base.to(compute_dtype)
            output_query = output_query.to(compute_dtype)
        _stage_end(stage_events, "dense_chunk_summary", detail)

        detail = _stage_start(stage_events, "dense_chunk_prefix")
        bh = batch * heads
        transition = (
            transition.reshape(batch, chunks, heads, width, width)
            .permute(0, 2, 1, 3, 4)
            .reshape(bh, chunks, width, width)
        )
        additive = (
            additive.reshape(batch, chunks, heads, width, width)
            .permute(0, 2, 1, 3, 4)
            .reshape(bh, chunks, width, width)
        )
        initial = current.reshape(bh, width, width)
        if dense_prefix_algorithm in ("tree", "tree_root"):
            if chunks & (chunks - 1):
                raise ValueError(
                    "tree dense prefix requires a power-of-two chunk count"
                )
            level_transitions = [transition]
            level_additives = [additive]
            while level_transitions[-1].shape[1] > 1:
                child_t = level_transitions[-1]
                child_a = level_additives[-1]
                earlier_t, later_t = child_t[:, 0::2], child_t[:, 1::2]
                earlier_a, later_a = child_a[:, 0::2], child_a[:, 1::2]
                combined = _cube_bmm(
                    later_t.reshape(-1, width, width),
                    torch.cat((earlier_t, earlier_a), dim=-1).reshape(
                        -1, width, 2 * width
                    ),
                    compute_dtype,
                ).reshape(bh, -1, width, 2 * width)
                parent_t, propagated = combined.split(width, dim=-1)
                parent_a = later_a.float() + propagated
                if compute_dtype == torch.bfloat16:
                    parent_t = parent_t.to(compute_dtype)
                    if dense_prefix_algorithm == "tree":
                        parent_a = parent_a.to(compute_dtype)
                level_transitions.append(parent_t)
                level_additives.append(parent_a)

            starts = initial[:, None]
            if compute_dtype == torch.bfloat16:
                starts = starts.to(compute_dtype)
            for child_t, child_a in zip(
                reversed(level_transitions[:-1]),
                reversed(level_additives[:-1]),
            ):
                left_t = child_t[:, 0::2]
                left_a = child_a[:, 0::2]
                right_starts = _cube_bmm(
                    left_t.reshape(-1, width, width),
                    starts.reshape(-1, width, width),
                    compute_dtype,
                ).reshape_as(starts)
                right_starts = right_starts + left_a
                if compute_dtype == torch.bfloat16:
                    right_starts = right_starts.to(compute_dtype)
                starts = torch.stack((starts, right_starts), dim=2).reshape(
                    bh, -1, width, width
                )
            if dense_prefix_algorithm == "tree_root":
                final_state = _cube_bmm(
                    level_transitions[-1][:, 0],
                    initial,
                    compute_dtype,
                ) + level_additives[-1][:, 0]
            else:
                # Keep the output starts work-efficient, but advance the
                # returned cache in chronological order.  This avoids an
                # association-only BF16 drift at long context that can flip
                # the first decode token.
                final_state = initial
                for chunk in range(chunks):
                    final_state = _cube_bmm(
                        transition[:, chunk], final_state, compute_dtype
                    ) + additive[:, chunk]
        elif dense_prefix_algorithm == "blelloch":
            if chunks & (chunks - 1):
                raise ValueError(
                    "Blelloch dense prefix requires a power-of-two chunk count"
                )

            def compose(later_t, later_a, earlier_t, earlier_a):
                combined = _cube_bmm(
                    later_t.reshape(-1, width, width),
                    torch.cat((earlier_t, earlier_a), dim=-1).reshape(
                        -1, width, 2 * width
                    ),
                    compute_dtype,
                ).reshape(*later_t.shape[:-1], 2 * width)
                product, propagated = combined.split(width, dim=-1)
                return product, later_a + propagated

            step = 2
            while step <= chunks:
                half = step // 2
                left_slice = slice(half - 1, None, step)
                right_slice = slice(step - 1, None, step)
                product, accumulated = compose(
                    transition[:, right_slice],
                    additive[:, right_slice],
                    transition[:, left_slice],
                    additive[:, left_slice],
                )
                transition[:, right_slice].copy_(product)
                additive[:, right_slice].copy_(accumulated)
                step *= 2
            total_transition = transition[:, -1].clone()
            total_additive = additive[:, -1].clone()
            transition[:, -1].copy_(
                torch.eye(
                    width,
                    device=transition.device,
                    dtype=transition.dtype,
                )
            )
            additive[:, -1].zero_()
            step = chunks
            while step >= 2:
                half = step // 2
                left_slice = slice(half - 1, None, step)
                right_slice = slice(step - 1, None, step)
                left_t = transition[:, left_slice].clone()
                left_a = additive[:, left_slice].clone()
                parent_t = transition[:, right_slice].clone()
                parent_a = additive[:, right_slice].clone()
                right_t, right_a = compose(
                    left_t, left_a, parent_t, parent_a
                )
                transition[:, left_slice].copy_(parent_t)
                additive[:, left_slice].copy_(parent_a)
                transition[:, right_slice].copy_(right_t)
                additive[:, right_slice].copy_(right_a)
                step //= 2
            starts = _cube_bmm(
                transition.reshape(-1, width, width),
                initial[:, None]
                .expand(-1, chunks, -1, -1)
                .reshape(-1, width, width),
                compute_dtype,
            ).reshape(bh, chunks, width, width)
            starts = starts + additive
            final_state = _cube_bmm(
                total_transition, initial, compute_dtype
            ) + total_additive
        else:
            offset = 1
            while offset < chunks:
                later_transition = transition[:, offset:]
                earlier = torch.cat(
                    (
                        transition[:, :-offset],
                        additive[:, :-offset],
                    ),
                    dim=-1,
                )
                composed = _cube_bmm(
                    later_transition.reshape(-1, width, width),
                    earlier.reshape(-1, width, 2 * width),
                    compute_dtype,
                ).reshape(bh, chunks - offset, width, 2 * width)
                product, propagated = composed.split(width, dim=-1)
                if compute_dtype == torch.bfloat16:
                    product = product.to(compute_dtype)
                    propagated = (
                        additive[:, offset:].float() + propagated
                    ).to(compute_dtype)
                else:
                    propagated = additive[:, offset:] + propagated
                transition = torch.cat(
                    (transition[:, :offset], product), dim=1
                )
                additive = torch.cat(
                    (
                        additive[:, :offset],
                        propagated,
                    ),
                    dim=1,
                )
                offset *= 2

            prefix_state = _cube_bmm(
                transition[:, :-1].reshape(-1, width, width),
                initial[:, None]
                .expand(-1, chunks - 1, -1, -1)
                .reshape(-1, width, width),
                compute_dtype,
            ).reshape(bh, chunks - 1, width, width)
            prefix_state = prefix_state + additive[:, :-1]
            starts = torch.cat((initial[:, None], prefix_state), dim=1)
            final_state = _cube_bmm(
                transition[:, -1], initial, compute_dtype
            ) + additive[:, -1]
        _stage_end(stage_events, "dense_chunk_prefix", detail)

        detail = _stage_start(stage_events, "dense_chunk_output")
        output_query = (
            output_query.reshape(
                batch, chunks, heads, chunk_size, width
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(bh * chunks, chunk_size, width)
        )
        output_base = (
            output_base.reshape(
                batch, chunks, heads, chunk_size, width
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(bh * chunks, chunk_size, width)
        )
        output = output_base + _cube_bmm(
            output_query,
            starts.reshape(bh * chunks, width, width),
            compute_dtype,
        )
        output = (
            output.reshape(batch, heads, chunks, chunk_size, width)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch, tokens, heads * width)
            .to(r.dtype)
        )
        current = final_state.reshape(batch, heads, width, width)
        _stage_end(stage_events, "dense_chunk_output", detail)
        _stage_end(stage_events, "chunk_apply", stage)
        return output, current.transpose(-1, -2).contiguous()

    output_chunks = []
    bh = batch * heads
    for chunk in range(chunks):
        current_f = current.reshape(bh, width, width)
        u_i = u[:, chunk].reshape(bh, chunk_size, width)
        v_i = v_c[:, chunk].reshape(bh, chunk_size, width)
        detail = _stage_start(stage_events, "chunk_state_projection")
        state_projection = _cube_bmm(
            state_queries[:, chunk].reshape(
                bh, 2 * chunk_size, width
            ),
            current_f,
            compute_dtype,
        )
        _stage_end(stage_events, "chunk_state_projection", detail)
        v2 = u_i + state_projection[:, :chunk_size]
        if compute_dtype == torch.bfloat16:
            v2 = v2.to(compute_dtype)
        values = torch.cat((v_i, v2), dim=-2)
        detail = _stage_start(stage_events, "chunk_output_bmm")
        out_i = state_projection[:, chunk_size:] + _cube_bmm(
            output_weights[:, chunk].reshape(
                bh, chunk_size, 2 * chunk_size
            ),
            values,
            compute_dtype,
        )
        _stage_end(stage_events, "chunk_output_bmm", detail)
        output_chunks.append(out_i.reshape(batch, heads, chunk_size, width))

        detail = _stage_start(stage_events, "chunk_state_update")
        state_update = _cube_bmm(
            state_keys[:, chunk]
            .reshape(bh, 2 * chunk_size, width)
            .transpose(1, 2),
            values,
            compute_dtype,
        ).reshape(batch, heads, width, width)
        current = torch.addcmul(
            state_update,
            current,
            end_decay[:, chunk].unsqueeze(-1),
        )
        _stage_end(stage_events, "chunk_state_update", detail)
    _stage_end(stage_events, "chunk_apply", stage)

    stage = _stage_start(stage_events, "output_materialize")
    output = (
        torch.stack(output_chunks, dim=1)
        .to(r.dtype)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, tokens, heads * width)
    )
    _stage_end(stage_events, "output_materialize", stage)
    return output, current.transpose(-1, -2).contiguous()


class TorchChunkScanModule:
    """Adapter matching the compiled scan extension's benchmark interface."""

    scan_row_blocks = None

    def __init__(
        self,
        *,
        chunk_size: int = 16,
        compute_dtype: torch.dtype = torch.float16,
        inverse_module=None,
        input_is_log_decay: bool = False,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.compute_dtype = compute_dtype
        self.inverse_module = inverse_module
        self.input_is_log_decay = bool(input_is_log_decay)
        self.expects_log_decay = self.input_is_log_decay
        self.stage_events = None
        self.use_dense_chunk_prefix = False
        self.dense_prefix_algorithm = "hillis"
        self.inverse_backend = "native"
        self.inverse_base_size = 32

    def rwkv7_prefill_scan(
        self,
        state: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        heads: int,
        head_size: int,
        _row_blocks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if heads != state.shape[1] or head_size != state.shape[2]:
            raise ValueError("scan metadata does not match state shape")
        effective_chunk_size = math.gcd(self.chunk_size, int(w.shape[1]))
        output, final_state = rwkv7_chunk_scan(
            state,
            w,
            k,
            v,
            kk,
            a,
            r,
            chunk_size=effective_chunk_size,
            compute_dtype=self.compute_dtype,
            stage_events=self.stage_events,
            inverse_module=self.inverse_module,
            w_is_log_decay=self.input_is_log_decay,
            dense_chunk_prefix=self.use_dense_chunk_prefix,
            dense_prefix_algorithm=self.dense_prefix_algorithm,
            inverse_backend=self.inverse_backend,
            inverse_base_size=self.inverse_base_size,
        )
        state.copy_(final_state)
        return output, state


__all__ = ["TorchChunkScanModule", "rwkv7_chunk_scan"]
