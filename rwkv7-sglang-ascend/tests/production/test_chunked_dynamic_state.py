from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sglang_rwkv7_ascend.kernels.wkv import wkv_recurrent
from sglang_rwkv7_ascend.state_cache import SGLangMambaPoolStateAdapter


def _inputs(length, seed, h=2, d=4):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(1, length, h, d, generator=g)
    w = -torch.rand(1, length, h, d, generator=g)
    k = torch.randn(1, length, h, d, generator=g)
    v = torch.randn(1, length, h, d, generator=g)
    kk = torch.nn.functional.normalize(k, dim=-1)
    a = torch.sigmoid(torch.randn(1, length, h, d, generator=g))
    return r, w, k, v, kk, a


def _run_chunks(args, chunks, state=None):
    outs = []
    pos = 0
    for width in chunks:
        part = tuple(t[:, pos : pos + width] for t in args)
        out, state = wkv_recurrent(
            *part,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
        )
        outs.append(out)
        pos += width
    assert pos == args[0].shape[1]
    return torch.cat(outs, dim=1), state


@pytest.mark.parametrize("chunks", [[17], [1] * 17, [3, 5, 2, 7], [8, 1, 8]])
def test_irregular_chunked_prefill_is_monolithic(chunks):
    args = _inputs(17, 123)
    expected_out, expected_state = _run_chunks(args, [17])
    actual_out, actual_state = _run_chunks(args, chunks)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-6, atol=1e-6)


def test_packed_varlen_matches_independent_sequences():
    seqs = [_inputs(3, 1), _inputs(7, 2), _inputs(2, 3)]
    packed = tuple(torch.cat([s[i] for s in seqs], dim=1) for i in range(6))
    cu = torch.tensor([0, 3, 10, 12], dtype=torch.int32)
    out, state = wkv_recurrent(
        *packed,
        scale=1.0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    for n, seq in enumerate(seqs):
        ref_out, ref_state = wkv_recurrent(
            *seq, scale=1.0, output_final_state=True
        )
        bos, eos = int(cu[n]), int(cu[n + 1])
        torch.testing.assert_close(out[:, bos:eos], ref_out, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(state[n : n + 1], ref_state, rtol=1e-6, atol=1e-6)


@dataclass(frozen=True)
class _State:
    conv: list[torch.Tensor]
    temporal: torch.Tensor


class _Pool:
    def __init__(self):
        self.mamba_cache = _State(
            conv=[torch.zeros(2, 6, 1, 8), torch.zeros(2, 6, 1, 8)],
            temporal=torch.zeros(2, 6, 2, 4, 4),
        )

    def clear_slots(self, idx):
        for t in [*self.mamba_cache.conv, self.mamba_cache.temporal]:
            t[:, idx] = 0

    def copy_from(self, src, dst):
        # clone first so swaps/reorders cannot alias writes.
        for t in [*self.mamba_cache.conv, self.mamba_cache.temporal]:
            t[:, dst] = t[:, src].clone()


def test_slot_reorder_cancel_reuse_and_offload_restore():
    pool = _Pool()
    adapter = SGLangMambaPoolStateAdapter(pool)
    tensors = [*pool.mamba_cache.conv, pool.mamba_cache.temporal]
    for i, t in enumerate(tensors):
        t[:, 1] = i + 1
        t[:, 2] = (i + 1) * 10

    snap = adapter.offload([1, 2])
    adapter.reorder([2, 1], [3, 4])
    for i, t in enumerate(tensors):
        assert torch.all(t[:, 3] == (i + 1) * 10)
        assert torch.all(t[:, 4] == i + 1)

    # Cancellation must erase the state before slot reuse.
    adapter.clear([1, 2])
    for t in tensors:
        assert torch.count_nonzero(t[:, [1, 2]]) == 0

    adapter.restore([1, 2], snap)
    for i, t in enumerate(tensors):
        assert torch.all(t[:, 1] == i + 1)
        assert torch.all(t[:, 2] == (i + 1) * 10)


def test_reserved_padding_slot_is_never_user_state():
    adapter = SGLangMambaPoolStateAdapter(_Pool())
    with pytest.raises(ValueError, match="reserved"):
        adapter.clear([0])
