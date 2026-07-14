from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PERF_DIR = Path(__file__).resolve().parents[1] / "perf"
sys.path.insert(0, str(PERF_DIR))

from rwkv7_chunk_scan import (  # noqa: E402
    TorchChunkScanModule,
    _invert_unit_minus_strict_lower,
    rwkv7_chunk_scan,
)


def _recurrent_reference(state, w, k, v, kk, a, r):
    batch, tokens, heads, width = w.shape
    current = state.float().clone()
    rows = []
    for token in range(tokens):
        projection = torch.matmul(
            current,
            (-kk[:, token]).unsqueeze(-1).float(),
        )
        current = (
            current * w[:, token].unsqueeze(-2).float()
            + projection * (kk[:, token] * a[:, token]).unsqueeze(-2).float()
            + v[:, token].unsqueeze(-1).float()
            * k[:, token].unsqueeze(-2).float()
        )
        rows.append(
            torch.matmul(
                current,
                r[:, token].unsqueeze(-1).float(),
            ).squeeze(-1)
        )
    return torch.stack(rows, dim=1).reshape(batch, tokens, heads * width), current


def test_chunk_scan_matches_recurrent_reference_with_nonzero_state():
    torch.manual_seed(17)
    batch, tokens, heads, width = 2, 8, 3, 4
    shape = (batch, tokens, heads, width)
    state = torch.randn(batch, heads, width, width) * 0.05
    w = torch.sigmoid(torch.randn(shape)) * 0.2 + 0.75
    k = torch.randn(shape) * 0.1
    v = torch.randn(shape) * 0.1
    kk = torch.nn.functional.normalize(torch.randn(shape), dim=-1)
    a = torch.sigmoid(torch.randn(shape))
    r = torch.randn(shape) * 0.1

    expected_out, expected_state = _recurrent_reference(
        state, w, k, v, kk, a, r
    )
    actual_out, actual_state = rwkv7_chunk_scan(
        state,
        w,
        k,
        v,
        kk,
        a,
        r,
        chunk_size=4,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)


def test_chunk_scan_rejects_nondivisible_prompt():
    shape = (1, 6, 1, 4)
    vectors = [torch.ones(shape) for _ in range(6)]
    state = torch.zeros(1, 1, 4, 4)
    try:
        rwkv7_chunk_scan(state, *vectors, chunk_size=4)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected a non-divisible prompt to be rejected")


def test_chunk_scan_module_updates_extension_style_state_in_place():
    shape = (1, 4, 1, 2)
    state = torch.zeros(1, 1, 2, 2)
    w = torch.full(shape, 0.9)
    k = torch.full(shape, 0.1)
    v = torch.full(shape, 0.2)
    kk = torch.nn.functional.normalize(torch.ones(shape), dim=-1)
    a = torch.full(shape, 0.5)
    r = torch.full(shape, 0.1)
    module = TorchChunkScanModule(chunk_size=2, compute_dtype=torch.float32)

    output, returned_state = module.rwkv7_prefill_scan(
        state, w, k, v, kk, a, r, 1, 2, None
    )

    assert output.shape == (1, 4, 2)
    assert returned_state.data_ptr() == state.data_ptr()
    assert torch.count_nonzero(state) > 0


def test_block_unit_lower_inverse_matches_triangular_solve():
    torch.manual_seed(23)
    size = 64
    lower = torch.tril(torch.randn(3, size, size) * 0.01, diagonal=-1)
    identity = torch.eye(size)

    actual = _invert_unit_minus_strict_lower(
        lower, torch.float32, identity
    )
    expected = torch.linalg.solve_triangular(
        identity.expand(3, -1, -1) - lower,
        identity.expand(3, -1, -1),
        upper=False,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_chunk_scan_can_route_inverse_through_compiled_module_contract():
    class FakeInverseModule:
        calls = 0

        def rwkv7_chunk_inverse(self, lower):
            self.calls += 1
            size = lower.shape[-1]
            eye = torch.eye(size).expand(lower.shape[0], -1, -1)
            return torch.linalg.solve_triangular(
                eye - lower, eye, upper=False
            )

    shape = (1, 4, 1, 2)
    state = torch.zeros(1, 1, 2, 2)
    w = torch.full(shape, 0.9)
    k = torch.full(shape, 0.1)
    v = torch.full(shape, 0.2)
    kk = torch.nn.functional.normalize(torch.ones(shape), dim=-1)
    a = torch.full(shape, 0.5)
    r = torch.full(shape, 0.1)
    inverse_module = FakeInverseModule()

    rwkv7_chunk_scan(
        state,
        w,
        k,
        v,
        kk,
        a,
        r,
        chunk_size=4,
        compute_dtype=torch.float32,
        inverse_module=inverse_module,
    )

    assert inverse_module.calls == 1


def test_log_decay_input_matches_decay_input():
    torch.manual_seed(29)
    shape = (1, 4, 1, 4)
    state = torch.randn(1, 1, 4, 4) * 0.01
    log_decay = -torch.rand(shape) * 0.2
    w = log_decay.exp()
    vectors = [torch.randn(shape) * 0.05 for _ in range(5)]

    expected = rwkv7_chunk_scan(
        state, w, *vectors, chunk_size=4, compute_dtype=torch.float32
    )
    actual = rwkv7_chunk_scan(
        state,
        log_decay,
        *vectors,
        chunk_size=4,
        compute_dtype=torch.float32,
        w_is_log_decay=True,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("algorithm", ["tree", "tree_root"])
def test_tree_dense_prefix_matches_sequential_chunk_application(algorithm):
    torch.manual_seed(31)
    shape = (1, 16, 2, 4)
    state = torch.randn(1, 2, 4, 4) * 0.02
    w = torch.sigmoid(torch.randn(shape)) * 0.2 + 0.75
    vectors = [torch.randn(shape) * 0.05 for _ in range(5)]

    expected = rwkv7_chunk_scan(
        state,
        w,
        *vectors,
        chunk_size=4,
        compute_dtype=torch.float32,
    )
    actual = rwkv7_chunk_scan(
        state,
        w,
        *vectors,
        chunk_size=4,
        compute_dtype=torch.float32,
        dense_chunk_prefix=True,
        dense_prefix_algorithm=algorithm,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-6)
