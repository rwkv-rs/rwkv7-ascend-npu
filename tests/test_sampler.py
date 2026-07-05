"""Sampler unit tests — CI-runnable (pure torch, no NPU, no rwkv7_hf)."""
import torch
from sampler import SamplerCfg, sample_rows

VOCAB = 100


def test_greedy_fastpath_single():
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
    assert sample_rows(logits, [SamplerCfg()]) == [1]  # argmax


def test_greedy_batch():
    logits = torch.tensor([[3.0, 1.0], [1.0, 9.0]])
    assert sample_rows(logits, [SamplerCfg(), SamplerCfg()]) == [0, 1]


def test_temperature_returns_valid_token():
    torch.manual_seed(0)
    logits = torch.zeros(1, VOCAB); logits[0, 7] = 10.0
    tok = sample_rows(logits, [SamplerCfg(temperature=1.0)])[0]
    assert 0 <= tok < VOCAB


def test_top_k_restricts_to_argmax():
    torch.manual_seed(0)
    logits = torch.zeros(1, VOCAB); logits[0, 5] = 10.0
    for _ in range(10):
        assert sample_rows(logits, [SamplerCfg(temperature=1.0, top_k=1)])[0] == 5


def test_top_p_restricts_to_argmax():
    torch.manual_seed(0)
    logits = torch.zeros(1, VOCAB); logits[0, 3] = 10.0
    for _ in range(10):
        assert sample_rows(logits, [SamplerCfg(temperature=1.0, top_p=0.01)])[0] == 3


def test_mixed_greedy_and_sample():
    torch.manual_seed(0)
    logits = torch.zeros(2, VOCAB); logits[0, 2] = 10.0; logits[1, 9] = 10.0
    cfgs = [SamplerCfg(), SamplerCfg(temperature=1.0)]  # row 0 greedy, row 1 sampled
    out = sample_rows(logits, cfgs)
    assert out[0] == 2  # greedy row -> argmax
    assert 0 <= out[1] < VOCAB


def test_greedy_ties_returns_first():
    # all-equal logits -> argmax is index 0
    logits = torch.zeros(1, 8)
    assert sample_rows(logits, [SamplerCfg()])[0] == 0


def test_top_k_zero_returns_valid():
    # top_k=0 means no top-k filtering; still returns a valid token
    torch.manual_seed(0)
    logits = torch.zeros(1, VOCAB); logits[0, 4] = 10.0
    tok = sample_rows(logits, [SamplerCfg(temperature=1.0, top_k=0)])[0]
    assert 0 <= tok < VOCAB


def test_top_p_one_returns_valid():
    # top_p=1.0 means no nucleus filtering
    torch.manual_seed(0)
    logits = torch.zeros(1, VOCAB); logits[0, 4] = 10.0
    tok = sample_rows(logits, [SamplerCfg(temperature=1.0, top_p=1.0)])[0]
    assert 0 <= tok < VOCAB


def test_single_token_vocab():
    # vocab=1 -> only token 0 is possible, both greedy and sampled
    logits = torch.zeros(2, 1)
    assert sample_rows(logits, [SamplerCfg(temperature=1.0), SamplerCfg()]) == [0, 0]


def test_top_k_larger_than_vocab():
    # top_k >= vocab -> effectively no filtering, returns valid token
    torch.manual_seed(0)
    logits = torch.zeros(1, 8); logits[0, 5] = 10.0
    tok = sample_rows(logits, [SamplerCfg(temperature=1.0, top_k=999)])[0]
    assert 0 <= tok < 8

