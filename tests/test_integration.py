"""NPU integration tests — run on 910B3 (auto-skipped in CI: no NPU there).

Covers: greedy bit-exactness vs HF-native, batched == single, scheduler greedy
matches standalone, scheduler stop-string truncation, sampler variety.
"""
import os
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
import pytest
import torch
from serve_engine import RWKV7Engine
from serve_full import SlottedScheduler, Seq, SamplerCfg
from rwkv7_hf.tokenization_rwkv7 import _RWKVTrie

pytestmark = pytest.mark.npu

MODEL = os.environ.get("RWKV7_TEST_MODEL", "/root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf")
VOCAB_FILE = os.environ.get("RWKV7_VOCAB", "/root/rwkv7-ascend/assets/rwkv_vocab_v20230424.txt")
H, N, L = 12, 64, 12
EXPECTED_NEXT8 = [16, 17, 18, 21, 18, 21, 18, 21]  # verified HF-native greedy


@pytest.fixture(scope="module")
def eng():
    return RWKV7Engine(MODEL, H, N, L)


@pytest.fixture(scope="module")
def tok():
    return _RWKVTrie(VOCAB_FILE)


class _Fut:  # sync stand-in for asyncio.Future, so we can drive the scheduler without a loop
    def __init__(self): self.result = None; self._done = False
    def set_result(self, r): self.result = r; self._done = True
    def set_exception(self, e): self.result = ("error", str(e)); self._done = True
    def done(self): return self._done


def _run(eng, tok, prompt_ids, max_new, cfg, stop=None):
    sch = SlottedScheduler(eng, tok)
    fut = _Fut()
    sch.add(Seq(prompt_ids, max_new, cfg, stop or [], False, fut))
    while sch.B > 0:
        sch.step()
    return fut.result


def test_greedy_bitexact(eng):
    gen = eng.generate([list(range(16))], max_new=8)[0]
    assert gen == EXPECTED_NEXT8


def test_batched_matches_single(eng):
    prompts = [list(range(16)), list(range(8)), list(range(4))]
    batch = eng.generate(prompts, max_new=8)
    single = [eng.generate([p], max_new=8)[0] for p in prompts]
    assert batch == single


def test_scheduler_greedy_matches_standalone(eng, tok):
    sched_text = _run(eng, tok, list(range(16)), 8, SamplerCfg())
    ref_text = tok.decode(eng.generate([list(range(16))], max_new=8)[0])
    assert sched_text == ref_text


def test_scheduler_stop_truncates(eng, tok):
    ref_text = tok.decode(eng.generate([list(range(16))], max_new=20)[0])
    stop = ref_text[:2] if len(ref_text) >= 2 else ref_text
    stopped = _run(eng, tok, list(range(16)), 20, SamplerCfg(), [stop])
    assert stop not in stopped, "stop string leaked into output"


def test_scheduler_sampler_varies(eng, tok):
    torch.manual_seed(0)
    a = _run(eng, tok, list(range(16)), 10, SamplerCfg(0.8, 40, 1.0))
    torch.manual_seed(1)
    b = _run(eng, tok, list(range(16)), 10, SamplerCfg(0.8, 40, 1.0))
    assert a != b or a != tok.decode(eng.generate([list(range(16))], max_new=10)[0])
