"""CPU-only state-machine tests for graph-resident greedy token reuse."""
import ast
import asyncio
import os

import torch

from sampler import SamplerCfg, sample_rows


def _load_scheduler_classes():
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "serving", "serve_full.py"
    )
    source = open(source_path, "r", encoding="utf-8", errors="replace").read()
    tree = ast.parse(source, filename=source_path)
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef) and node.name == "prefill_one"
        )
        or (
            isinstance(node, ast.ClassDef)
            and node.name in {"Seq", "SlottedScheduler"}
        )
    ]
    namespace = {
        "asyncio": asyncio,
        "DEV": "cpu",
        "sample_rows": sample_rows,
        "torch": torch,
        "VOCAB": 32,
    }
    exec(compile(ast.Module(selected, type_ignores=[]), source_path, "exec"), namespace)
    return namespace["Seq"], namespace["SlottedScheduler"]


Seq, SlottedScheduler = _load_scheduler_classes()


class _Embeddings:
    def __call__(self, tokens):
        return tokens.reshape(-1, 1)


class _Mod:
    def rwkv7_decode_full(self, embeddings, *args):
        logits = torch.full((embeddings.shape[0], 32), -1000.0)
        next_tokens = (embeddings.reshape(-1).long() + 1) % 32
        logits.scatter_(1, next_tokens[:, None], 1.0)
        return logits


class _Engine:
    L = H = N = hidden = 1
    base = type("Base", (), {"embeddings": _Embeddings()})()
    mod = _Mod()
    W = ()
    lm_w_m = fnorm_w = fnorm_b = None


class _Tokenizer:
    def decode(self, tokens):
        return "".join(chr(65 + token % 26) for token in tokens)


class _GraphDecoder:
    capture_greedy_token = True

    def __init__(self):
        self.current_token = None
        self.greedy_calls = []
        self.decode_calls = []

    @staticmethod
    def _logits(token):
        logits = torch.full((1, 32), -1000.0)
        logits[0, token % 32] = 1.0
        return logits

    def decode_greedy(self, token, *state, reuse_token=False):
        self.greedy_calls.append((token, reuse_token))
        if not reuse_token:
            self.current_token = token
        self.current_token = (self.current_token + 1) % 32
        return self._logits(self.current_token), self.current_token

    def decode(self, token, *state):
        self.decode_calls.append(token)
        return self._logits((token + 1) % 32)


def _sequence(next_token, *, temperature=0.0):
    seq = Seq([], 20, SamplerCfg(temperature), [], False, None)
    seq.next = next_token
    seq.gen = [next_token]
    return seq


def _set_batch(scheduler, sequences):
    batch = len(sequences)
    scheduler.seqs = list(sequences)
    scheduler.B = batch
    scheduler.sa = torch.zeros(1, batch, 1, 1, 1)
    scheduler.xp = torch.zeros(1, batch, 1)
    scheduler.xf = torch.zeros(1, batch, 1)
    scheduler.vf = torch.zeros(batch, 1)


def test_greedy_token_reuse_resets_across_batch_transition():
    graph = _GraphDecoder()
    scheduler = SlottedScheduler(_Engine(), _Tokenizer(), graph)
    first = _sequence(10)
    _set_batch(scheduler, [first])

    scheduler.step()
    scheduler.step()
    assert graph.greedy_calls == [(10, False), (None, True)]
    assert first.next == 12

    second = _sequence(20)
    _set_batch(scheduler, [first, second])
    scheduler.step()
    assert scheduler._graph_token_seq is None
    assert first.next == 13

    scheduler._shrink(1)
    scheduler.step()
    assert graph.greedy_calls[-1] == (13, False)
    assert first.next == 14


def test_stochastic_sampling_never_reuses_greedy_graph_token():
    graph = _GraphDecoder()
    scheduler = SlottedScheduler(_Engine(), _Tokenizer(), graph)
    stochastic = _sequence(7, temperature=0.8)
    _set_batch(scheduler, [stochastic])

    scheduler.step()
    assert graph.greedy_calls == []
    assert graph.decode_calls == [7]
    assert scheduler._graph_token_seq is None


def test_graph_decoder_selects_the_engine_npu_before_capture():
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "serving", "graph_decode.py"
    )
    source = open(source_path, "r", encoding="utf-8", errors="replace").read()
    assert "self.dev = eng.lm_w_m.device" in source
    assert "torch.npu.set_device(self.dev)" in source
