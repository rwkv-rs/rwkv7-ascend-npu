from types import SimpleNamespace

import torch

from rwkv7_hf import ascend_graph_runtime as graph_runtime


def test_graph_availability_is_capability_based(monkeypatch):
    monkeypatch.setattr(torch, "npu", None, raising=False)
    assert not graph_runtime.ascend_graph_available()

    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        NPUGraph=object,
        graph=lambda graph: graph,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    assert graph_runtime.ascend_graph_available()


def test_graph_cache_size_is_bounded_and_fail_safe(monkeypatch):
    monkeypatch.delenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", raising=False)
    assert graph_runtime.ascend_graph_cache_size() == 3

    monkeypatch.setenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "0")
    assert graph_runtime.ascend_graph_cache_size() == 1

    monkeypatch.setenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "not-an-integer")
    assert graph_runtime.ascend_graph_cache_size() == 3


def test_graph_runtime_signature_tracks_only_ascend_graph_and_quant(monkeypatch):
    monkeypatch.setenv("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "4")
    monkeypatch.setenv("RWKV7_ASCEND_QUANT_POLICY", "candidate")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH", "1")

    assert graph_runtime.ascend_graph_runtime_signature() == (
        ("RWKV7_ASCEND_GRAPH_CACHE_SIZE", "4"),
        ("RWKV7_ASCEND_QUANT_POLICY", "candidate"),
    )


def test_quant_buffer_replacement_changes_graph_module_signature():
    class PackedProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bit = 8
            self.group_size = 0
            self.register_buffer("qweight", torch.ones(2, 3, dtype=torch.int8))
            self.register_buffer("scales", torch.ones(3, dtype=torch.float16))

    layers = torch.nn.ModuleList([PackedProjection()])
    owner = SimpleNamespace(model=SimpleNamespace(layers=layers))

    before = graph_runtime.ascend_graph_module_signature(owner)
    layers[0].qweight = layers[0].qweight.clone()
    after = graph_runtime.ascend_graph_module_signature(owner)

    assert before
    assert before != after
