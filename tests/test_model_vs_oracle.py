#!/usr/bin/env python3
# coding=utf-8
"""M1c gate: ascend_port.model greedy decode must match the numpy oracle
(reference/hakureirm/oracle_numpy.py) token-for-token on the 0.1B.

Reads the oracle fixture (prompt_tokens + greedy_tokens + arch) produced by
``oracle_numpy.py --dump-fixture``, loads the same .pth, decodes on the NPU,
and compares. Greedy must be EXACT.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from ascend_port.model import greedy_generate, load_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/rwkv7-models/rwkv7-g1d-0.1b.pth")
    ap.add_argument("--fixture", default="/data/rwkv7-models/oracle_fixture_0.1b.json")
    args = ap.parse_args()

    device = "npu" if torch.npu.is_available() else "cpu"
    if device == "npu":
        import torch_npu  # noqa: F401
        print(f"device=npu:{torch.npu.get_device_name(0)}")
    else:
        print("device=cpu (NPU not available)")

    fx = json.load(open(args.fixture))
    arch = fx["arch"]
    n_layer, n_head, head_size = arch["n_layer"], arch["n_head"], arch["head_size"]
    print(f"arch: n_layer={n_layer} n_head={n_head} head_size={head_size} vocab={arch['vocab']}")
    print(f"prompt ({len(fx['prompt_tokens'])} tok): {fx['prompt']}")

    params, _, _ = load_params(args.model, device)
    n_new = len(fx["greedy_tokens"])
    gen = greedy_generate(params, fx["prompt_tokens"], n_new, n_layer, n_head, head_size, device)
    ref = fx["greedy_tokens"]

    match = sum(int(a == b) for a, b in zip(gen, ref))
    print(f"mine:   {gen}")
    print(f"oracle: {ref}")
    print(f"match: {match}/{len(ref)}")
    if gen == ref:
        print("GATE: PASS  (ascend_port.model is token-exact vs the numpy oracle on 0.1B)")
        sys.exit(0)
    for i, (a, b) in enumerate(zip(gen, ref)):
        if a != b:
            print(f"first divergence at idx {i}: mine={a} oracle={b}")
            break
    print("GATE: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
