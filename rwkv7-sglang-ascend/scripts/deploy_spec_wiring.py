#!/usr/bin/env python3
# coding=utf-8
"""Idempotently wire the RWKV-7 chain spec-decode worker into sglang's
SpeculativeAlgorithm (v0.5.14 create_worker). Run after copying
rwkv_chain_worker.py into sglang/srt/speculative/.

Patches spec_info.py: (1) RWKV_CHAIN enum value, (2) is_rwkv_chain() predicate,
(3) create_worker branch returning RwkvChainWorker. The worker itself is
device-agnostic Python; the Ascend backend's recurrence extend already commits
final_state into the MambaPool (the contract the worker's snapshot/restore relies
on), so no backend change is needed.
"""
SI = "/data/sglang/python/sglang/srt/speculative/spec_info.py"
with open(SI) as f:
    s = f.read()
log = []

# 1. RWKV_CHAIN enum value (after STANDALONE).
if "RWKV_CHAIN = auto()" not in s:
    s = s.replace("    STANDALONE = auto()\n",
                  "    STANDALONE = auto()\n    RWKV_CHAIN = auto()\n", 1)
    log.append("enum: added RWKV_CHAIN")
else:
    log.append("enum: skip")

# 2. is_rwkv_chain() predicate (before is_ngram).
if "def is_rwkv_chain" not in s:
    s = s.replace(
        "    def is_ngram(self) -> bool:\n",
        "    def is_rwkv_chain(self) -> bool:\n"
        "        return self == SpeculativeAlgorithm.RWKV_CHAIN\n\n"
        "    def is_ngram(self) -> bool:\n",
        1,
    )
    log.append("predicate: added is_rwkv_chain")
else:
    log.append("predicate: skip")

# 3. create_worker branch returning RwkvChainWorker (before the Unreachable raise).
CREATE_BRANCH = (
    '        elif self.is_rwkv_chain():\n'
    '            from sglang.srt.speculative.rwkv_chain_worker import RwkvChainWorker\n'
    '            return RwkvChainWorker\n\n'
)
if "is_rwkv_chain():" not in s.split("def create_worker")[1]:
    anchor = '        raise ValueError("Unreachable code path in create_worker.")'
    if anchor in s:
        s = s.replace(anchor, CREATE_BRANCH + anchor, 1)
        log.append("create_worker: added rwkv_chain branch")
    else:
        log.append("create_worker: MISS (anchor not found)")
else:
    log.append("create_worker: skip")

with open(SI, "w") as f:
    f.write(s)
print("=== spec-decode wiring ===")
for line in log:
    print(" ", line)
