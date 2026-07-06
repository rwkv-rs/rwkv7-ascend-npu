"""NPUGraph decode acceleration for RWKV7-Ascend (B=1 single-sequence fast path).

The C++ op-coalesced forward (`rwkv7_decode_full`) still issues ~960 CANN kernel
launches per step; on the 910B3 each carries ~17us of host dispatch overhead, so a
single-sequence decode step costs ~16ms (~60 tok/s) — purely dispatch-bound, not
compute (at::linear latency is identical at B=1 and B=128).

`torch.npu.NPUGraph` records the whole step into one device-side graph; `replay()`
re-runs it with a single host launch, collapsing the dispatch overhead. Measured on
the 910B3 (CANN 8.5.0): 0.1B B=1 16.3ms -> 2.8ms (**5.9x**, 60 -> 358 tok/s, matching
an A100's CUDA-graph decode); 1.5B B=1 33ms -> 8.9ms (3.7x, 30 -> 114 tok/s). Bit-exact
vs eager (single-step maxabs=0; multi-step greedy tokens identical).

Design — dedicated fixed-address buffers decoupled from the SlottedScheduler's batch
state. Each `decode()` copies the active slot's state into these buffers, replays the
graph (which evolves the state in place), and copies the evolved state back. The
scheduler stays the source of truth, so mid-flight joins/leaves (B=1 -> B>1 -> B=1)
are safe. Copy cost is ~13MB @ ~1.5TB/s ≈ 8us per step — negligible next to the ~14ms
saved. B>1 falls back to the eager batched forward (not graph-captured).
"""
import torch
import torch_npu  # noqa: F401  (registers npu; required for torch.npu.graph)


class NpuGraphDecoder:
    """Captures one B=1 decode step as an NPUGraph and replays it per token.

    Built lazily from a loaded `RWKV7Engine` (its C++ `mod`, weights `W`, embeddings).
    Only the single-sequence (B=1) decode step is accelerated; multi-sequence steps
    use the engine's eager batched forward.
    """

    def __init__(self, eng):
        self.eng = eng
        L, H, N, hd = eng.L, eng.H, eng.N, eng.hidden
        self.dev = eng.lm_w_m.device
        # dedicated B=1 buffers — fixed addresses, captured into the graph
        self.sa = torch.zeros(L, 1, H, N, N, dtype=torch.float32, device=self.dev)
        self.xp = torch.zeros(L, 1, hd, dtype=torch.float16, device=self.dev)
        self.xf = torch.zeros(L, 1, hd, dtype=torch.float16, device=self.dev)
        self.vf = torch.zeros(1, hd, dtype=torch.float16, device=self.dev)
        self.emb = torch.zeros(1, hd, dtype=torch.float16, device=self.dev)
        self.logits = None
        self.graph = None

    def _fwd(self):
        eng = self.eng
        return eng.mod.rwkv7_decode_full(
            self.emb, *eng.W, self.sa, self.xp, self.xf, self.vf,
            eng.H, eng.N, eng.lm_w_m, eng.fnorm_w, eng.fnorm_b)

    def capture(self, warmup=5):
        """Warm up + capture the decode step into an NPUGraph. Call once after the
        engine + C++ module are loaded."""
        with torch.no_grad():
            for _ in range(warmup):
                self.logits = self._fwd()
        torch.npu.synchronize()
        # side-stream warmup is required for a clean capture
        s = torch.npu.Stream(); s.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(s):
            for _ in range(warmup):
                self.logits = self._fwd()
        torch.npu.current_stream().wait_stream(s)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self.logits = self._fwd()
        torch.npu.synchronize()

    def decode(self, token_id, sa_slot, xp_slot, xf_slot, vf_slot):
        """One decode step via graph replay.

        Copies the scheduler's active-slot state into the dedicated buffers, sets the
        input embedding, replays the graph (evolving the dedicated state in place),
        copies the evolved state back to the scheduler slots, and returns the logits
        (a view of the captured output buffer — read before the next replay).
        """
        # scheduler slot -> dedicated buffer
        self.sa[:, 0:1].copy_(sa_slot)
        self.xp[:, 0:1].copy_(xp_slot)
        self.xf[:, 0:1].copy_(xf_slot)
        self.vf[0:1].copy_(vf_slot)
        self.emb.copy_(self.eng.base.embeddings(torch.tensor([token_id], device=self.dev)))
        self.graph.replay()
        # evolved dedicated buffer -> scheduler slot (scheduler stays authoritative)
        sa_slot.copy_(self.sa[:, 0:1])
        xp_slot.copy_(self.xp[:, 0:1])
        xf_slot.copy_(self.xf[:, 0:1])
        vf_slot.copy_(self.vf[0:1])
        return self.logits
