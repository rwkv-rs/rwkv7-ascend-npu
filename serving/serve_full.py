"""RWKV7-Ascend full serving server.

Single-threaded asyncio loop (torch_npu is main-thread-only):
  - SlottedScheduler: persistent batched recurrent state, no per-step cat
  - Sampler: greedy (batched argmax fast-path) / temperature / top_k / top_p
  - Streaming + stop-string support (buffered to avoid over-emitting partial stops)
  - FastAPI /v1/completions (OpenAI-style, JSON or SSE) + /health
"""
import os
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1"); os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import sys; sys.path.insert(0, "/root/rwkv7-ascend")
import json, asyncio
import torch, torch_npu
from serve_engine import RWKV7Engine, DEV, VOCAB
from rwkv7_hf.tokenization_rwkv7 import _RWKVTrie


# ---------------- sampler ----------------
class SamplerCfg:
    def __init__(self, temperature=0.0, top_k=0, top_p=1.0):
        self.temperature = temperature; self.top_k = top_k; self.top_p = top_p


def sample_rows(logits, cfgs):
    if all(c.temperature <= 0 for c in cfgs):
        return logits.argmax(-1).tolist()  # batched greedy fast-path
    out = []
    for i, cfg in enumerate(cfgs):
        row = logits[i].float()
        if cfg.temperature <= 0:
            out.append(int(row.argmax())); continue
        row = row / cfg.temperature
        if cfg.top_k > 0:
            k = min(cfg.top_k, row.shape[-1]); vals, idx = torch.topk(row, k)
            mask = torch.full_like(row, float("-inf")); mask[idx] = vals; row = mask
        if cfg.top_p < 1.0:
            sval, sidx = torch.sort(row, descending=True)
            cum = torch.softmax(sval, dim=-1).cumsum(dim=-1); keep = cum <= cfg.top_p; keep[0] = True
            row = torch.full_like(row, float("-inf")); row[sidx[keep]] = sval[keep]
        probs = torch.softmax(row, dim=-1)
        try:
            tok = int(torch.multinomial(probs, 1))
        except Exception:
            tok = int(probs.argmax())
        out.append(tok)
    return out


def prefill_one(eng, tokens):
    L, H, N = eng.L, eng.H, eng.N; hd = eng.hidden
    sa = torch.zeros(L, 1, H, N, N, dtype=torch.float32, device=DEV)
    xp = torch.zeros(L, 1, hd, dtype=torch.float16, device=DEV)
    xf = torch.zeros(L, 1, hd, dtype=torch.float16, device=DEV)
    vf = torch.zeros(1, hd, dtype=torch.float16, device=DEV)
    out = None
    for t in (tokens or [0]):
        emb = eng.base.embeddings(torch.tensor([t], device=DEV))
        out = eng.mod.rwkv7_decode_full(emb, *eng.W, sa, xp, xf, vf, eng.H, eng.N, eng.lm_w_m, eng.fnorm_w, eng.fnorm_b)
    first = int(out.reshape(-1, VOCAB).argmax(-1)[-1])
    return (sa, xp, xf, vf), first


# ---------------- slotted scheduler ----------------
class Seq:
    def __init__(self, prompt_ids, max_new, cfg, stop_strings, is_stream, fut):
        self.prompt_ids = prompt_ids; self.max_new = max_new; self.cfg = cfg
        self.stop_strings = stop_strings or []; self.is_stream = is_stream; self.fut = fut
        self.gen = []; self.next = None
        self.emitted_safe = 0          # chars definitely safe to emit (buffered past partial-stop window)
        self.final_text = None
        self.queue = asyncio.Queue() if is_stream else None
        self.max_stop_len = max((len(s) for s in self.stop_strings), default=0)


class SlottedScheduler:
    def __init__(self, eng, tok):
        self.eng = eng; self.tok = tok
        L, H, N = eng.L, eng.H, eng.N; hd = eng.hidden
        self.sa = torch.zeros(L, 0, H, N, N, dtype=torch.float32, device=DEV)
        self.xp = torch.zeros(L, 0, hd, dtype=torch.float16, device=DEV)
        self.xf = torch.zeros(L, 0, hd, dtype=torch.float16, device=DEV)
        self.vf = torch.zeros(0, hd, dtype=torch.float16, device=DEV)
        self.seqs = []; self.B = 0; self.tok_count = 0

    def add(self, seq):
        (sa, xp, xf, vf), first = prefill_one(self.eng, seq.prompt_ids)
        self.sa = torch.cat([self.sa, sa], dim=1); self.xp = torch.cat([self.xp, xp], dim=1)
        self.xf = torch.cat([self.xf, xf], dim=1); self.vf = torch.cat([self.vf, vf], dim=0)
        seq.gen = [first]; seq.next = first
        self.seqs.append(seq); self.B += 1; return seq

    def _shrink(self, i):
        last = self.B - 1
        if i != last:
            self.sa[:, i].copy_(self.sa[:, last]); self.xp[:, i].copy_(self.xp[:, last])
            self.xf[:, i].copy_(self.xf[:, last]); self.vf[i].copy_(self.vf[last])
            self.seqs[i] = self.seqs[last]
        self.seqs.pop()
        self.sa = self.sa[:, :-1].contiguous(); self.xp = self.xp[:, :-1].contiguous()
        self.xf = self.xf[:, :-1].contiguous(); self.vf = self.vf[:-1].contiguous()
        self.B -= 1

    def step(self, loop=None):
        if self.B == 0:
            return []
        emb = self.eng.base.embeddings(torch.tensor([s.next for s in self.seqs], device=DEV))
        logits = self.eng.mod.rwkv7_decode_full(emb, *self.eng.W, self.sa, self.xp, self.xf, self.vf,
                                                 self.eng.H, self.eng.N, self.eng.lm_w_m, self.eng.fnorm_w, self.eng.fnorm_b)
        nxt = sample_rows(logits, [s.cfg for s in self.seqs])
        done = []
        for i in range(self.B):
            seq = self.seqs[i]
            seq.gen.append(nxt[i]); seq.next = nxt[i]; self.tok_count += 1
            decoded = self.tok.decode(seq.gen)
            stop_idx = None
            for ss in seq.stop_strings:
                j = decoded.find(ss)
                if j >= 0 and (stop_idx is None or j < stop_idx):
                    stop_idx = j
            if stop_idx is not None:
                safe_end = min(len(decoded) - seq.max_stop_len, stop_idx)
                final_end = stop_idx; terminate = True
            elif len(seq.gen) >= seq.max_new:
                safe_end = len(decoded) - seq.max_stop_len
                final_end = len(decoded); terminate = True
            else:
                safe_end = len(decoded) - seq.max_stop_len
                final_end = None; terminate = False
            if seq.is_stream:
                if safe_end > seq.emitted_safe:
                    seq.queue.put_nowait(decoded[seq.emitted_safe:safe_end]); seq.emitted_safe = safe_end
                if terminate:
                    if final_end > seq.emitted_safe:
                        seq.queue.put_nowait(decoded[seq.emitted_safe:final_end]); seq.emitted_safe = final_end
                    seq.queue.put_nowait(None)  # sentinel
            if terminate:
                seq.final_text = decoded[:final_end]
                done.append(i)
        results = []
        for i in sorted(done, reverse=True):
            seq = self.seqs[i]
            if not seq.is_stream and seq.fut is not None:
                if loop is not None:
                    loop.call_soon_threadsafe(seq.fut.set_result, seq.final_text)
                else:
                    seq.fut.set_result(seq.final_text)
            results.append(seq); self._shrink(i)
        return results


# ---------------- async server (single-threaded loop) ----------------
class AsyncServer:
    def __init__(self, eng, tok, loop):
        self.eng = eng; self.tok = tok; self.loop = loop
        self.sch = SlottedScheduler(eng, tok); self.pending = []
        loop.create_task(self._run())

    async def _run(self):
        while True:
            if self.pending:
                for seq in self.pending:
                    try:
                        self.sch.add(seq)
                    except Exception as e:
                        if seq.fut is not None:
                            seq.fut.set_exception(e)
                        elif seq.is_stream:
                            seq.queue.put_nowait(None)
                self.pending = []
            if self.sch.B > 0:
                self.sch.step(self.loop)
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.001)

    def _new_seq(self, prompt, max_tokens, temperature, top_k, top_p, stop, is_stream):
        cfg = SamplerCfg(temperature, top_k, top_p)
        stops = [stop] if isinstance(stop, str) else (stop or [])
        ids = self.tok.encode(prompt)
        fut = None if is_stream else asyncio.Future()
        return Seq(ids, max_tokens, cfg, stops, is_stream, fut)

    async def complete(self, prompt, max_tokens=50, temperature=0.0, top_k=0, top_p=1.0, stop=None):
        seq = self._new_seq(prompt, max_tokens, temperature, top_k, top_p, stop, False)
        self.pending.append(seq)
        return await seq.fut

    def submit_stream(self, prompt, max_tokens=50, temperature=0.0, top_k=0, top_p=1.0, stop=None):
        seq = self._new_seq(prompt, max_tokens, temperature, top_k, top_p, stop, True)
        self.pending.append(seq)
        return seq

    async def stream_tokens(self, seq):
        while True:
            item = await seq.queue.get()
            if item is None:
                break
            yield item


# ---------------- FastAPI face ----------------
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional, Union, List
from pydantic import BaseModel
app = FastAPI(title="RWKV7-Ascend-Full")
_server = [None]
_eng = None; _tok = None


@app.on_event("startup")
async def _startup():
    _server[0] = AsyncServer(_eng, _tok, asyncio.get_running_loop())


class Req(BaseModel):
    prompt: str = ""
    max_tokens: int = 50
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


def _sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


@app.post("/v1/completions")
async def completions(r: Req):
    srv = _server[0]
    if r.stream:
        seq = srv.submit_stream(r.prompt, r.max_tokens, r.temperature, r.top_k, r.top_p, r.stop)

        async def sse():
            try:
                async for piece in srv.stream_tokens(seq):
                    yield _sse({"choices": [{"delta": {"content": piece}, "index": 0, "finish_reason": None}]})
            except Exception as e:
                yield _sse({"error": str(e)})
            yield _sse({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")
    else:
        try:
            txt = await srv.complete(r.prompt, r.max_tokens, r.temperature, r.top_k, r.top_p, r.stop)
            return {"text": txt}
        except Exception as e:
            return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf")
    ap.add_argument("--H", type=int, default=12); ap.add_argument("--N", type=int, default=64); ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--vocab", default="/root/rwkv7-ascend/assets/rwkv_vocab_v20230424.txt")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    eng = RWKV7Engine(args.model, args.H, args.N, args.L)
    tok = _RWKVTrie(args.vocab)
    _eng = eng; _tok = tok
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
