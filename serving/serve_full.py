import os
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1"); os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import sys; sys.path.insert(0, "/root/rwkv7-ascend")
import time, threading, queue, asyncio
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


# ---------------- slotted scheduler (persistent batched state, NO per-step cat) ----------------
class Seq:
    def __init__(self, prompt, max_new, cfg, fut):
        self.prompt = prompt; self.max_new = max_new; self.cfg = cfg; self.fut = fut
        self.gen = []; self.next = None


class SlottedScheduler:
    def __init__(self, eng):
        self.eng = eng; L, H, N = eng.L, eng.H, eng.N; hd = eng.hidden
        self.sa = torch.zeros(L, 0, H, N, N, dtype=torch.float32, device=DEV)
        self.xp = torch.zeros(L, 0, hd, dtype=torch.float16, device=DEV)
        self.xf = torch.zeros(L, 0, hd, dtype=torch.float16, device=DEV)
        self.vf = torch.zeros(0, hd, dtype=torch.float16, device=DEV)
        self.seqs = []; self.B = 0; self.tok_count = 0

    def add(self, prompt_ids, max_new, cfg, fut=None):
        (sa, xp, xf, vf), first = prefill_one(self.eng, prompt_ids)
        self.sa = torch.cat([self.sa, sa], dim=1); self.xp = torch.cat([self.xp, xp], dim=1)
        self.xf = torch.cat([self.xf, xf], dim=1); self.vf = torch.cat([self.vf, vf], dim=0)
        seq = Seq(prompt_ids, max_new, cfg, fut); seq.gen = [first]; seq.next = first
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
            self.seqs[i].gen.append(nxt[i]); self.seqs[i].next = nxt[i]; self.tok_count += 1
            if len(self.seqs[i].gen) >= self.seqs[i].max_new:
                done.append(i)
        results = []
        for i in sorted(done, reverse=True):
            seq = self.seqs[i]
            if seq.fut is not None:
                if loop is not None:
                    loop.call_soon_threadsafe(seq.fut.set_result, list(seq.gen))
                else:
                    seq.fut.set_result(list(seq.gen))
            results.append(seq); self._shrink(i)
        return results


# ---------------- async server (single-threaded asyncio loop — torch_npu is main-thread-only) ----------------
class AsyncServer:
    def __init__(self, eng, tok, loop):
        self.eng = eng; self.tok = tok; self.loop = loop
        self.sch = SlottedScheduler(eng); self.pending = []
        loop.create_task(self._run())

    async def _run(self):
        while True:
            if self.pending:
                for tokens, max_new, cfg, fut in self.pending:
                    try:
                        self.sch.add(tokens, max_new, cfg, fut)
                    except Exception as e:
                        fut.set_exception(e)
                self.pending = []
            if self.sch.B > 0:
                self.sch.step(self.loop)
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.001)

    async def complete(self, prompt, max_tokens=50, temperature=0.0, top_k=0, top_p=1.0):
        cfg = SamplerCfg(temperature, top_k, top_p)
        fut = asyncio.Future()
        self.pending.append((self.tok.encode(prompt), max_tokens, cfg, fut))
        ids = await fut
        return self.tok.decode(ids)


# ---------------- FastAPI face ----------------
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(title="RWKV7-Ascend-Full")
_server = [None]
_eng = None; _tok = None


@app.on_event("startup")
async def _startup():
    # create the scheduler loop task on uvicorn's RUNNING loop (torch_npu is this thread)
    _server[0] = AsyncServer(_eng, _tok, asyncio.get_running_loop())


class Req(BaseModel):
    prompt: str = ""; max_tokens: int = 50; temperature: float = 0.0; top_k: int = 0; top_p: float = 1.0


@app.post("/v1/completions")
async def completions(r: Req):
    txt = await _server[0].complete(r.prompt, r.max_tokens, r.temperature, r.top_k, r.top_p)
    return {"text": txt}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf")
    ap.add_argument("--H", type=int, default=12); ap.add_argument("--N", type=int, default=64); ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--vocab", default="/root/rwkv7-ascend/assets/rwkv_vocab_v20230424.txt")
    ap.add_argument("--serve", action="store_true"); ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    eng = RWKV7Engine(args.model, args.H, args.N, args.L)
    tok = _RWKVTrie(args.vocab)
    if args.serve:
        _eng = eng; _tok = tok
        import uvicorn; uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # ---- correctness: slotted (greedy, mid-flight join) == standalone ----
        reqs = [(list(range(16)), 8), (list(range(8)), 5), (list(range(4)), 10), (list(range(12)), 6)]
        sch = SlottedScheduler(eng)
        for t, mn in reqs:
            sch.add(t, mn, SamplerCfg())
        sch.step(); sch.step()
        sch.add(list(range(6)), 9, SamplerCfg())  # mid-flight join
        done = []
        while sch.B > 0:
            done += sch.step()
        print("=== slotted scheduler (greedy, mid-flight join) vs standalone ===", flush=True)
        allok = True
        for s in done:
            ref = eng.generate([s.prompt], max_new=s.max_new)[0]
            ok = s.gen == ref; allok &= ok
            print("len%d max%d: %s vs %s %s" % (len(s.prompt), s.max_new, s.gen, ref, "OK" if ok else "DIFF"), flush=True)
        print("ALL_MATCH" if allok else "MISMATCH", flush=True)

        # ---- throughput: 64 concurrent x32 (slotted, no per-step cat) ----
        for _ in range(2):
            s = SlottedScheduler(eng)
            for _ in range(64):
                s.add(list(range(16)), 32, SamplerCfg())
            while s.B > 0:
                s.step()
        s = SlottedScheduler(eng)
        for _ in range(64):
            s.add(list(range(16)), 32, SamplerCfg())
        torch.npu.synchronize(); t0 = time.time()
        while s.B > 0:
            s.step()
        torch.npu.synchronize(); dt = time.time() - t0
        print("64x32 slotted: %d tok in %.2fs = %.0f aggregate tok/s" % (s.tok_count, dt, s.tok_count / dt), flush=True)

        # ---- sampler variety: temp0.8 top_k40, 4 runs (should differ from greedy + each other) ----
        print("=== sampler (temp0.8 top_k40), 4 runs ===", flush=True)
        runs = []
        for _ in range(4):
            s = SlottedScheduler(eng); s.add(list(range(16)), 10, SamplerCfg(0.8, 40, 1.0))
            d = []
            while s.B > 0:
                d += s.step()
            runs.append(d[0].gen)
        greedy = eng.generate([list(range(16))], max_new=10)[0]
        print("greedy: %s" % greedy, flush=True)
        for r in runs:
            print("sample: %s" % r, flush=True)
        print("sampler_varied: %s" % (len(set(map(tuple, runs))) > 1 or any(r != greedy for r in runs)), flush=True)
