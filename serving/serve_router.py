"""RWKV7-Ascend front-end router.

Fans /v1/completions out to N backend workers (each = serve_full.py on its own
port + NPU). Routes each request to the least-in-flight worker. Forwards JSON
and SSE-streaming responses transparently.

Single-NPU box: run 1 worker (RWKV7_WORKERS=http://localhost:8001).
Multi-NPU box: run N workers (one per NPU) + set RWKV7_WORKERS to all of them.
"""
import os, json, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

WORKERS = [w.strip() for w in os.environ.get("RWKV7_WORKERS", "http://localhost:8001").split(",") if w.strip()]
app = FastAPI(title="RWKV7-Ascend-Router")
_inflight = {w: 0 for w in WORKERS}
_lock = asyncio.Lock()


async def _pick():
    async with _lock:
        return min(WORKERS, key=lambda w: _inflight[w])


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.body()
    try:
        data = json.loads(body) if body else {}
    except Exception:
        data = {}
    is_stream = bool(data.get("stream", False))
    w = await _pick()
    async with _lock:
        _inflight[w] += 1
    headers = {"Content-Type": "application/json"}
    timeout = httpx.Timeout(300.0, connect=10.0)

    if is_stream:
        async def sse():
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", w + "/v1/completions", content=body, headers=headers) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
            finally:
                async with _lock:
                    _inflight[w] -= 1
        return StreamingResponse(sse(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(w + "/v1/completions", content=body, headers=headers)
            return JSONResponse(content=r.json(), status_code=r.status_code)
    except httpx.HTTPError as e:
        return JSONResponse({"error": "worker unavailable: %s" % e}, status_code=503)
    finally:
        async with _lock:
            _inflight[w] -= 1


@app.get("/health")
async def health():
    async with httpx.AsyncClient() as c:
        results = {}
        for w in WORKERS:
            try:
                r = await c.get(w + "/health", timeout=5.0)
                results[w] = r.json()
            except Exception as e:
                results[w] = {"error": str(e)}
        return {"workers": results, "inflight": dict(_inflight)}


@app.get("/metrics")
async def metrics():
    lines = ["# HELP rwkv7_inflight In-flight requests per worker",
             "# TYPE rwkv7_inflight gauge",
             'rwkv7_workers_total %d' % len(WORKERS)]
    for w, n in _inflight.items():
        lines.append('rwkv7_inflight{worker="%s"} %d' % (w, n))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse, uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print("[router] forwarding to workers: %s" % WORKERS, flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
