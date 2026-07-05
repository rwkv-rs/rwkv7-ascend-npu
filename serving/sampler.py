"""RWKV7 sampler — separated so it's unit-testable without NPU / rwkv7_hf."""
import torch


class SamplerCfg:
    def __init__(self, temperature=0.0, top_k=0, top_p=1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


def sample_rows(logits, cfgs):
    """logits: [B, vocab] (fp16/fp32, any device); cfgs: list[B] of SamplerCfg -> list[B] int tokens."""
    if all(c.temperature <= 0 for c in cfgs):
        return logits.argmax(-1).tolist()  # batched greedy fast-path
    out = []
    for i, cfg in enumerate(cfgs):
        row = logits[i].float()
        if cfg.temperature <= 0:
            out.append(int(row.argmax()))
            continue
        row = row / cfg.temperature
        if cfg.top_k > 0:
            k = min(cfg.top_k, row.shape[-1])
            vals, idx = torch.topk(row, k)
            mask = torch.full_like(row, float("-inf"))
            mask[idx] = vals
            row = mask
        if cfg.top_p < 1.0:
            sval, sidx = torch.sort(row, descending=True)
            cum = torch.softmax(sval, dim=-1).cumsum(dim=-1)
            keep = cum <= cfg.top_p
            keep[0] = True
            row = torch.full_like(row, float("-inf"))
            row[sidx[keep]] = sval[keep]
        probs = torch.softmax(row, dim=-1)
        try:
            tok = int(torch.multinomial(probs, 1))
        except Exception:
            tok = int(probs.argmax())
        out.append(tok)
    return out
