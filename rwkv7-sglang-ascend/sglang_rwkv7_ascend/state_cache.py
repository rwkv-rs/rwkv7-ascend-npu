"""Explicit recurrent-state operations used by tests and operational tooling.

SGLang owns request-slot allocation in production.  This adapter makes the
RWKV-specific state contract observable and supplies safe copy/offload/restore
operations without depending on a KV-cache layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class RWKV7StateSnapshot:
    conv_att: torch.Tensor
    conv_ffn: torch.Tensor
    temporal: torch.Tensor

    def pin_memory(self) -> "RWKV7StateSnapshot":
        if torch.cuda.is_available():
            self.conv_att = self.conv_att.pin_memory()
            self.conv_ffn = self.conv_ffn.pin_memory()
            self.temporal = self.temporal.pin_memory()
        return self


class SGLangMambaPoolStateAdapter:
    """RWKV state operations over an SGLang ``MambaPool``.

    Slots are physical Mamba slots, not request-pool ids.  Slot 0 is reserved
    for graph padding and is rejected for user state operations.
    """

    def __init__(self, pool):
        self.pool = pool

    @staticmethod
    def _idx(slots: Sequence[int] | torch.Tensor, device) -> torch.Tensor:
        out = torch.as_tensor(slots, dtype=torch.long, device=device)
        if out.ndim != 1 or out.numel() == 0:
            raise ValueError("slots must be a non-empty 1-D sequence")
        if bool(torch.any(out <= 0)):
            raise ValueError("slot 0 is reserved; state slots must be positive")
        return out

    @property
    def device(self):
        return self.pool.mamba_cache.temporal.device

    def clear(self, slots: Sequence[int] | torch.Tensor) -> None:
        idx = self._idx(slots, self.device)
        self.pool.clear_slots(idx)

    def clone(self, src_slots, dst_slots) -> None:
        src = self._idx(src_slots, self.device)
        dst = self._idx(dst_slots, self.device)
        if src.shape != dst.shape:
            raise ValueError("source and destination slot counts differ")
        self.pool.copy_from(src, dst)

    def reorder(self, src_slots, dst_slots) -> None:
        """Alias for clone with an explicit scheduler-facing name."""

        self.clone(src_slots, dst_slots)

    def offload(self, slots, *, pin_memory: bool = False) -> RWKV7StateSnapshot:
        idx = self._idx(slots, self.device)
        state = self.pool.mamba_cache

        def gather(t):
            # tensors are [layer, slot, ...]
            return t[:, idx].detach().to("cpu", copy=True)

        snap = RWKV7StateSnapshot(
            conv_att=gather(state.conv[0]),
            conv_ffn=gather(state.conv[1]),
            temporal=gather(state.temporal),
        )
        return snap.pin_memory() if pin_memory else snap

    def restore(self, slots, snapshot: RWKV7StateSnapshot) -> None:
        idx = self._idx(slots, self.device)
        state = self.pool.mamba_cache
        expected = idx.numel()
        for name, dst, src in (
            ("conv_att", state.conv[0], snapshot.conv_att),
            ("conv_ffn", state.conv[1], snapshot.conv_ffn),
            ("temporal", state.temporal, snapshot.temporal),
        ):
            if src.shape[1] != expected:
                raise ValueError(
                    f"{name} snapshot has {src.shape[1]} slots, expected {expected}"
                )
            dst[:, idx] = src.to(device=dst.device, dtype=dst.dtype, non_blocking=True)


__all__ = ["RWKV7StateSnapshot", "SGLangMambaPoolStateAdapter"]
