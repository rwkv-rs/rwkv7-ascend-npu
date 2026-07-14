"""Process-level Ascend memory telemetry backed by ``npu-smi info``."""
from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class NPUProcessMemory:
    device_id: int
    process_id: int
    process_name: str
    memory_mib: float


def parse_npu_smi_process_memory(output: str) -> tuple[NPUProcessMemory, ...]:
    """Parse process rows from the stable pipe-delimited ``npu-smi`` table."""
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) < 5 or not re.fullmatch(r"\d+\s+\d+", fields[0]):
            continue
        if not fields[1].isdigit():
            continue
        memory = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", fields[3])
        if memory is None:
            continue
        rows.append(
            NPUProcessMemory(
                device_id=int(fields[0].split()[0]),
                process_id=int(fields[1]),
                process_name=fields[2],
                memory_mib=float(memory.group(1)),
            )
        )
    return tuple(rows)


def query_npu_process_memory() -> tuple[NPUProcessMemory, ...]:
    completed = subprocess.run(
        ["npu-smi", "info"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return parse_npu_smi_process_memory(completed.stdout)


class PeakNPUMemorySampler:
    """Poll aggregate process memory for selected NPUs in a background thread."""

    scope = "all_npu_processes_on_selected_devices"

    def __init__(
        self,
        device_ids: Iterable[int],
        *,
        interval_seconds: float = 0.2,
        query: Callable[[], tuple[NPUProcessMemory, ...]] = query_npu_process_memory,
    ):
        self.device_ids = frozenset(int(value) for value in device_ids)
        if not self.device_ids:
            raise ValueError("at least one NPU device id is required")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.query = query
        self.samples_mib: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            rows = self.query()
            self.samples_mib.append(
                sum(
                    row.memory_mib
                    for row in rows
                    if row.device_id in self.device_ids
                )
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.errors.append(str(error))

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> "PeakNPUMemorySampler":
        if self._thread is not None:
            raise RuntimeError("memory sampler already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float | None:
        if self._thread is None:
            raise RuntimeError("memory sampler was not started")
        self._stop.set()
        self._thread.join(timeout=15)
        self._sample()
        self._thread = None
        return self.peak_memory_mib

    @property
    def peak_memory_mib(self) -> float | None:
        return max(self.samples_mib) if self.samples_mib else None

    def __enter__(self) -> "PeakNPUMemorySampler":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
