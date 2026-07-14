from perf.npu_memory import NPUProcessMemory, PeakNPUMemorySampler, parse_npu_smi_process_memory


SAMPLE = """
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
| 0       0                 | 123           | python3            | 8192                  | 123                     |
| 1       0                 | 456           | VLLM::Worker       | 4096.5                | 456                     |
| No running processes found in NPU 2 |
"""


def test_parse_npu_smi_process_rows():
    assert parse_npu_smi_process_memory(SAMPLE) == (
        NPUProcessMemory(0, 123, "python3", 8192.0),
        NPUProcessMemory(1, 456, "VLLM::Worker", 4096.5),
    )


def test_peak_sampler_sums_only_selected_devices():
    samples = iter(
        [
            (
                NPUProcessMemory(0, 1, "python", 100.0),
                NPUProcessMemory(1, 2, "worker", 200.0),
            ),
            (
                NPUProcessMemory(0, 1, "python", 150.0),
                NPUProcessMemory(1, 2, "worker", 250.0),
            ),
        ]
    )
    sampler = PeakNPUMemorySampler([1], query=lambda: next(samples))

    sampler._sample()
    sampler._sample()

    assert sampler.peak_memory_mib == 250.0
    assert sampler.scope == "all_npu_processes_on_selected_devices"
