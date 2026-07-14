# rwkv7-vs-qwen35-dense

Global status: **FAIL**

| Tier | Workload | Status | Prefill | Decode | Memory | Reasons |
| --- | --- | --- | ---: | ---: | ---: | --- |
| dense-0 | B1/P512/D128/fp16 | pass | 1.631x | 2.251x | 0.047x | - |
| dense-0 | B4/P512/D128/fp16 | pass | 1.712x | 1.901x | 0.059x | - |
| dense-0 | B1/P2048/D128/fp16 | pass | 1.074x | 2.290x | 0.056x | - |
| dense-0 | B4/P2048/D128/fp16 | pass | 1.018x | 1.920x | 0.096x | - |
| dense-1 | B1/P512/D128/fp16 | pass | 1.235x | 1.674x | 0.148x | - |
| dense-1 | B4/P512/D128/fp16 | pass | 1.155x | 1.527x | 0.172x | - |
| dense-1 | B1/P2048/D128/fp16 | fail | 0.831x | 1.647x | 0.167x | RWKV prefill is not faster |
| dense-1 | B4/P2048/D128/fp16 | fail | 0.644x | 1.532x | 0.252x | RWKV prefill is not faster |
| dense-2 | B1/P512/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-2 | B4/P512/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-2 | B1/P2048/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-2 | B4/P2048/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-3 | B1/P512/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-3 | B4/P512/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-3 | B1/P2048/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-3 | B4/P2048/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-4 | B1/P512/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-4 | B4/P512/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-4 | B1/P2048/D128/fp16 | missing | - | - | - | paired result is missing |
| dense-4 | B4/P2048/D128/fp16 | missing | - | - | - | paired result is missing |
