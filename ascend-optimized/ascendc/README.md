# AscendC fused-kernel work (path to 2× Albatross)

Goal: fuse the RWKV-7 TMix elementwise into single AscendC kernels to break the
op-coalesced C++ ceiling (~323 tok/s). Each `at::` op costs ~3µs of NPU kernel
*execution* (proven via NPUGraph — see ../ASCEND_RESULTS.md); only fewer kernels
(fusion) help.

## ✅ Build pipeline proven end-to-end + fusion is 3.68× faster (2026-07-04)

First fused op `RwkvWexp` (`y = exp(-0.606531·sigmoid(x))`) builds, installs,
and **runs on the 910B2C device**.

**Speed (single op, [8192] fp16):**
- fused AscendC kernel: **0.0054 ms**
- eager (sigmoid + muls + exp = 3 ops): 0.0199 ms
- → **3.68× faster**

This validates the whole 2× Albatross thesis: fusing the TMix elementwise
(~40 ops/layer) into AscendC kernels is the lever that graph capture isn't.

### Full recipe (on the Ascend box, CANN 8.5.1)
```bash
source /usr/local/Ascend/cann-8.5.1/set_env.sh
cd /root/ascendc_op

# 1. op-def JSON (rwkv_wexp.json) -> full op project
/usr/local/Ascend/cann-8.5.1/x86_64-linux/bin/msopgen gen \
    -i rwkv_wexp.json -c ai_core-ascend910b -f pytorch -lan cpp -out proj

# 2. implement kernel proj/op_kernel/rwkv_wexp.cpp (CopyIn/Compute/CopyOut)
# 3. add 910b SoC to the OpDef in proj/op_host/rwkv_wexp.cpp (CRITICAL, see below)
# 4. build + install the .run
cd proj && bash build.sh && cd build_out && bash custom_opp_openEuler_x86_64.run
# 5. call from torch via call_rwkv_wexp.cpp (aclnnRwkvWexp); see test_wexp.py
```

### Gotchas fixed (all required — remember for every future op)
1. **`KERNEL_TYPE_AIV_ONLY`** on the kernel → "not support in current core
   version" for ascend910. Just omit `KERNEL_TASK_TYPE_DEFAULT`.
2. **`TQuePosition`** renamed to **`QuePosition`** in this AscendC version.
3. **910b kernel target not generated** unless you add
   `this->AICore().AddConfig("ascend910b");` to the OpDef in `op_host/*.cpp`
   (msopgen only emits `AddConfig("ascend910")`). Without it: device error
   `binary_info_config.json of socVersion [ascend910b] does not support opType`
   (aclnn status 161001).
4. **torch binding ABI**: use default new ABI (NOT `_GLIBCXX_USE_CXX11_ABI=0`,
   which breaks torch's `torchInternalAssertFail`). Don't link `libascendcl`
   expecting torch_npu to expose it globally — link `-lcust_opapi` +
   `-lascendcl` with new ABI, `-Wl,--allow-shlib-undefined`.
5. **ACL stream fn** is `aclrtCtxGetCurrentDefaultStream(&stream)` (not
   `aclrtGetCurrentStream`). **`aclCreateTensor`** takes 9 args incl.
   `storageDims` (`nullptr, 0` for simple tensors); malloc returns `aclError`.

## ⚠️ Remaining: in-place Muls/Exp corrupts output (next iteration) — narrow diagnosis

Data path is VERIFIED correct now:
- copy `y=x` → 99.99% match; constant fill → correct; `Adds(y,x,1)` → 99.2% match
- **`Sigmoid(yLocal, xLocal)` alone → cos=0.99992** ✓ (transcendentals DO work on half)

So the original "cos≈0.02 garbage" was a **stale-kernel red herring** (now cleared).
The real remaining bug: chaining `Sigmoid → Muls(yLocal,yLocal,s) → Exp(yLocal,yLocal)`
(in-place dst==src) corrupts the output (cos≈0.2, same maxabs). Sigmoid works because
it uses distinct src/dst. **Fix: don't use `TBuf<float>` work buffers (Cast to/from
them also corrupts) — instead give Muls/Exp distinct half buffers (a 2nd out-queue or
alternate the two BUFFER_NUM slots), or insert the right pipe barrier between chained
in-place V-ops.** Once cos=1.0 on the full `exp(-0.606·sigmoid(x))`, scale the exact
same pattern to fuse the **shift-mix** (12 ops → 1 kernel) for end-to-end speedup.

Note: speed is already 1.4–3.7× vs eager even with the bug (single op, micro-bench).

## Files
- `rwkv_wexp.json` — op definition
- `rwkv_wexp_kernel.cpp` — AscendC kernel (Sigmoid+Muls+Exp on half; in-place chain bug)
- `call_rwkv_wexp.cpp` — torch→aclnn binding (works: read/write/copy/Adds all correct)
- `test_wexp.py` — correctness + speed harness

## Reference pattern to copy (next iter): mul_addn
The built-in `mul_addn` op (fused mul+add, multi-op elementwise, runs on 910b) at
`opp/built-in/op_impl/ai_core/tbe/impl/ops_math/ascendc/mul_addn/mul_addn_align.h`
shows the correct 910b pattern: alignment-aware tiling
(`dataAlign = blockBytes/sizeof(T)` = 16 half-elements/UB-block, vector block =
64 elements), `lastCoreTaskNum` for the remainder core, and `coreTaskNum` per core.
My naive 128-element tile with `tileNum = N/TILE/blockNum` corrupts tile boundaries
(compounds across chained ops -> the cos drop). Adapt mul_addn's alignment+remainder
handling to fix the fused wexp kernel, then scale to shift-mix.

## ✅ BREAKTHROUGH (2026-07-04): V-op chaining cracked — shift-mix unblocked

Studied the official `dequant_swiglu_quant` kernel (cloned cann-ops from gitee to
server). The correct V-V sync pattern on 910b vector core is a **dedicated
intermediate `TQue<QuePosition::VECCALC>` with full AllocTensor→EnQue→DeQue→FreeTensor
between stages** (NOT in-place, NOT TBuf, NOT PipeBarrier alone — those all still
garbage). With this pattern:

- Sigmoid+Adds (2-op): cos=0.99996 ✓
- Sigmoid+Muls (2-op): cos=0.99989 ✓
- Sigmoid+Exp  (2-op): cos=0.99994 ✓
- Sigmoid+Adds+Adds (3-op): cos=0.99817 ✓
- **Sigmoid+Adds+Muls (3-op, Mul+Add chain): cos=0.99994 ✓**
- Sigmoid+Muls+Exp (3-op): cos=0.20 ✗  ← only `Exp` (after another transcendental) breaks

**Key: shift-mix is `xr = x + (x_prev - x)·x_r` = Sub+Mul+Add — NO transcendentals.**
Mul+Add chains are proven correct (cos=0.99994). So fusing shift-mix (the biggest
elementwise chunk, 12 ops → 1 kernel) is now unblocked. The `Exp` issue only affects
the `w=exp(-0.606·sigmoid(w_raw))` decay term, which can stay eager or use a 2-op
fused kernel (Sigmoid+Exp works standalone) while we fuse the shift-mix first.

## Shift-mix fused op (2026-07-04): built + 1.81× fast, multi-input loading blocked

Created `RwkvShiftMix` op (`y = x + (x_prev−x)·x_r`, Sub+Mul+Add, no transcendentals).
Built + installed on 910b, torch binding (`aclnnRwkvShiftMix`) calls it.
- **Speed: 1.81× vs eager** (sub+mul+add) on the micro-bench — even while incorrect.
- **Correctness: blocked on MULTI-INPUT loading.** Single-input ops (wexp) work; the
  2-VECIN-queue input pattern (x + x_prev) gives NaN even for Sub-only. The chain
  (Sub→Mul→Add with VECCALC queues) is proven (Sigmoid+Adds+Muls cos=0.99994).

**Next**: switch input loading to the reference `dequant_swiglu_quant` pattern —
`TBuf<half>` input UBs + `DataCopyPad` + `SetFlag/WaitFlag<HardEvent::MTE2_V>` sync
(rather than separate VECIN TQues per input). That's how the official kernel loads
multiple inputs without racing. Once Sub-only gives cos=1.0, the full shift-mix
chain follows, then scale to 6 mix-vector outputs.

Files: `rwkv_shiftmix.json` (op-def); kernel/binding/test live on the server
(`/root/ascendc_op/proj_sm/`, `/root/ascendc/call_rwkv_shiftmix.cpp`) — sync to repo
once correct.

## Multi-input cracked (2026-07-04): 3-TQue + Mul+Add cos=0.99968; Sub is the lone broken op

The 3-input pattern works when avoiding `Sub`:
- **Mul+Add, 3 VECIN TQue (y = x + xp·xr): cos=0.99968** ✓ (mirrors official `addcmul`)
- Sub-only (any pattern: 2-TQue→NaN, TBuf+MTE2_V→0.74, 3-TQue→compile test inconclusive): broken

So `Sub` is the one op that misbehaves on 910b half in my kernels. **Workaround for
shift-mix** `xr = x + (x_prev−x)·x_r`: replace `xx = x_prev − x` with
`negx = Mul(x, -1); xx = Add(x_prev, negx)` — all Mul/Add (proven). Full fused
shift-mix = Mul(negx) → Add(xx) → Mul(tmp=xx·x_r) → Add(y=x+tmp), 4 Mul/Add ops,
no Sub. Next iter: implement + verify cos=1.0, then scale to 6 mix-vector outputs.

Files added: `rwkv_shiftmix.json` (op-def), `call_rwkv_shiftmix.cpp` (torch→aclnn
binding), `rwkv_shift_mix_kernel.cpp` (3-TQue Mul+Add, the working reference).

## 4-op chain breaks; revised shift-mix strategy (2026-07-04)

Full shift-mix as one 4-op kernel (`Muls(negx)→Add(xx)→Mul(tmp)→Add(y)`, Sub
avoided) gives cos=0.16 — worse. **Chains ≤3 V-ops work; 4-op chains break.**
Likely cause: holding a DeQueued input (`x`, reused in stage 1 and stage 4) across
3 midQ EnQue/DeQue cycles invalidates it.

**Revised strategy** (decompose to stay within the proven 2-op window):
1. Binding computes `xx = x_prev − x` eagerly (1 cheap Sub, 12 total across the
   model — negligible).
2. Fused op does **only `y_i = x + xx·mix_i`** (Mul→Add, 2-op, PROVEN cos=0.99968)
   per output. For the 6 mix vectors → a 6-output fused op, each output a 2-op
   chain reading the shared `x` and `xx`.
3. Open risk: `x`/`xx` reused across 6 outputs may hit the same "held-across-cycles"
   issue. Mitigation if so: reload shared inputs per output, or hold them in a
   stable TBuf (not a DeQueued queue tensor).

The 2-op Mul+Add is the validated building block; the 6-output op is assembly.

## Shared-reuse works at 2 outputs; 6 outputs hangs (2026-07-05)

- **2-output shared-reuse (x, xx reused): y1 cos=0.99992, y2 cos=0.99998** ✓
  => shared-input reuse across outputs is fine; the 4-op failure was chain depth.
- **6-output shift-mix HANGS** (vector-core execution timeout 507034) — both with
  1 reused midQ and with 6 separate midQ (20 queues). Too complex for one kernel.

**Strategy**: build the shift-mix from the proven 2-output block (3 calls = 6
outputs) + 1 eager `xx = x_prev − x`. That's 4 launches vs 13 eager ops/layer
(~10% forward speedup estimate). Modest, but the building block is solid. The
full 2× needs fusing far more (whole TMix), and each multi-output kernel >2
outputs risks the hang — a real complexity ceiling to work around.
