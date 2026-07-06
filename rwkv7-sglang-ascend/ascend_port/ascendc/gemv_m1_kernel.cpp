// AscendC M=1 GEMV kernel for RWKV-7 projections (DRAFT v0 -- needs rebuild+test).
// y[m] = sum_k x[k] * W[m, k]   (W row-major [M, K]; x [K]; y [M])
// Bandwidth-bound: read W once. Adapted from sgl-kernel-npu/lora/op_kernel/
// sgemmv_shrink_kernel.cpp (Huawei, Apache-2.0), stripped of LoRA indexing.
//
// Status: scaffold -- the AscendC API (DataCopy/Compute/tiling) needs the
// sgl-kernel-npu rebuild loop to validate. Each core owns a slab of M outputs;
// it tiles over K, accumulating the dot product in fp32, then writes y.
//
// Build: add to /data/sgl-kernel-npu/csrc/ (op_kernel + op_host/tiling) +
// csrc/CMakeLists.txt + pytorch_extensions.cpp binding, then rebuild.
#ifndef SGL_KERNEL_NPU_KERNEL_GEMV_M1_H
#define SGL_KERNEL_NPU_KERNEL_GEMV_M1_H

#include "kernel_operator.h"

template <typename scalar_t>
class GemvM1
{
public:
    using X_T = scalar_t;   // fp16 (half)
    using W_T = scalar_t;   // fp16
    using Y_T = scalar_t;   // fp16

    static constexpr uint64_t BUFFER_NUM = 2;
    // K-tile loaded into UB each step (in elements). 2048 fits a UB slab; tune.
    static constexpr uint64_t K_TILE = 2048;

    __aicore__ inline GemvM1(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                uint32_t mTotal, uint32_t kDim,
                                uint32_t mPerCore)
    {
        mTotal_ = mTotal;
        kDim_ = kDim;
        mPerCore_ = mPerCore;
        kTiles_ = (kDim + K_TILE - 1) / K_TILE;

        int64_t coreId = AscendC::GetBlockIdx();
        mStart_ = coreId * mPerCore_;
        mEnd_ = AscendC::Min(mStart_ + mPerCore_, mTotal_);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(y));

        // UB buffers: a K_TILE of x, a K_TILE of one W row, an accumulator row.
        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, K_TILE * sizeof(X_T));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, K_TILE * sizeof(W_T));
        pipe_->InitBuffer(accBuf_, mPerCore_ * sizeof(float));   // per-output fp32 acc
        // (mPerCore outputs accumulated across K tiles; tile over K, not M, so the
        //  acc is [mPerCore] reused each K_TILE step.)
    }

    __aicore__ inline void Process()
    {
        // init accumulator to 0
        AscendC::SetVectorMask<float>(0, mPerCore_ - 1);
        // (pseudo: acc = 0; the real AscendC uses Duplicate/Computes on the LocalTensor)

        for (uint32_t m = mStart_; m < mEnd_; ++m) {
            float acc = 0.0f;
            for (uint32_t kt = 0; kt < kTiles_; ++kt) {
                uint32_t kLen = AscendC::Min(K_TILE, kDim_ - kt * K_TILE);
                // DataCopy x[kt*K_TILE : kt*K_TILE+kLen] -> inQueueX (UB)
                // DataCopy W[m, kt*K_TILE : ...] -> inQueueW (UB)
                // Compute: acc += sum(x_tile * w_tile)   (elementwise mul + ReduceSum)
                // (AscendC: Mul + ReduceSum on the LocalTensors, fp32 accumulate.)
            }
            // write y[m] = (Y_T)acc
        }
        // NOTE: the loop bodies above are the AscendC DataCopy/Compute calls that
        // need to be filled per the AscendC API (DataCopy<T>, Add, Mul, ReduceSum,
        // Cast). This draft captures the structure + tiling; the rebuild-test loop
        // (sgl-kernel-npu rebuild) validates + tunes.
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    uint32_t mTotal_, kDim_, mPerCore_, kTiles_, mStart_, mEnd_;
};

extern "C" __global__ __aicore__ void gemv_m1(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    // tiling carries: mTotal, kDim, mPerCore (computed by the op_host tiling fn).
    GemvM1<half> op(&pipe);
    // op.Init(x, weight, y, tiling->mTotal, tiling->kDim, tiling->mPerCore);
    // op.Process();
    (void)x; (void)weight; (void)y; (void)workspace; (void)tiling;
}

#endif  // SGL_KERNEL_NPU_KERNEL_GEMV_M1_H
