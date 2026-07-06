// op_kernel/rwkv_wexp.cpp — AscendC fused kernel: y = exp(-EXP_HALF * sigmoid(x)).
// First custom op to prove the AscendC build pipeline end-to-end on 910B.
// Standard AscendC tutorial pattern (TPipe/TQue/DataCopy + vector Sigmoid/Muls/Exp).
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 128;          // half elements per tile (256 B, 1 UB block aligned)
constexpr float EXP_HALF = 0.606531f;

class KernelRwkvWexp {
public:
    __aicore__ inline KernelRwkvWexp() {}
    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR y_gm, int32_t totalLength) {
        int32_t blockNum = GetBlockNum();
        int32_t blockIdx = GetBlockIdx();
        // per-core tile count (assumes totalLength divisible by blockNum*TILE_LENGTH)
        this->tileNum = totalLength / TILE_LENGTH / blockNum;
        xGm.SetGlobalBuffer((__gm__ half *)x_gm + blockIdx * this->tileNum * TILE_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y_gm + blockIdx * this->tileNum * TILE_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue<half>(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        // y = exp(-EXP_HALF * sigmoid(x))
        Sigmoid(yLocal, xLocal, TILE_LENGTH);
        Muls(yLocal, yLocal, (half)(-EXP_HALF), TILE_LENGTH);
        Exp(yLocal, yLocal, TILE_LENGTH);
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * TILE_LENGTH], yLocal, TILE_LENGTH);
        outQueueY.FreeTensor(yLocal);
    }
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    int32_t tileNum;
};

extern "C" __global__ __aicore__ void rwkv_wexp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelRwkvWexp op;
    op.Init(x, y, tilingData.size);
    op.Process();
}
