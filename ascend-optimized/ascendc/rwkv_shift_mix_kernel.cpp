#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 128;
class K {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xp, GM_ADDR xr, GM_ADDR y, int32_t totalLength) {
        int32_t bn=GetBlockNum(), bi=GetBlockIdx();
        tileNum=totalLength/TILE_LENGTH/bn;
        xGm.SetGlobalBuffer((__gm__ half*)x + bi*tileNum*TILE_LENGTH);
        xpGm.SetGlobalBuffer((__gm__ half*)xp + bi*tileNum*TILE_LENGTH);
        xrGm.SetGlobalBuffer((__gm__ half*)xr + bi*tileNum*TILE_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half*)y + bi*tileNum*TILE_LENGTH);
        pipe.InitBuffer(inQx, BUFFER_NUM, TILE_LENGTH*sizeof(half));
        pipe.InitBuffer(inQp, BUFFER_NUM, TILE_LENGTH*sizeof(half));
        pipe.InitBuffer(inQr, BUFFER_NUM, TILE_LENGTH*sizeof(half));
        pipe.InitBuffer(midQ, BUFFER_NUM, TILE_LENGTH*sizeof(half));
        pipe.InitBuffer(outQ, BUFFER_NUM, TILE_LENGTH*sizeof(half));
    }
    __aicore__ inline void Process(){ for(int32_t i=0;i<tileNum;i++){CopyIn(i);Compute(i);CopyOut(i);} }
private:
    __aicore__ inline void CopyIn(int32_t p){
        LocalTensor<half> x=inQx.AllocTensor<half>(); DataCopy(x,xGm[p*TILE_LENGTH],TILE_LENGTH); inQx.EnQue<half>(x);
        LocalTensor<half> xp=inQp.AllocTensor<half>(); DataCopy(xp,xpGm[p*TILE_LENGTH],TILE_LENGTH); inQp.EnQue<half>(xp);
        LocalTensor<half> xr=inQr.AllocTensor<half>(); DataCopy(xr,xrGm[p*TILE_LENGTH],TILE_LENGTH); inQr.EnQue<half>(xr);
    }
    __aicore__ inline void Compute(int32_t p){
        LocalTensor<half> x=inQx.DeQue<half>();
        LocalTensor<half> xp=inQp.DeQue<half>();
        LocalTensor<half> xr=inQr.DeQue<half>();
        LocalTensor<half> y=outQ.AllocTensor<half>();
        Sub(y, xp, x, TILE_LENGTH);
        outQ.EnQue<half>(y);
        midQ.FreeTensor(tmp); inQx.FreeTensor(x);
    }
    __aicore__ inline void CopyOut(int32_t p){ LocalTensor<half> y=outQ.DeQue<half>(); DataCopy(yGm[p*TILE_LENGTH],y,TILE_LENGTH); outQ.FreeTensor(y);}
    TPipe pipe;
    TQue<QuePosition::VECIN,BUFFER_NUM> inQx;
    TQue<QuePosition::VECIN,BUFFER_NUM> inQp;
    TQue<QuePosition::VECIN,BUFFER_NUM> inQr;
    TQue<QuePosition::VECCALC,BUFFER_NUM> midQ;
    TQue<QuePosition::VECOUT,BUFFER_NUM> outQ;
    GlobalTensor<half> xGm,xpGm,xrGm,yGm; int32_t tileNum;
};
extern "C" __global__ __aicore__ void rwkv_shift_mix(GM_ADDR x, GM_ADDR x_prev, GM_ADDR x_r, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling); K op; op.Init(x, x_prev, x_r, y, t.size); op.Process();
}
