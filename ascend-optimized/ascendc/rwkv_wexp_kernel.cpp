#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 128;
constexpr float EXP_HALF = 0.606531f;
class K {
public:
    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR y_gm, int32_t totalLength) {
        int32_t bn=GetBlockNum(), bi=GetBlockIdx();
        tileNum=totalLength/TILE_LENGTH/bn;
        xGm.SetGlobalBuffer((__gm__ half*)x_gm + bi*tileNum*TILE_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half*)y_gm + bi*tileNum*TILE_LENGTH);
        pipe.InitBuffer(inQ, BUFFER_NUM, TILE_LENGTH*sizeof(half));
        pipe.InitBuffer(midQ, BUFFER_NUM, TILE_LENGTH*sizeof(half));
        pipe.InitBuffer(outQ, BUFFER_NUM, TILE_LENGTH*sizeof(half));
    }
    __aicore__ inline void Process(){ for(int32_t i=0;i<tileNum;i++){CopyIn(i);Compute(i);CopyOut(i);} }
private:
    __aicore__ inline void CopyIn(int32_t p){ LocalTensor<half> x=inQ.AllocTensor<half>(); DataCopy(x,xGm[p*TILE_LENGTH],TILE_LENGTH); inQ.EnQue<half>(x);}
    __aicore__ inline void Compute(int32_t p){
        LocalTensor<half> x=inQ.DeQue<half>();
        LocalTensor<half> t=midQ.AllocTensor<half>();
        Sigmoid(t,x,TILE_LENGTH);
        midQ.EnQue<half>(t);
        inQ.FreeTensor(x);
        t=midQ.DeQue<half>();
        PipeBarrier<PIPE_V>();
        Adds(t,t,(half)1.0f,TILE_LENGTH);
        midQ.EnQue<half>(t);
        t=midQ.DeQue<half>();
        PipeBarrier<PIPE_V>();
        LocalTensor<half> y=outQ.AllocTensor<half>();
        Muls(y,t,static_cast<half>(2.0f),TILE_LENGTH);
        outQ.EnQue<half>(y);
        midQ.FreeTensor(t);
    }
    __aicore__ inline void CopyOut(int32_t p){ LocalTensor<half> y=outQ.DeQue<half>(); DataCopy(yGm[p*TILE_LENGTH],y,TILE_LENGTH); outQ.FreeTensor(y);}
    TPipe pipe;
    TQue<QuePosition::VECIN,BUFFER_NUM> inQ;
    TQue<QuePosition::VECCALC,BUFFER_NUM> midQ;
    TQue<QuePosition::VECOUT,BUFFER_NUM> outQ;
    GlobalTensor<half> xGm,yGm; int32_t tileNum;
};
extern "C" __global__ __aicore__ void rwkv_wexp(GM_ADDR x, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling){
    GET_TILING_DATA(t,tiling); K op; op.Init(x,y,t.size); op.Process();
}
