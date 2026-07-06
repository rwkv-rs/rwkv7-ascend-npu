#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BN=2, TL=128;
class K {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR xx,GM_ADDR m1,GM_ADDR m2,GM_ADDR y1,GM_ADDR y2,int32_t N){
        int32_t b=GetBlockNum(),i=GetBlockIdx(); tn=N/TL/b;
        xGm.SetGlobalBuffer((__gm__ half*)x+i*tn*TL); xxGm.SetGlobalBuffer((__gm__ half*)xx+i*tn*TL);
        m1Gm.SetGlobalBuffer((__gm__ half*)m1+i*tn*TL); m2Gm.SetGlobalBuffer((__gm__ half*)m2+i*tn*TL);
        y1Gm.SetGlobalBuffer((__gm__ half*)y1+i*tn*TL); y2Gm.SetGlobalBuffer((__gm__ half*)y2+i*tn*TL);
        pipe.InitBuffer(qx,BN,TL*2); pipe.InitBuffer(qxx,BN,TL*2); pipe.InitBuffer(qm1,BN,TL*2); pipe.InitBuffer(qm2,BN,TL*2);
        pipe.InitBuffer(mid1,BN,TL*2); pipe.InitBuffer(mid2,BN,TL*2);
        pipe.InitBuffer(oq1,BN,TL*2); pipe.InitBuffer(oq2,BN,TL*2);
    }
    __aicore__ inline void Process(){for(int32_t p=0;p<tn;p++){CopyIn(p);Compute(p);CopyOut(p);}}
private:
    __aicore__ inline void CopyIn(int32_t p){
        LocalTensor<half> x=qx.AllocTensor<half>(); DataCopy(x,xGm[p*TL],TL); qx.EnQue<half>(x);
        LocalTensor<half> xx=qxx.AllocTensor<half>(); DataCopy(xx,xxGm[p*TL],TL); qxx.EnQue<half>(xx);
        LocalTensor<half> a=qm1.AllocTensor<half>(); DataCopy(a,m1Gm[p*TL],TL); qm1.EnQue<half>(a);
        LocalTensor<half> b=qm2.AllocTensor<half>(); DataCopy(b,m2Gm[p*TL],TL); qm2.EnQue<half>(b);
    }
    __aicore__ inline void Compute(int32_t p){
        LocalTensor<half> x=qx.DeQue<half>(),xx=qxx.DeQue<half>(),a=qm1.DeQue<half>(),b=qm2.DeQue<half>();
        LocalTensor<half> t1=mid1.AllocTensor<half>(); Mul(t1,xx,a,TL); mid1.EnQue<half>(t1); qm1.FreeTensor(a);
        t1=mid1.DeQue<half>(); LocalTensor<half> yo1=oq1.AllocTensor<half>(); Add(yo1,x,t1,TL); oq1.EnQue<half>(yo1); mid1.FreeTensor(t1);
        LocalTensor<half> t2=mid2.AllocTensor<half>(); Mul(t2,xx,b,TL); mid2.EnQue<half>(t2); qm2.FreeTensor(b);
        t2=mid2.DeQue<half>(); LocalTensor<half> yo2=oq2.AllocTensor<half>(); Add(yo2,x,t2,TL); oq2.EnQue<half>(yo2); mid2.FreeTensor(t2);
        qx.FreeTensor(x); qxx.FreeTensor(xx);
    }
    __aicore__ inline void CopyOut(int32_t p){
        LocalTensor<half> yo1=oq1.DeQue<half>(); DataCopy(y1Gm[p*TL],yo1,TL); oq1.FreeTensor(yo1);
        LocalTensor<half> yo2=oq2.DeQue<half>(); DataCopy(y2Gm[p*TL],yo2,TL); oq2.FreeTensor(yo2);
    }
    TPipe pipe;
    TQue<QuePosition::VECIN,BN> qx,qxx,qm1,qm2;
    TQue<QuePosition::VECCALC,BN> mid1,mid2;
    TQue<QuePosition::VECOUT,BN> oq1,oq2;
    GlobalTensor<half> xGm,xxGm,m1Gm,m2Gm,y1Gm,y2Gm; int32_t tn;
};
extern "C" __global__ __aicore__ void rwkv_shift_mix2(GM_ADDR x,GM_ADDR xx,GM_ADDR m1,GM_ADDR m2,GM_ADDR y1,GM_ADDR y2,GM_ADDR workspace,GM_ADDR tiling){
    GET_TILING_DATA(t,tiling); K op; op.Init(x,xx,m1,m2,y1,y2,t.size); op.Process();
}
