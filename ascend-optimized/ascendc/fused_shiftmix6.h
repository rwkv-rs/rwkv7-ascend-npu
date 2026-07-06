// aclnn fused shift-mix helper: y_i = h + (x_prev - h)*mix_i for 6 mix vectors.
#pragma once
#include <torch/extension.h>
#include <vector>
#include "acl/acl.h"
#include "aclnn_rwkv_shift_mix6.h"
static inline aclDataType _acl_d(at::ScalarType t){return t==at::kFloat?ACL_FLOAT:ACL_FLOAT16;}
static inline aclTensor* _acl_t(const at::Tensor& x){
    auto c=x.contiguous(); int n=c.dim();
    std::vector<int64_t> sh(n),st(n);
    for(int i=0;i<n;i++){sh[i]=c.size(i);st[i]=c.stride(i);}
    return aclCreateTensor(sh.data(),(uint64_t)n,_acl_d(c.scalar_type()),st.data(),(int64_t)n,ACL_FORMAT_ND,nullptr,(uint64_t)0,c.data_ptr());
}
static inline std::vector<at::Tensor> fused_shiftmix6(const at::Tensor& h, const at::Tensor& xp,
        const at::Tensor& mr,const at::Tensor& mw,const at::Tensor& mk,const at::Tensor& mv,
        const at::Tensor& ma,const at::Tensor& mg){
    auto xx = xp - h;
    int64_t B=h.size(0), hidden=h.size(1);
    auto ex=[&](const at::Tensor& m){ return m.numel()==hidden ? m.view({1,hidden}).expand({B,hidden}).contiguous() : m.contiguous(); };
    auto mre=ex(mr),mwe=ex(mw),mke=ex(mk),mve=ex(mv),mae=ex(ma),mge=ex(mg);
    auto yr=at::empty_like(h),yw=at::empty_like(h),yk=at::empty_like(h),yv=at::empty_like(h),ya=at::empty_like(h),yg=at::empty_like(h);
    aclTensor *ha=_acl_t(h),*xxa=_acl_t(xx),*mra=_acl_t(mre),*mwa=_acl_t(mwe),*mka=_acl_t(mke),*mva=_acl_t(mve),*maa=_acl_t(mae),*mga=_acl_t(mge);
    aclTensor *yra=_acl_t(yr),*ywa=_acl_t(yw),*yka=_acl_t(yk),*yva=_acl_t(yv),*yaa=_acl_t(ya),*yga=_acl_t(yg);
    uint64_t ws=0; aclOpExecutor* exs=nullptr;
    aclnnStatus s1=aclnnRwkvShiftMix6GetWorkspaceSize(ha,xxa,mra,mwa,mka,mva,maa,mga,yra,ywa,yka,yva,yaa,yga,&ws,&exs);
    void* w=nullptr; if(ws>0){aclrtMalloc(&w,ws,ACL_MEM_MALLOC_NORMAL_ONLY);}
    aclrtStream st=nullptr; aclrtCtxGetCurrentDefaultStream(&st);
    aclnnStatus s2=aclnnRwkvShiftMix6(w,ws,exs,st)
    for(aclTensor* a:{ha,xxa,mra,mwa,mka,mva,maa,mga,yra,ywa,yka,yva,yaa,yga}) aclDestroyTensor(a);
    return {yr,yw,yk,yv,ya,yg};
}
