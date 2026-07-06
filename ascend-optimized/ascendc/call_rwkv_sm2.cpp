#define ACL_API
#include <torch/extension.h>
#include <vector>
#include "acl/acl.h"
#include "aclnn_rwkv_shift_mix2.h"
static aclDataType td(at::ScalarType t){if(t==at::kHalf)return ACL_FLOAT16;if(t==at::kFloat)return ACL_FLOAT;return ACL_FLOAT16;}
static aclTensor* mt(const at::Tensor& x){auto c=x.contiguous();int n=c.dim();std::vector<int64_t>s(n),st(n);for(int i=0;i<n;i++){s[i]=c.size(i);st[i]=c.stride(i);}return aclCreateTensor(s.data(),(uint64_t)n,td(c.scalar_type()),st.data(),(int64_t)n,ACL_FORMAT_ND,nullptr,(uint64_t)0,c.data_ptr());}
std::vector<at::Tensor> rwkv_sm2(at::Tensor x,at::Tensor xx,at::Tensor m1,at::Tensor m2){
    x=x.contiguous();xx=xx.contiguous();m1=m1.contiguous();m2=m2.contiguous();
    auto y1=at::empty_like(x),y2=at::empty_like(x);
    aclTensor *xa=mt(x),*xa2=mt(xx),*ma=mt(m1),*mb=mt(m2),*ya1=mt(y1),*ya2=mt(y2);
    uint64_t ws=0;aclOpExecutor* ex=nullptr;
    aclnnStatus s1=aclnnRwkvShiftMix2GetWorkspaceSize(xa,xa2,ma,mb,ya1,ya2,&ws,&ex);
    TORCH_CHECK(s1==0,"ws failed",(int)s1);
    void* w=nullptr;if(ws>0)aclrtMalloc(&w,ws,ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtStream st=nullptr;aclrtCtxGetCurrentDefaultStream(&st);
    aclnnStatus s2=aclnnRwkvShiftMix2(w,ws,ex,st);TORCH_CHECK(s2==0,"run failed",(int)s2);
    aclDestroyTensor(xa);aclDestroyTensor(xa2);aclDestroyTensor(ma);aclDestroyTensor(mb);aclDestroyTensor(ya1);aclDestroyTensor(ya2);
    return {y1,y2};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("rwkv_sm2",&rwkv_sm2);}
