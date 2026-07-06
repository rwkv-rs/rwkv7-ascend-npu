#define ACL_API
#include <torch/extension.h>
#include <vector>
#include "acl/acl.h"
#include "aclnn_rwkv_shift_mix.h"
static aclDataType to_acl_dtype(at::ScalarType t){
    if(t==at::kHalf) return ACL_FLOAT16; if(t==at::kFloat) return ACL_FLOAT;
    if(t==at::kInt) return ACL_INT32; if(t==at::kLong) return ACL_INT64; return ACL_FLOAT16;
}
static aclTensor* to_acl_tensor(const at::Tensor& x){
    auto c=x.contiguous(); int n=c.dim();
    std::vector<int64_t> shape(n), stride(n);
    for(int i=0;i<n;i++){shape[i]=c.size(i); stride[i]=c.stride(i);}
    return aclCreateTensor(shape.data(),(uint64_t)n,to_acl_dtype(c.scalar_type()),stride.data(),(int64_t)n,ACL_FORMAT_ND,nullptr,(uint64_t)0,c.data_ptr());
}
at::Tensor rwkv_shiftmix(at::Tensor x, at::Tensor xp, at::Tensor xr){
    x=x.contiguous(); xp=xp.contiguous(); xr=xr.contiguous();
    auto y=at::empty_like(x);
    aclTensor *xa=to_acl_tensor(x), *pa=to_acl_tensor(xp), *ra=to_acl_tensor(xr), *ya=to_acl_tensor(y);
    uint64_t wsSize=0; aclOpExecutor* ex=nullptr;
    aclnnStatus s1=aclnnRwkvShiftMixGetWorkspaceSize(xa,pa,ra,ya,&wsSize,&ex);
    TORCH_CHECK(s1==0,"GetWorkspaceSize failed: ",(int)s1);
    void* ws=nullptr;
    if(wsSize>0){aclError rc=aclrtMalloc(&ws,wsSize,ACL_MEM_MALLOC_NORMAL_ONLY); TORCH_CHECK(rc==ACL_SUCCESS,"malloc failed");}
    aclrtStream stream=nullptr; aclrtCtxGetCurrentDefaultStream(&stream);
    aclnnStatus s2=aclnnRwkvShiftMix(ws,wsSize,ex,stream);
    TORCH_CHECK(s2==0,"aclnnRwkvShiftMix failed: ",(int)s2);
    aclDestroyTensor(xa);aclDestroyTensor(pa);aclDestroyTensor(ra);aclDestroyTensor(ya);
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.def("rwkv_shiftmix",&rwkv_shiftmix,"y=x+(xp-x)*xr"); }
