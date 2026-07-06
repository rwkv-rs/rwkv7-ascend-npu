"""Build the torch binding for aclnnRwkvWexp and verify correctness + speed."""
import os, time
os.environ["RWKV7_NATIVE_MODEL"]="1"; os.environ["TORCHDYNAMO_DISABLE"]="1"
# custom op runtime lib must be on the path
os.environ["LD_LIBRARY_PATH"] = "/usr/local/Ascend/cann-8.5.1/opp/vendors/customize/op_api/lib/:" + os.environ.get("LD_LIBRARY_PATH","")
import torch, torch_npu
import torch.nn.functional as F
from torch.utils.cpp_extension import load

CANN="/usr/local/Ascend/cann-8.5.1"
mod = load(name="call_rwkv_wexp", sources=["/root/ascendc/call_rwkv_wexp.cpp"],
           verbose=False, extra_cflags=["-O3","-std=c++17"],
           extra_include_paths=[f"{CANN}/include",
                                f"{CANN}/opp/vendors/customize/op_api/include",
                                os.path.join(os.path.dirname(torch_npu.__file__), "include")],
           extra_ldflags=[f"-L{CANN}/opp/vendors/customize/op_api/lib", "-lcust_opapi",
                          f"-L{CANN}/x86_64-linux/devlib", "-lascendcl",
                          f"-Wl,-rpath,{CANN}/opp/vendors/customize/op_api/lib",
                          f"-Wl,-rpath,{CANN}/x86_64-linux/devlib",
                          f"-Wl,-rpath,{CANN}/lib64",
                          "-Wl,--allow-shlib-undefined"])

dev="npu:0"
EXP_HALF=0.606531
torch.manual_seed(0)
x = torch.randn(8192, device=dev).half()

# correctness
y_op = mod.rwkv_wexp(x)
y_ref = torch.exp(-EXP_HALF * torch.sigmoid(x.float())).half()
cos = F.cosine_similarity(y_op.float().cpu().flatten().unsqueeze(0), y_ref.float().cpu().flatten().unsqueeze(0)).item()
mx = (y_op.float()-y_ref.float()).abs().max().item()
print("CORRECTNESS: cos=%.6f maxabs=%.6e" % (cos, mx), flush=True)

# speed: fused single op vs 2-op eager (sigmoid + mul + exp = 3 ops actually)
xb = torch.randn(8192, device=dev).half()
def fused(): return mod.rwkv_wexp(xb)
def eager(): return torch.exp(-EXP_HALF * torch.sigmoid(xb.float())).half()
for f in (fused, eager): f()
torch.npu.synchronize(); t0=time.time()
for _ in range(500): fused()
torch.npu.synchronize(); fused_ms=(time.time()-t0)/500*1000
torch.npu.synchronize(); t0=time.time()
for _ in range(500): eager()
torch.npu.synchronize(); eager_ms=(time.time()-t0)/500*1000
print("fused AscendC wexp: %.4f ms" % fused_ms, flush=True)
print("eager (3 ops):      %.4f ms  (%.2fx)" % (eager_ms, eager_ms/fused_ms), flush=True)
