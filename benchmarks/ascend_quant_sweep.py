#!/usr/bin/env python3
import json, platform, statistics, time
import torch
import torch_npu

DEVICE='npu:0'
DTYPE=torch.float16
WARMUP=30
ITERS=150
ROUNDS=5
W8_MS=list(range(1,65))+[96,128,256]
W4_MS=list(range(1,17))+[24,32,48,64]
SHAPES=[(4096,16384),(16384,4096)]
torch.manual_seed(20260724)

def sync(): torch.npu.synchronize()
def timed(fn):
    for _ in range(WARMUP): fn()
    sync()
    samples=[]
    for _ in range(ROUNDS):
        t=time.perf_counter_ns()
        for _ in range(ITERS): fn()
        sync()
        samples.append((time.perf_counter_ns()-t)/1e6/ITERS)
    return statistics.median(samples), samples

def cos(a,b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(),b.float().flatten(),dim=0).item()

def bytes_of(t): return t.numel()*t.element_size()

print(json.dumps({'kind':'environment','torch':torch.__version__,'torch_npu':torch_npu.__version__,
                  'device':torch.npu.get_device_name(0),'python':platform.python_version(),
                  'warmup':WARMUP,'iters':ITERS,'rounds':ROUNDS}),flush=True)
for k,n in SHAPES:
    w=torch.randn(n,k,device=DEVICE,dtype=DTYPE)/(k**.5)
    s8=(w.float().abs().amax(1)/127).clamp_min(1e-8).half()
    q8=torch.round(w.float()/s8[:,None]).clamp(-127,127).to(torch.int8).t().contiguous()
    g=128
    wg=w.reshape(n,k//g,g)
    s4=(wg.float().abs().amax(2)/7).clamp_min(1e-8).half()
    q4=torch.round(wg.float()/s4[:,:,None]).clamp(-8,7).int().reshape(n,k)
    p4=torch_npu.npu_convert_weight_to_int4pack(q4.t().contiguous())
    st4=s4.t().contiguous(); off4=torch.zeros_like(st4)
    print(json.dumps({'kind':'storage','K':k,'N':n,'fp16_bytes':bytes_of(w),
                      'w8_bytes':bytes_of(q8)+bytes_of(s8),
                      'w4_bytes':bytes_of(p4)+bytes_of(st4)+bytes_of(off4),
                      'w4_packed_shape':list(p4.shape),'w4_packed_dtype':str(p4.dtype),
                      'w8_ratio':(bytes_of(q8)+bytes_of(s8))/bytes_of(w),
                      'w4_ratio':(bytes_of(p4)+bytes_of(st4)+bytes_of(off4))/bytes_of(w)}),flush=True)
    for m in sorted(set(W8_MS+W4_MS)):
        x=torch.randn(m,k,device=DEVICE,dtype=DTYPE)
        def fp(): return torch.matmul(x,w.t())
        ref=fp()
        if m in W8_MS:
            def w8(): return torch_npu.npu_weight_quant_batchmatmul(x,q8,s8)
            out=w8(); sync(); fp_ms,fp_samples=timed(fp); q_ms,q_samples=timed(w8)
            print(json.dumps({'kind':'result','bit':8,'M':m,'K':k,'N':n,'fp16_ms':fp_ms,
                 'quant_ms':q_ms,'speedup':fp_ms/q_ms,'cosine':cos(ref,out),
                 'fp16_samples_ms':fp_samples,'quant_samples_ms':q_samples}),flush=True)
        if m in W4_MS:
            def w4(): return torch_npu.npu_weight_quant_batchmatmul(x,p4,st4,off4,None,None,None,g,1)
            out=w4(); sync(); fp_ms,fp_samples=timed(fp); q_ms,q_samples=timed(w4)
            print(json.dumps({'kind':'result','bit':4,'group_size':g,'M':m,'K':k,'N':n,
                 'fp16_ms':fp_ms,'quant_ms':q_ms,'speedup':fp_ms/q_ms,'cosine':cos(ref,out),
                 'fp16_samples_ms':fp_samples,'quant_samples_ms':q_samples}),flush=True)
        del x,ref
    del w,s8,q8,wg,s4,q4,p4,st4,off4
    torch.npu.empty_cache()
