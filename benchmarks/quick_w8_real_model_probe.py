import gc,json,statistics,time,torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from rwkv7_hf import enable_ascend
from rwkv7_ascend_model_quant import RWKV7FFNQuantSpec,quantize_rwkv7_ffn_model

P='/data/work/hf-adapter/.tmp/native_oracle_view';O='/data/work/quant-ascend/benchmarks/results/quick_w8_all_ffn_probe.json'
enable_ascend('npu:0',backend='eager');torch.manual_seed(20260724)
def load():
 m=AutoModelForCausalLM.from_pretrained(P,trust_remote_code=True,dtype=torch.float16,low_cpu_mem_usage=True).eval().to('npu:0')
 if type(m).__name__!='NativeRWKV7ForCausalLM':raise RuntimeError(f'unexpected AutoModel class {type(m)}')
 return m
def footprint(m):return sum(x.numel()*x.element_size() for x in list(m.parameters())+list(m.buffers()))
ids=torch.tensor([[1,2,3,4]],device='npu',dtype=torch.long)
def trace(m,steps=8,forced=None):
 with torch.inference_mode():
  out=m(ids,use_cache=True,logits_to_keep=1);past=out.past_key_values;ls=[];ts=[]
  for i in range(steps):
   ls.append(out.logits[:,-1].float().cpu());tok=torch.argmax(out.logits[:,-1:],-1) if forced is None else torch.tensor([[forced[i]]],device='npu');ts.append(int(torch.argmax(out.logits[:,-1:],-1).item()));out=m(tok,past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
 return ls,ts
fixed=[torch.tensor([[100+i]],device='npu') for i in range(24)]
def once(m):
 with torch.inference_mode():
  out=m(ids,use_cache=True,logits_to_keep=1);past=out.past_key_values;torch.npu.synchronize();t=time.perf_counter()
  for tok in fixed:out=m(tok,past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
  torch.npu.synchronize();return time.perf_counter()-t
dense=load();torch.npu.synchronize();dense_alloc=torch.npu.memory_allocated();dense_bytes=footprint(dense);dl,dt=trace(dense)
candidate=load();torch.npu.synchronize();spec=RWKV7FFNQuantSpec(bit=8,admitted_rows=tuple(range(1,29)))
t=time.perf_counter();report=quantize_rwkv7_ffn_model(candidate,spec,admission_scope='experiment',allow_unverified_experiment=True);torch.npu.synchronize();quant_s=time.perf_counter()-t
for rec in report.projections:
 m=candidate.get_submodule(rec.module_path); raw=m.bind_npu_fastpath(1,scope='experiment');k=m.in_features;n=m.out_features
 def nd(x,raw=raw,k=k,n=n):
  shape=x.shape[:-1];return raw(x.reshape(-1,k)).reshape(*shape,n)
 m.forward=nd
ql,qt=trace(candidate,forced=dt)
once(dense);once(candidate);pairs=[]
for i in range(7):
 if i%2==0:d=once(dense);q=once(candidate)
 else:q=once(candidate);d=once(dense)
 pairs.append({'order':'dq' if i%2==0 else 'qd','dense_s':d,'quant_s':q,'speedup':d/q})
cos=[];nrmse=[];kl=[];top=[]
for a,b in zip(dl,ql):
 cos.append(F.cosine_similarity(a.flatten(),b.flatten(),dim=0).item());nrmse.append(float((a-b).square().mean().sqrt()/a.square().mean().sqrt()));pa=F.softmax(a,-1);kl.append(float((pa*(F.log_softmax(a,-1)-F.log_softmax(b,-1))).sum()));ia=set(torch.topk(a,20).indices.flatten().tolist());ib=set(torch.topk(b,20).indices.flatten().tolist());top.append(len(ia&ib)/20)
quant_bytes=footprint(candidate);del dense;gc.collect();torch.npu.empty_cache();torch.npu.synchronize();quant_alloc=torch.npu.memory_allocated()
row={'bit':8,'scope':'quick_probe_not_production','layers':'all','projections':['key','value'],'replaced':len(report.projections),'quantize_s':quant_s,'dense_bytes':dense_bytes,'quant_bytes':quant_bytes,'payload_ratio':quant_bytes/dense_bytes,'dense_alloc':dense_alloc,'quant_single_alloc':quant_alloc,'hbm_ratio':quant_alloc/dense_alloc,'pairs':pairs,'speedup':statistics.median(x['dense_s'] for x in pairs)/statistics.median(x['quant_s'] for x in pairs),'min_cos':min(cos),'max_nrmse':max(nrmse),'max_kl':max(kl),'min_top20':min(top),'dense_tokens':dt,'quant_argmax_on_dense_path':qt,'greedy_equal':dt==qt}
open(O,'w').write(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
