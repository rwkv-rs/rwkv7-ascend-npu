import os
os.environ.setdefault("RWKV7_NATIVE_MODEL","1"); os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
import sys; sys.path.insert(0, os.environ.get("RWKV7_HF_PATH", "/root/rwkv7-ascend"))
import torch, torch_npu, time
from torch.utils.cpp_extension import load
from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
DEV=os.environ.get("RWKV7_DEVICE", "npu:0"); VOCAB=65536

def _extract(base, hidden):
    rw,kw,vw,ow,fkw,fvw=[],[],[],[],[],[]
    w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l=[],[],[],[],[],[],[],[]
    w2b,a2b,v2b=[],[],[]
    xr,xw,xk,xv,xa,xg=[],[],[],[],[],[]
    kk_l,ka_l,rk_l,gnw,gnb,fxk=[],[],[],[],[],[]
    anw,anb,fnw,fnb,pnw,pnb=[],[],[],[],[],[]
    for layer in base.layers:
        a=layer.attn; f=layer.ffn
        rw.append(a.r_proj.weight.data); kw.append(a.k_proj.weight.data)
        vw.append(a.v_proj.weight.data); ow.append(a.o_proj.weight.data)
        fkw.append(f.key.weight.data); fvw.append(f.value.weight.data)
        w0l.append(a.w_lora.lora[0].weight.data); w2l.append(a.w_lora.lora[2].weight.data)
        a0l.append(a.a_lora.lora[0].weight.data); a2l.append(a.a_lora.lora[2].weight.data)
        g0l.append(a.g_lora.lora[0].weight.data); g2l.append(a.g_lora.lora[2].weight.data)
        if hasattr(a,"v_lora") and a.v_lora is not None:
            v0l.append(a.v_lora.lora[0].weight.data); v2l.append(a.v_lora.lora[2].weight.data); v2b.append(a.v_lora.lora[2].bias.data)
        else:
            v0l.append(w0l[-1]); v2l.append(w2l[-1]); v2b.append(torch.zeros(hidden,dtype=torch.float16,device=DEV))
        w2b.append(a.w_lora.lora[2].bias.data); a2b.append(a.a_lora.lora[2].bias.data)
        xr.append(a.x_r.data.reshape(-1)); xw.append(a.x_w.data.reshape(-1))
        xk.append(a.x_k.data.reshape(-1)); xv.append(a.x_v.data.reshape(-1))
        xa.append(a.x_a.data.reshape(-1)); xg.append(a.x_g.data.reshape(-1))
        kk_l.append(a.k_k.data.reshape(-1)); ka_l.append(a.k_a.data.reshape(-1)); rk_l.append(a.r_k.data.reshape(-1))
        gnw.append(a.g_norm.weight.data); gnb.append(a.g_norm.bias.data if a.g_norm.bias is not None else torch.zeros(hidden,dtype=torch.float16,device=DEV)); fxk.append(f.x_k.data.reshape(-1))
        anw.append(layer.attn_norm.weight.data); anb.append(layer.attn_norm.bias.data)
        fnw.append(layer.ffn_norm.weight.data); fnb.append(layer.ffn_norm.bias.data)
        if hasattr(layer,"pre_norm"): pnw.append(layer.pre_norm.weight.data); pnb.append(layer.pre_norm.bias.data)
        else: pnw.append(torch.ones(hidden,dtype=torch.float16,device=DEV)); pnb.append(torch.zeros(hidden,dtype=torch.float16,device=DEV))
    fnorm_w=base.norm.weight.data; fnorm_b=base.norm.bias.data; lm_w=base.lm_head.weight.data if hasattr(base,"lm_head") else None
    W=(rw,kw,vw,ow,fkw,fvw,w0l,w2l,a0l,a2l,g0l,g2l,v0l,v2l,w2b,a2b,v2b,xr,xw,xk,xv,xa,xg,kk_l,ka_l,rk_l,gnw,gnb,fxk,anw,anb,fnw,fnb,pnw,pnb)
    return W, lm_w, fnorm_w, fnorm_b

class RWKV7Engine:
    def __init__(self, model_dir, H=12, N=64, L=12):
        self.H,self.N,self.L=H,N,L; self.hidden=H*N
        cfg=RWKV7Config(vocab_size=VOCAB,hidden_size=self.hidden,num_hidden_layers=L,num_heads=H,head_dim=N,intermediate_size=4*self.hidden)
        print("[engine] loading %s (H%d N%d L%d)"%(model_dir,H,N,L),flush=True)
        self.model=NativeRWKV7ForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16).to(DEV).eval()
        self.base=self.model.model
        self.lm_w_m=self.model.lm_head.weight.data
        self.W,_,self.fnorm_w,self.fnorm_b=_extract(self.base,self.hidden)
        self.mod=load(name="rwkv7_ascend_v3",sources=[os.environ.get("RWKV7_CPP_PATH", "/root/rwkv7_ascend_v3.cpp")],verbose=False,extra_cflags=["-O3","-std=c++17"])
        print("[engine] ready",flush=True)
    def _new_state(self,B):
        return (torch.zeros(self.L,B,self.H,self.N,self.N,dtype=torch.float32,device=DEV),
                torch.zeros(self.L,B,self.hidden,dtype=torch.float16,device=DEV),
                torch.zeros(self.L,B,self.hidden,dtype=torch.float16,device=DEV),
                torch.zeros(B,self.hidden,dtype=torch.float16,device=DEV))
    def _step(self, tokens, state):
        sa,xp,xf,vf=state
        emb=self.base.embeddings(torch.tensor(tokens,device=DEV))
        return self.mod.rwkv7_decode_full(emb,*self.W,sa,xp,xf,vf,self.H,self.N,self.lm_w_m,self.fnorm_w,self.fnorm_b)
    def generate(self, prompts, max_new=16):
        B=len(prompts)
        states=[]; first=[]
        for p in prompts:
            st=self._new_state(1); out=None
            for t in (p if p else [0]):
                out=self._step([t],st)
            states.append(st); first.append(int(out.reshape(-1,VOCAB).argmax(-1)[-1]))
        state=(torch.cat([s[0] for s in states],dim=1),torch.cat([s[1] for s in states],dim=1),
               torch.cat([s[2] for s in states],dim=1),torch.cat([s[3] for s in states],dim=0))
        gen=[[first[i]] for i in range(B)]; nxt=first
        for _ in range(max_new-1):
            out=self._step(nxt,state)
            nxt=out.argmax(-1).tolist()
            for i in range(B): gen[i].append(nxt[i])
        return gen

if __name__=="__main__":
    eng=RWKV7Engine("/root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf")
    prompts=[list(range(16)),list(range(8)),list(range(4)),list(range(12))]
    gb=eng.generate(prompts,max_new=8)
    gs=[eng.generate([p],max_new=8)[0] for p in prompts]
    print("=== correctness: batched == per-seq ===",flush=True)
    allok=True
    for i,p in enumerate(prompts):
        ok = gb[i]==gs[i]; allok &= ok
        print("seq%d (len%d): batch=%s single=%s %s"%(i,len(p),gb[i],gs[i],"OK" if ok else "DIFF"),flush=True)
    print("ALL_MATCH" if allok else "MISMATCH",flush=True)
    # throughput: B=64 random seqs decode
    B=64
    rp=[list(range(16)) for _ in range(B)]
    for _ in range(3): eng.generate(rp,max_new=8)
    torch.npu.synchronize(); t0=time.time()
    eng.generate(rp,max_new=33)
    torch.npu.synchronize(); dt=time.time()-t0
    print("B=%d decode: %.0f aggregate tok/s (%.1fms/step)"%(B, B*32/dt*1000, dt/32*1000),flush=True)
