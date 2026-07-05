import os
os.environ.setdefault("RWKV7_NATIVE_MODEL","1"); os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
import sys; sys.path.insert(0,"/root/rwkv7-ascend")
import torch, torch_npu, time
from serve_engine import RWKV7Engine, DEV, VOCAB

def prefill_one(eng, tokens):
    st=eng._new_state(1); out=None
    for t in (tokens or [0]): out=eng._step([t], st)
    return st, int(out.reshape(-1,VOCAB).argmax(-1)[-1])

def decode_batch(eng, states, next_tokens):
    sa=torch.cat([s[0] for s in states],dim=1)
    xp=torch.cat([s[1] for s in states],dim=1)
    xf=torch.cat([s[2] for s in states],dim=1)
    vf=torch.cat([s[3] for s in states],dim=0)
    out=eng._step(next_tokens,(sa,xp,xf,vf))
    nt=out.argmax(-1).tolist()
    ns=[(sa[:,i:i+1],xp[:,i:i+1],xf[:,i:i+1],vf[i:i+1]) for i in range(len(states))]
    return ns, nt

class Seq:
    def __init__(self, tokens, max_new):
        self.prompt=tokens; self.max_new=max_new; self.gen=[]; self.done=False; self.state=None; self.next=None

class Scheduler:
    def __init__(self, eng): self.eng=eng; self.active=[]; self.tok_count=0; self.step_count=0
    def submit(self, tokens, max_new):
        s=Seq(tokens,max_new); s.state,s.next=prefill_one(self.eng,tokens); s.gen=[s.next]; self.tok_count+=1; self.active.append(s); return s
    def step(self):
        if not self.active: return False
        states=[s.state for s in self.active]; nexts=[s.next for s in self.active]
        ns,nt=decode_batch(self.eng,states,nexts)
        for i,s in enumerate(self.active):
            s.state=ns[i]; s.next=nt[i]; s.gen.append(nt[i]); self.tok_count+=1
            if len(s.gen)>=s.max_new: s.done=True
        self.active=[s for s in self.active if not s.done]; self.step_count+=1
        return True

if __name__=="__main__":
    eng=RWKV7Engine("/root/rwkv7-ascend/models/rwkv7-g1d-0.1b-hf")
    sch=Scheduler(eng)
    reqs=[(list(range(16)),8),(list(range(8)),5),(list(range(4)),10),(list(range(12)),6),(list(range(20)),7)]
    seqs=[sch.submit(t,mn) for t,mn in reqs]
    # mid-flight join: after 2 steps, add another
    sch.step(); sch.step()
    late=sch.submit(list(range(6)),9); seqs.append(late)
    while sch.step(): pass
    print("=== dynamic continuous-batch: each seq vs standalone ===",flush=True)
    allok=True
    for s in seqs:
        ref=eng.generate([s.prompt],max_new=s.max_new)[0]
        ok=s.gen==ref; allok&=ok
        print("len%d max_new%d: sched=%s standalone=%s %s"%(len(s.prompt),s.max_new,s.gen,ref,"OK" if ok else "DIFF"),flush=True)
    print("ALL_MATCH" if allok else "MISMATCH",flush=True)
    # throughput: 64 concurrent requests, max_new=32, measure decode only
    sch2=Scheduler(eng)
    for _ in range(64): sch2.submit(list(range(16)),32)
    for _ in range(3):  # warmup
        sch2=Scheduler(eng)
        for _ in range(64): sch2.submit(list(range(16)),32)
        while sch2.step(): pass
    sch3=Scheduler(eng)
    for _ in range(64): sch3.submit(list(range(16)),32)
    torch.npu.synchronize(); t0=time.time()
    while sch3.step(): pass
    torch.npu.synchronize(); dt=time.time()-t0
    print("serving 64 concurrent x32 new tokens: %d tokens in %.2fs = %.0f aggregate tok/s (decode-only)"%(sch3.tok_count,dt,sch3.tok_count/dt),flush=True)
