"""Monte-Carlo population drift -dL/dm of the single-feature softmax-attention models (pinned Gamma=1, sigma_2, d=64) used to design exp 13.
prod: logits = beta * s_i s_q (exp 12 form; drift negative everywhere -> the model cannot learn);  rbf: logits = -beta (s_i - s_q)^2 (Nadaraya-Watson
form; same qualitative pattern as the linear model: negative at m_0 for small N, non-negative for N=1024).  Output: results/exp13/drift_mc.txt.
Run from /home/user/ml:  python3 scripts/softmax_drift.py > results/exp13/drift_mc.txt   (about 25 min on 4 threads)"""
import torch, math, sys
torch.set_num_threads(4); torch.manual_seed(0)
d=64
def sig(z): return (z*z-1)/math.sqrt(2)
def drift(beta, N, m, Gamma=1.0, reps=8, B=2048, mode="prod"):
    v=torch.zeros(d); v[0]=1.0
    u=torch.zeros(d); u[1]=1.0
    w=(m*v+math.sqrt(1-m*m)*u).requires_grad_(True)
    tot=0.0; tot2=0.0
    for _ in range(reps):
        x=torch.randn(B,N+1,d); c=torch.randn(B,1)
        y=c*sig(x@v)                      # B x (N+1)
        s=sig(x@w)                        # B x (N+1)
        if mode=="prod": logits=beta*s[:,:N]*s[:,N:N+1]
        elif mode=="rbf": logits=-beta*(s[:,:N]-s[:,N:N+1])**2
        a=torch.softmax(logits,dim=1)
        yhat=Gamma*(a*y[:,:N]).sum(1)
        L=((yhat-y[:,N])**2).mean()
        g,=torch.autograd.grad(L,w)
        t=(v-m*w.detach())/math.sqrt(1-m*m)
        val=-(g@t).item()/math.sqrt(1-m*m); tot+=val; tot2+=val*val
    mean=tot/reps; se=math.sqrt(max(tot2/reps-mean*mean,0)/reps)
    return mean,se
for mode in ("prod","rbf"):
  for beta in (0.1,0.3,1.0,3.0):
    for N in (16,1024):
        row=[]
        for m in (0.125,0.25,0.5):
            mu,se=drift(beta,N,m,Gamma=1.0,mode=mode); row.append(f"m={m}: {mu:+.2e}±{se:.0e}")
        print(f"{mode} beta={beta} N={N}: "+"  ".join(row), flush=True)
print("=== precise, rbf ===", flush=True)
for beta in (0.3,1.0):
    for N in (16,64,1024):
        row=[]
        for m in (0.125,0.18,0.25):
            mu,se=drift(beta,N,m,Gamma=1.0,reps=48,B=4096,mode="rbf"); row.append(f"m={m}: {mu:+.2e}±{se:.0e}")
        print(f"rbf beta={beta} N={N}: "+"  ".join(row), flush=True)
