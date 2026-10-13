# NEW (final report): independent read-only recomputation of key numbers from saved outputs. Run: PYTHONPATH=/home/chi/outfitmatch-research/step0/pylibs python3 verify_recompute.py
import numpy as np, sys
from scipy.stats import rankdata
R='/home/chi/outfitmatch-research/step0/'
sc=np.load(R+'runs/exp2/scores.npz'); tab=np.load(R+'results/exp1_instance_table_taskB.npz',allow_pickle=True)
roles=["val_id","test_id","test_ud","test_2014","test_2015","test_2017"]
sm=lambda l:(lambda e:e/e.sum(1,keepdims=True))(np.exp(l-l.max(1,keepdims=True)))
P={s:{r:sm(sc[f's{s}/{r}/logits']) for r in roles} for s in range(5)}
PE={r:np.mean([P[s][r] for s in range(5)],0) for r in roles}
y={r:tab[f'{r}/label'] for r in roles}; us={r:tab[f'{r}/user_id'] for r in roles}
def auroc(u,e):  # positive=error, score=uncertainty, ties=0.5
    rk=rankdata(u); n1=e.sum(); n0=len(e)-n1
    return (rk[e==1].sum()-n1*(n1+1)/2)/(n1*n0)
ent=lambda p:-(p*np.log(p+1e-12)).sum(1)
def ece(p,yy,bins=15):
    c=p.max(1); ok=(p.argmax(1)==yy).astype(float); o=np.argsort(c,kind='mergesort'); e=0
    for ch in np.array_split(o,bins): e+=len(ch)/len(yy)*abs(ok[ch].mean()-c[ch].mean())
    return e
print("set n users effusers | acc0 accE acc5 | err0")
for r in roles:
    n=len(y[r]); _,c=np.unique(us[r],return_counts=True); eff=c.sum()**2/(c**2).sum()
    a=[(P[s][r].argmax(1)==y[r]).mean()*100 for s in range(5)]
    print(f"{r:10s} {n} {len(c)} {eff:.0f} | {a[0]:.1f} {(PE[r].argmax(1)==y[r]).mean()*100:.1f} {np.mean(a):.1f}")
print("\nAUROC (S-family, seed0 errors)")
for r in roles[1:]:
    p=P[0][r]; e=(p.argmax(1)!=y[r]).astype(int); pe=PE[r]
    k=sc[f's0/{r}/knn10']; mi=ent(pe)-np.mean([ent(P[s][r]) for s in range(5)],0)
    print(r,'msp',round(auroc(-p.max(1),e),3),'knn',round(auroc(k,e),3),'ens_msp',round(auroc(-pe.max(1),e),3),'ens_mi',round(auroc(mi,e),3))
print("\nAUROC (E-family)")
for r in roles[1:]:
    pe=PE[r]; e=(pe.argmax(1)!=y[r]).astype(int); mi=ent(pe)-np.mean([ent(P[s][r]) for s in range(5)],0)
    print(r,'ens_msp',round(auroc(-pe.max(1),e),3),'ens_mi',round(auroc(mi,e),3),'msp0',round(auroc(-P[0][r].max(1),e),3))
print("\nmean5 MSP AUROC, ECE(array_split), NLL, conf-acc")
for r in roles:
    au=[auroc(-P[s][r].max(1),(P[s][r].argmax(1)!=y[r]).astype(int)) for s in range(5)]
    ec=[ece(P[s][r],y[r]) for s in range(5)]
    nl=[-np.log(P[s][r][np.arange(len(y[r])),y[r]]+1e-12).mean() for s in range(5)]
    ca=[(P[s][r].max(1).mean()-(P[s][r].argmax(1)==y[r]).mean())*100 for s in range(5)]
    print(r,'AUROC',round(np.mean(au),3),'ECE',round(np.mean(ec),3),'NLL',round(np.mean(nl),3),'conf-acc',round(np.mean(ca),1))
print("\nfixed val_id threshold (msp seed0): coverage / risk accepted")
u=-P[0]['val_id'].max(1); tau=np.sort(u)[int(np.ceil(0.8*len(u)))-1]
for r in roles:
    p=P[0][r]; uu=-p.max(1); acc=uu<=tau; e=(p.argmax(1)!=y[r])
    print(r,round(acc.mean()*100,1),round(e[acc].mean()*100,1))
print("\nrisk@80 msp seed0 (own ranking)")
for r in roles:
    p=P[0][r]; uu=-p.max(1); e=(p.argmax(1)!=y[r]); o=np.argsort(uu,kind='mergesort'); k=int(round(0.8*len(e)))
    print(r,round(e[o[:k]].mean()*100,1),'base',round(e.mean()*100,1))
print("\nHCE seed-avg (msp, per-seed val_id tau) & coverage")
for r in roles:
    h=[];cv=[]
    for s in range(5):
        uv=-P[s]['val_id'].max(1); t=np.sort(uv)[int(np.ceil(0.8*len(uv)))-1]
        p=P[s][r]; a=(-p.max(1))<=t; e=(p.argmax(1)!=y[r]); h.append(e[a].mean()*100); cv.append(a.mean()*100)
    print(r,round(np.mean(h),1),round(np.mean(cv),1))
