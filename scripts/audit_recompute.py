"""Independent recomputation of the headline numbers: plain pandas/numpy, no imports from
cityshift.analysis. Usage: python scripts/audit_recompute.py <evals dir>
(the per-scenario tables are attached to the GitHub release)."""
import json, sys
import numpy as np, pandas as pd
E=(sys.argv[1] if len(sys.argv) > 1 else "evals").rstrip("/") + "/"
CITIES=["austin","dearborn","miami","palo-alto","pittsburgh","washington-dc"]
A=pd.read_parquet(E+'ALL.parquet')
res=json.load(open(E+'results.json')); s2=json.load(open(E+'stage2_results.json'))
def sm(df,m): return df[[f's{i}_{m}' for i in range(3)]].mean(1).to_numpy()
# H1
rel={}
for c in CITIES:
    L=pd.read_parquet(E+f'LOCO-{c}.parquet'); assert (L.scenario_id.values==A.scenario_id.values).all()
    m=(A.city==c).to_numpy()
    rel[c]=sm(L,'miss')[m].mean()/sm(A,'miss')[m].mean()-1
print('H1 pooled recomputed', np.mean(list(rel.values())), 'reported', res['H1']['pooled_rel_change'])
print('H1 per fold max abs diff', max(abs(rel[c]-res['H1']['per_fold'][c]) for c in CITIES))
# H2: CF with independent implementation (keep 80% lowest score)
def cf(miss,score,err):
    n=int(round(.8*len(miss)))
    k=lambda v: np.argsort(v,kind='stable')[:n]
    base=miss.mean(); return (base-miss[k(score)].mean())/(base-miss[k(err)].mean())
cfs={}; cfi={}
for c in CITIES:
    L=pd.read_parquet(E+f'LOCO-{c}.parquet'); m=(L.city==c).to_numpy()
    cfs[c]=cf(sm(L,'miss')[m],L.disagree.to_numpy()[m],sm(L,'min_fde')[m])
    cfi[c]=cf(sm(L,'miss')[~m],L.disagree.to_numpy()[~m],sm(L,'min_fde')[~m])
print('H2 pooled recomputed', np.mean(list(cfs.values())), 'reported', res['H2']['pooled_CF_U1_heldout'])
print('H3 pooled recomputed', np.mean([cfs[c]-cfi[c] for c in CITIES]), 'reported', res['H3']['pooled_CF_diff_heldout_minus_indist'])
# H4
D=pd.read_parquet(E+'stage2_closedloop.parquet')
fa=D[[f'ALL_s{i}_failure' for i in range(3)]].astype(float).mean(1); fl=D[[f'LOCO_s{i}_failure' for i in range(3)]].astype(float).mean(1)
r4={c:fl[D.city==c].mean()/fa[D.city==c].mean()-1 for c in CITIES}
print('H4 pooled recomputed', np.mean(list(r4.values())), 'reported', s2['H4']['pooled_rel_change'])
print('H4 per fold', {c:round(v,4) for c,v in r4.items()})
print('n val scenarios', len(A), 'stage2 rows', len(D), 'ALL failures seed-avg', fa.sum())
# PCs
N=pd.read_parquet(E+'NOMAP.parquet'); print('PC1 recomputed', N.s0_miss.mean()/A.s0_miss.mean()-1, 'reported', res['PC1']['rel'])
print('PC3 recomputed static/ALL collision', D.static_collision.mean(), D[[f'ALL_s{i}_collision' for i in range(3)]].astype(float).mean(1).mean(), 'LOG', D.log_collision.mean())
