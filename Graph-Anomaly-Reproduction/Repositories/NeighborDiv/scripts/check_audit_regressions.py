"""Checks explicitly requested in the 2026-10-13 paper/code audit."""
from pathlib import Path
import sys,json
import numpy as np
import scipy.sparse as sp
PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent.parent
sys.path.insert(0,str(PROJECT))
from src.data import binary_adjacency
from src.neighbordiv import _full_pair_moments,_normalize_projected,score_graph
from src.heuristic_baselines import score_amen_ego,normalized_undirected,_score_amen_ego_native


def amen_equation_reference(a,x):
    """Direct scalar loops implementing original AMEN Sec. 4.2 / Eq. 4.6."""
    a=a.toarray().astype(float); d=a.sum(axis=1); two_m=d.sum()
    span=np.ptp(x,axis=0); attr=np.divide(x-x.min(axis=0),span,out=np.zeros_like(x),where=span>0)
    raw=np.zeros(len(x))
    for ego in range(len(x)):
        if d[ego]==0: continue
        c=np.union1d([ego],np.flatnonzero(a[ego])); outside=np.setdiff1d(np.arange(len(x)),c)
        vals=[]
        imin=-sum(d[p]*d[q]/two_m for p in c for q in c)
        for f in range(x.shape[1]):
            xi=sum((a[p,q]-d[p]*d[q]/two_m)*attr[p,f]*attr[q,f] for p in c for q in c)
            cut=sum(a[p,b]*(1-min(1.,d[p]*d[b]/two_m))*attr[p,f]*attr[b,f] for p in c for b in outside)
            internal=sum(a[p,q]*(1-min(1.,d[p]*d[q]/two_m))*attr[p,f]*attr[q,f] for p in c for q in c)
            vals.append((xi-imin)/(len(c)**2-imin)-(cut/(internal+cut) if internal+cut>0 else 0.))
        vals=np.array(vals); raw[ego]=-(np.linalg.norm(np.maximum(vals,0)) if (vals>0).any() else vals.max())
    return raw,attr


def main():
    rng=np.random.default_rng(707); checks=[]
    for d in (2,3,8,67,251):
        for case in ('random','constant','zero-row'):
            x=rng.normal(size=(d,8))
            if case=='constant': x[:]=x[0]
            if case=='zero-row': x[0]=0
            z,_=_normalize_projected(x); mean,var=_full_pair_moments(z)
            dots=np.einsum('ij,ij->i',z[np.triu_indices(d,1)[0]],z[np.triu_indices(d,1)[1]])
            assert abs(var-dots.var())<1e-12 and abs(mean-dots.mean())<1e-12
            if case=='constant': assert var==0
            checks.append(dict(check='moment-vs-direct',degree=d,case=case,error=float(abs(var-dots.var()))))
    raw=sp.csr_matrix(([1,1,1],([0,1,1],[1,1,2])),shape=(4,4))
    out=binary_adjacency(raw); inc=binary_adjacency(raw,'incoming'); union=binary_adjacency(raw,'undirected')
    assert (inc!=out.T).nnz==0 and (union!=union.T).nnz==0 and union.diagonal().sum()==0
    checks.append(dict(check='edge-policy',passed=True))
    aa=rng.random((13,13))<.27; np.fill_diagonal(aa,False); aa=np.logical_or(aa,aa.T)
    aa[-1,:]=False; aa[:,-1]=False; a=sp.csr_matrix(aa); x=rng.normal(size=(13,8))
    expected,attr=amen_equation_reference(a,x); _,actual,valid=score_amen_ego(a,x)
    assert np.max(abs(expected-actual))<1e-12
    _,partial,partial_valid=score_amen_ego(a,x,max_nodes=3)
    assert partial_valid.sum()==3 and np.array_equal(np.flatnonzero(partial_valid),np.flatnonzero(valid)[:3])
    undirected,degree,_=normalized_undirected(a)
    native=_score_amen_ego_native(undirected,attr,degree,degree.sum())
    assert np.max(abs(expected-native))<1e-12
    checks.append(dict(check='AMEN-equations-python-native-and-partial-mask',python_error=float(np.max(abs(expected-actual))),native_error=float(np.max(abs(expected-native))),partial_valid_count=int(partial_valid.sum())))
    for cap in (None,7):
        stream=score_graph(a,x,rank=8,max_pairs=cap,pair_batch_size=5)
        moments=score_graph(a,x,rank=8,max_pairs=cap,pair_batch_size=5,full_backend='moments')
        assert np.max(abs(stream.scores-moments.scores))<1e-10
        checks.append(dict(check='production-stream-vs-moments',max_pairs=cap,score_error=float(np.max(abs(stream.scores-moments.scores)))))
    target=ROOT/'results/NeighborDiv/protocol-audit-2026-10-13/regression_checks.json'
    target.write_text(json.dumps(dict(status='passed',checks=checks),indent=2))
    print('passed',len(checks),'checks',target,flush=True)

if __name__=='__main__': main()
