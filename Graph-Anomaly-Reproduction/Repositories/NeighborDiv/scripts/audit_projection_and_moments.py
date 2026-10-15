"""Independent Full moment identity, numerical checks, SVD and input audit.

This is an audit backend, not a replacement for the streamed production code.
It visits every neighbor but does not sample neighbor pairs. Labels enter only
the last evaluation step. Both sample sizes and solvers are fixed in advance.
"""
from pathlib import Path
import sys, time, json, csv, hashlib
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent.parent
sys.path.insert(0, str(PROJECT))
from src.data import load_graph, binary_adjacency
from src.neighbordiv import _normalize_projected, _similarity_batches, _pair_mean_variance
from src.heuristic_baselines import calibrate_raw
OUT = ROOT/'results/NeighborDiv/protocol-audit-2026-10-13'


def full_moments(a, z):
    """Stable centered identity for all unordered-pair first/second moments.

    Let y_i=z_i-mu, a_i=y_i.mu, b_i=||y_i||^2, G=Y.T Y,
    M=d(d-1)/2. Var=( (d-2)sum(a_i^2)
    +(||G||_F^2-sum(b_i^2))/2-2sum(a_i*b_i) )/M
    -(sum(b_i)/(2M))^2. This needs O(d*r^2) work per node.
    """
    degree=np.diff(a.indptr); valid=degree>=2
    means=np.full(len(degree),np.nan); variance=means.copy()
    for i in np.flatnonzero(valid):
        v=z[a.indices[a.indptr[i]:a.indptr[i+1]]]; d=len(v)
        if d==2:
            means[i]=np.dot(v[0],v[1]); variance[i]=0.; continue
        mu=v.mean(axis=0); y=v-mu; linear=y.dot(mu)
        b=np.einsum('ij,ij->i',y,y); gram=y.T.dot(y); m=d*(d-1)/2
        qmean=-b.sum()/(2*m)
        numerator=(d-2)*np.dot(linear,linear)+(np.square(gram).sum()-np.dot(b,b))/2-2*np.dot(linear,b)
        means[i]=np.dot(mu,mu)+qmean
        variance[i]=max(0.,numerator/m-qmean*qmean)
    return means,variance,valid


def metric(y,s):
    return dict(AUROC=float(roc_auc_score(y,s)),AP=float(average_precision_score(y,s)))


def check_identity():
    rng=np.random.default_rng(404); rows=[]
    for d in (2,3,7,32,131):
        for case in ('random','constant','almost-constant','zero-row'):
            x=rng.normal(size=(d,8))
            if case=='constant': x[:]=x[0]
            if case=='almost-constant': x=x[0]+1e-8*x
            if case=='zero-row': x[0]=0
            z,_=_normalize_projected(x)
            a=sp.csr_matrix((np.ones(d),(np.zeros(d,dtype=int),np.arange(d))),shape=(d,d))
            # This synthetic CSR intentionally includes diagonal index 0:
            # full_moments operates on rows directly, not preprocessing.
            mean,var,_=full_moments(a,z)
            dots=z.dot(z.T)[np.triu_indices(d,1)]
            rows.append(dict(d=d,case=case,mean_error=float(abs(mean[0]-dots.mean())),variance_error=float(abs(var[0]-dots.var()))))
    assert max(r['variance_error'] for r in rows)<1e-12
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=check_identity(); result=[]; inputs=[]; benchmarks=[]
    config=json.loads((PROJECT/'configs/paper_experiments.json').read_text())
    for spec in config['datasets']:
        name=spec['name']; begin=time.perf_counter()
        graph=load_graph(ROOT/'data'/spec['path'],data_format=spec['format'],edge_policy=spec['edge_policy'])
        degree=np.diff(graph.adjacency.indptr)
        inputs.append(dict(dataset=name,policy=spec['edge_policy'],nodes=len(degree),features=graph.features.shape[1],anomalies=int(graph.labels.sum()),isolated=int((degree==0).sum()),degree_one=int((degree==1).sum()),valid=int((degree>=2).sum()),adjacency_nnz=int(graph.adjacency.nnz),feature_min=float(graph.features.min()),feature_max=float(graph.features.max()),feature_nonfinite=int((~np.isfinite(graph.features.data)).sum()),load_seconds=time.perf_counter()-begin))
        variants=[('randomized',0),('randomized',1),('arpack',0),('randomized32',0)]
        if name in ('T-Finance','Tolokers','Disney'): variants.append(('dense',0))
        reference_v=None; reference_z=None
        for solver,seed in variants:
            start=time.perf_counter()
            if solver=='dense':
                u,s,v=np.linalg.svd(graph.features.toarray(),full_matrices=False); projected=u[:,:8]*s[:8]
                s=s[:8]
            else:
                svd=TruncatedSVD(n_components=8,algorithm='randomized' if solver=='randomized32' else solver,n_iter=5,random_state=seed)
                projected=svd.fit_transform(graph.features.astype(np.float32) if solver=='randomized32' else graph.features); s=svd.singular_values_
                # This variant quantizes features/SVD, then accumulates in
                # float64. It is not per-dot-product float32 rounding.
                projected=np.asarray(projected,dtype=np.float64)
            z,zeros=_normalize_projected(projected); projected_at=time.perf_counter()
            mean,var,valid=full_moments(graph.adjacency,z); end=time.perf_counter()
            scores=calibrate_raw(np.nan_to_num(var),valid)
            row=dict(dataset=name,policy=spec['edge_policy'],solver=solver,seed=seed,projection_seconds=projected_at-start,full_moment_seconds=end-projected_at,zero_projected_rows=zeros,**metric(graph.labels,scores))
            if reference_v is None:
                reference_v=var; reference_z=z
                folder=ROOT/'results/NeighborDiv/paper-audit/main'/name/'full/seed-0'
                if name=='Cora': folder=OUT/'cora-freegad-full'
                if name=='T-Finance': folder=ROOT/'results/NeighborDiv/paper-audit-tfinance-full-retry/main/T-Finance/full/seed-0'
                if name in ('Disney','Questions','Tolokers'):
                    old=np.load(OUT/(name+'-'+spec['edge_policy']+'.npz'))
                    oldv=old['diversity']; olds=old['scores']
                else:
                    old=np.genfromtxt(folder/'node_diagnostics.csv',delimiter=',',names=True)
                    oldv=old['diversity']; olds=old['anomaly_score']
                row['max_error_vs_streamed_variance']=float(np.max(abs(var[valid]-oldv[valid])))
                row['max_error_vs_streamed_scores']=float(np.max(abs(scores-olds)))
                np.savez_compressed(OUT/(name+'-full-moments.npz'),mean=mean,variance=var,scores=scores,valid=valid,degree=degree,labels=graph.labels)
                if name=='T-Finance':
                    rows=[]
                    for statistic,values in [('variance',var),('std',np.sqrt(var)),('mean',mean)]:
                        deviation=abs(values[valid]-np.median(values[valid])); sc=np.zeros(len(valid))
                        if deviation.std()>0: sc[valid]=(deviation-deviation.mean())/deviation.std()
                        rows.append(dict(statistic=statistic,reference='median',**metric(graph.labels,sc)))
                    (OUT/'tfinance_complete_statistic_ablations.json').write_text(json.dumps(rows,indent=2))
                    # Same projection, selected nodes cover degree spectrum.
                    selected=np.unique([int(np.argmax(degree)),int(np.argsort(degree)[len(degree)//2]),int(np.flatnonzero(degree==2)[0])])
                    for i in selected:
                        neighbors=graph.adjacency.indices[graph.adjacency.indptr[i]:graph.adjacency.indptr[i+1]]
                        start2=time.perf_counter()
                        sm,sv=_pair_mean_variance(_similarity_batches(neighbors,z,None,65536,np.random.default_rng(0)))
                        elapsed=time.perf_counter()-start2
                        benchmarks.append(dict(node=int(i),degree=int(degree[i]),pairs=int(degree[i]*(degree[i]-1)//2),stream_seconds=elapsed,mean_error=float(abs(sm-mean[i])),variance_error=float(abs(sv-var[i]))))
            else:
                row['max_variance_change_vs_randomized0']=float(np.max(abs(var[valid]-reference_v[valid])))
            result.append(row); print(row,flush=True)
        # Flush each graph so provenance survives an interrupted audit.
        (OUT/'projection_moment_audit.json').write_text(json.dumps(dict(inputs=inputs,runs=result,identity_checks=checks,tfinance_stream_benchmarks=benchmarks),indent=2))
    candidates=[]
    for path in sorted((ROOT/'data/Cora').glob('*.mat')):
        graph=load_graph(path,edge_policy='preserve'); d=np.diff(graph.adjacency.indptr)
        projected=TruncatedSVD(n_components=8,random_state=0,n_iter=5).fit_transform(graph.features)
        z,_=_normalize_projected(projected); _,v,valid=full_moments(graph.adjacency,z)
        candidates.append(dict(file=path.name,nodes=len(d),adjacency_nnz=graph.adjacency.nnz,undirected_edges=graph.adjacency.maximum(graph.adjacency.T).nnz//2,valid=int((d>=2).sum()),degree_one=int((d==1).sum()),isolated=int((d==0).sum()),anomalies=int(graph.labels.sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**metric(graph.labels,calibrate_raw(np.nan_to_num(v),valid))))
    (OUT/'cora_candidates.json').write_text(json.dumps(candidates,indent=2))
    (OUT/'audit_source_sha256.json').write_text(json.dumps({str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),PROJECT/'src/data.py',PROJECT/'src/neighbordiv.py',PROJECT/'configs/paper_experiments.json']},indent=2))
    print('audit complete',flush=True)

if __name__=='__main__': main()
