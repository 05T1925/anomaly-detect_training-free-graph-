"""Final independent audit. Does not import NeighborDiv data/scoring helpers.

All projection alternatives are fixed before looking at evaluation labels.
Only 'randomized-XV' mirrors the current production projection. Centering and
raw-row normalization are diagnostic alternatives, not asserted paper settings.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time, zipfile
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT/'Repositories/NeighborDiv'
DEFAULT_OUT = ROOT/'results/NeighborDiv/final-audit-2026-10-13'


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def canonical_a(raw,policy):
    a=sp.csr_matrix(raw,copy=True);a.sum_duplicates();a.eliminate_zeros()
    a.data=np.ones(a.nnz,dtype=np.bool_);a.setdiag(False);a.eliminate_zeros()
    if policy=='incoming':a=a.transpose().tocsr()
    if policy=='undirected':a=(a+a.transpose()).astype(bool).tocsr()
    a.sort_indices()
    return a


def read_raw(path,fmt,policy):
    if fmt=='mat':
        m=sio.loadmat(str(path))
        if 'Network' in m:
            a=m['Network'];x=m['Attributes'];y=m['Label']
        else:
            a=m['A'];x=m['X'];y=m['gnd']
    elif fmt=='npz':
        with np.load(str(path),allow_pickle=False) as f:
            x=f['node_features'];y=f['node_labels'];e=f['edges']
            a=sp.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(x),len(x)))
    else:
        import dgl
        graphs,_=dgl.load_graphs(str(path));g=graphs[0]
        feature_key=next(k for k in ('feature','feat','features') if k in g.ndata)
        label_key=next(k for k in ('label','labels','y') if k in g.ndata)
        x=g.ndata[feature_key].detach().cpu().numpy()
        y=g.ndata[label_key].detach().cpu().numpy()
        source,target=g.edges(order='eid')
        a=sp.coo_matrix((np.ones(len(source),dtype=bool),(source.cpu().numpy(),target.cpu().numpy())),shape=(g.num_nodes(),g.num_nodes()))
        if y.ndim==2 and y.shape[1]==2:
            assert np.isin(y,(0,1)).all() and np.all(y.sum(axis=1)==1)
            y=y.argmax(axis=1)
    x=sp.csr_matrix(x,dtype=np.float64)
    y=(y.toarray() if sp.issparse(y) else np.asarray(y)).ravel()
    assert np.isin(y,(0,1)).all() and len(y)==x.shape[0]
    return canonical_a(a,policy),x,y.astype(int)


def csr_digest(x):
    x=sp.csr_matrix(x,dtype=np.float64);x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    h=hashlib.sha256()
    for a in (np.array(x.shape,dtype='<i8'),x.indptr.astype('<i8'),x.indices.astype('<i8'),x.data.astype('<f8')):
        h.update(a.tobytes())
    return h.hexdigest()


def literal_scores(a,projected):
    """Eq. 4 directly for small neighborhoods, uncentered Gram identity otherwise.

    Unlike the production centered identity: sum(pair dot)=
    (||sum z||^2-sum||z||^2)/2; sum(pair dot^2)=
    (||Z.T Z||_F^2-sum||z||^4)/2. No pair sampling.
    """
    norm1=np.abs(projected).sum(axis=1,keepdims=True)
    z=np.divide(projected,norm1,out=np.zeros_like(projected),where=norm1>0)
    norm2=np.linalg.norm(z,axis=1,keepdims=True)
    z=np.divide(z,norm2,out=np.zeros_like(z),where=norm2>0)
    d=np.diff(a.indptr);valid=d>=2
    v=np.full(len(d),np.nan)
    for i in np.flatnonzero(valid):
        t=z[a.indices[a.indptr[i]:a.indptr[i+1]]];n=len(t)
        if n<=128:
            s=(t@t.T)[np.triu_indices(n,1)]
            v[i]=float(np.var(s,ddof=0)) if n>2 else 0.
        elif np.all(t==t[0]):
            v[i]=0.
        else:
            total=t.sum(axis=0);norms=np.einsum('ij,ij->i',t,t);den=n*(n-1)
            mean=(np.dot(total,total)-norms.sum())/den
            second=(np.square(t.T@t).sum()-np.square(norms).sum())/den
            v[i]=max(0.,float(second-mean**2))
    delta=np.abs(v[valid]-np.median(v[valid]));score=np.zeros(len(d))
    if delta.std()>0:score[valid]=(delta-delta.mean())/delta.std()
    return score,v,valid,z


def metrics(y,s):
    return dict(AUROC=float(roc_auc_score(y,s)),AP=float(average_precision_score(y,s)))


def independent_metrics(y,s):
    """Independent rank formulas including score ties; no sklearn here."""
    order=np.argsort(s,kind='mergesort');yy=y[order];ss=s[order]
    starts=np.r_[0,np.flatnonzero(np.diff(ss)!=0)+1]
    ends=np.r_[starts[1:],len(ss)];sizes=ends-starts
    positives=np.add.reduceat(yy,starts);negatives=sizes-positives
    normal_below=np.cumsum(negatives)-negatives
    auc=np.sum(positives*(normal_below+negatives/2))/(yy.sum()*(len(yy)-yy.sum()))
    ps=positives[::-1];ns=sizes[::-1]
    ap=np.sum(ps*np.cumsum(ps)/np.cumsum(ns))/yy.sum()
    return dict(AUROC=float(auc),AP=float(ap))


def verify_zip(out):
    archive=ROOT/'data/T-Finance/tfinance-source.zip'
    extracted=ROOT/'data/T-Finance/tfinance-source/tfinance'
    h=hashlib.sha256()
    with zipfile.ZipFile(archive) as f:
        entry=next(n for n in f.namelist() if Path(n).name=='tfinance')
        with f.open(entry) as g:
            for b in iter(lambda:g.read(1<<20),b''):h.update(b)
    row=dict(archive_member=entry,uncompressed_sha256=h.hexdigest(),existing_extracted_sha256=digest_file(extracted))
    row['equal']=row['uncompressed_sha256']==row['existing_extracted_sha256']
    assert row['equal']
    (out/'tfinance_zip_identity.json').write_text(json.dumps(row,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    out=p.parse_args().output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config=json.loads((PROJECT/'configs/paper_experiments.json').read_text())
    verify_zip(out)
    rows=[];data=[];metric_checks=[];rounding=[]
    for spec in config['datasets']:
        name=spec['name'];source=ROOT/'data'/spec['path'];fmt=spec['format']
        if fmt=='dgl-zip':source=ROOT/'data/T-Finance/tfinance-source/tfinance';fmt='dgl'
        a,x,y=read_raw(source,fmt,spec['edge_policy']);d=np.diff(a.indptr)
        input_row=dict(dataset=name,source=str(source),policy=spec['edge_policy'],nodes=len(d),features=x.shape[1],anomalies=int(y.sum()),valid=int((d>=2).sum()),isolated=int((d==0).sum()),degree_one=int((d==1).sum()),adjacency_sha256=csr_digest(a),features_sha256=csr_digest(x),labels_sha256=hashlib.sha256(y.astype('<i8').tobytes()).hexdigest())
        data.append(input_row)
        gram=(x.T@x).toarray()
        w,v=eigh(gram,subset_by_index=[x.shape[1]-8,x.shape[1]-1],check_finite=False)
        exact=x@v[:,::-1]
        svd=TruncatedSVD(n_components=8,algorithm='randomized',n_iter=5,random_state=0)
        current=svd.fit_transform(x)
        u,s,vt=randomized_svd(x,n_components=8,n_iter=5,random_state=0)
        row_norm=np.asarray(abs(x).sum(axis=1)).ravel()
        xx=sp.diags(np.divide(1.,row_norm,out=np.zeros_like(row_norm),where=row_norm>0))@x
        before=TruncatedSVD(n_components=8,n_iter=5,random_state=0).fit_transform(xx)
        mu=np.asarray(x.mean(axis=0)).ravel();centered_gram=gram-x.shape[0]*np.outer(mu,mu)
        _,centered_v=eigh(centered_gram,subset_by_index=[x.shape[1]-8,x.shape[1]-1],check_finite=False)
        centered=x@centered_v[:,::-1]-mu@centered_v[:,::-1]
        # Appendix B.3 mentions BN for adapted learning baselines. This is
        # an explicit diagnostic of incorrectly sharing that preprocessing
        # with NeighborDiv, not an assertion that Eq. 1 requires BN.
        bn=(current-current.mean(axis=0))/np.sqrt(current.var(axis=0)+1e-5)
        variations=[('randomized-XV',current),('literal-randomized-US',u*s),('exact-eigh-XV',exact),('row-L1-before-SVD',before),('centered-PCA',centered),('post-SVD-column-BN',bn)]
        reference=None;baseline_z=None
        for variant,projection in variations:
            begin=time.perf_counter();scores,variance,valid,z=literal_scores(a,np.asarray(projection))
            row=dict(dataset=name,variant=variant,policy=spec['edge_policy'],**metrics(y,scores),paper_AUROC=spec['paper_auc'],paper_AP=spec['paper_ap'],score_seconds=time.perf_counter()-begin)
            if reference is None:
                reference=projection;baseline_z=z
                historical=ROOT/'results/NeighborDiv/protocol-audit-2026-10-13/moments/main'/name/'full/seed-0'
                saved=np.genfromtxt(historical/'node_diagnostics.csv',delimiter=',',names=True)
                row['max_variance_error_vs_production']=float(np.max(abs(variance[valid]-saved['diversity'][valid])))
                row['max_score_error_vs_production']=float(np.max(abs(scores-saved['anomaly_score'])))
                row['production_metrics']=metrics(y,saved['anomaly_score'])
                independent=independent_metrics(y,saved['anomaly_score']);sklearn=metrics(y,saved['anomaly_score'])
                assert all(abs(independent[k]-sklearn[k])<1e-12 for k in sklearn)
                metric_checks.append(dict(dataset=name,rank_formula=independent,sklearn=sklearn))
                np.savez_compressed(out/(name+'-oracle.npz'),scores=scores,variance=variance,valid=valid,degree=d,labels=y)
            else:
                row['relative_projection_change']=float(np.linalg.norm(projection-reference)/np.linalg.norm(reference))
            rows.append(row);print(row,flush=True)
        if name in ('Tolokers','Questions'):
            union=a.maximum(a.T).tocsr()
            scores,_,_,_=literal_scores(union,current)
            rows.append(dict(dataset=name,variant='official-undirected-XV',policy='undirected',**metrics(y,scores),paper_AUROC=spec['paper_auc'],paper_AP=spec['paper_ap']))
        if name=='T-Finance':
            import torch
            torch.set_num_threads(1)
            chosen=np.unique([int(np.argmax(d)),int(np.argsort(d)[len(d)//2]),int(np.flatnonzero(d==2)[0])])
            for node in chosen:
                neighbors=a.indices[a.indptr[node]:a.indptr[node+1]];z=baseline_z[neighbors]
                t=torch.from_numpy(z.astype(np.float32));dots=t@t.T
                index=torch.triu_indices(len(z),len(z),offset=1)
                values=dots[index[0],index[1]];v32=values.var(unbiased=False).item()
                v64=np.load(out/(name+'-oracle.npz'))['variance'][node]
                rounding.append(dict(node=int(node),degree=int(d[node]),float32_dot_variance=v32,float64_variance=float(v64),error=float(abs(v32-v64))))
                del dots,index,values,t
        (out/'independent_oracle.json').write_text(json.dumps(dict(inputs=data,runs=rows,independent_metric_checks=metric_checks,tfinance_float32_dot_checks=rounding),indent=2))
    candidates=[]
    for name in ('Cora','Reddit'):
        path=ROOT/'data'/name/('inj_'+name.lower()+'-tfgad.bin')
        try:
            a,x,y=read_raw(path,'dgl','preserve');d=np.diff(a.indptr)
            projected=TruncatedSVD(n_components=8,n_iter=5,random_state=0).fit_transform(x)
            scores,_,_,_=literal_scores(a,projected)
            candidates.append(dict(dataset=name,file=str(path),nodes=len(d),features=x.shape[1],valid=int((d>=2).sum()),anomalies=int(y.sum()),adjacency_nnz=int(a.nnz),**metrics(y,scores)))
        except Exception as e:candidates.append(dict(dataset=name,file=str(path),error=str(e)))
    (out/'additional_dgl_candidates.json').write_text(json.dumps(candidates,indent=2))
    (out/'oracle_source_sha256.json').write_text(json.dumps({str(Path(__file__).resolve().relative_to(ROOT)):digest_file(__file__), 'config':digest_file(PROJECT/'configs/paper_experiments.json')},indent=2))
    print('independent final audit complete',flush=True)

if __name__=='__main__':main()
