from pathlib import Path
import scipy.io as sio,scipy.sparse as sp,numpy as np,json
root=Path.cwd();out=root/'results/NeighborDiv/final-audit-2026-10-13';rows=[]
for name,file in [('Cora','Cora/cora-freegad.mat'),('YelpChi','YelpChi/yelpchi-freegad.mat'),('Reddit','Reddit/reddit-freegad.mat'),('Disney','Disney/disney-dominant.mat')]:
 m=sio.loadmat(str(root/'data'/file));keys=('Network','Attributes','Label') if 'Network' in m else ('A','X','gnd')
 a,x,y=[m[k] for k in keys];av=a.data if sp.issparse(a) else np.asarray(a).ravel();xv=x.data if sp.issparse(x) else np.asarray(x).ravel()
 row=dict(dataset=name,keys=list(keys),raw_adjacency_dtype=str(a.dtype),nonzero_edge_values=np.unique(av[av!=0]).tolist(),raw_self_loops=int(np.count_nonzero(a.diagonal())),feature_dtype=str(x.dtype),nonfinite_features=int((~np.isfinite(xv)).sum()),label_values=np.unique(y.toarray() if sp.issparse(y) else y).tolist())
 assert row['nonzero_edge_values']==[1] and row['nonfinite_features']==0 and row['label_values']==[0,1]
 rows.append(row)
(out/'raw_mat_schema.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
