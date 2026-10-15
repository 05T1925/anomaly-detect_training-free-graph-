from pathlib import Path
import sys,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
root=Path.cwd();sys.path.insert(0,str(root/'Repositories/NeighborDiv'))
from src.data import load_graph
from src.heuristic_baselines import project_l1,score_baseline
from src.neighbordiv import _normalize_projected,_full_pair_moments
p=root/'results/NeighborDiv/protocol-audit-2026-10-13'
g=load_graph(root/'data/Disney/disney-dominant.mat',edge_policy='undirected');x=project_l1(g.features)
rows=[];original={}
# Predeclared global sign reversal: a valid alternative SVD sign convention.
for sign in [1,-1]:
 for method in ['NRS','PCD','AMEN-Ego']:
  s,raw,valid=score_baseline(method,g.adjacency,sign*x)
  if sign==1:original[method]=s
  rows.append(dict(sign=sign,method=method,score_difference=float(np.max(abs(s-original[method]))),AUROC=float(roc_auc_score(g.labels,s)),AP=float(average_precision_score(g.labels,s))))
(p/'svd_sign_baseline_audit.json').write_text(json.dumps(rows,indent=2))
print(rows)
