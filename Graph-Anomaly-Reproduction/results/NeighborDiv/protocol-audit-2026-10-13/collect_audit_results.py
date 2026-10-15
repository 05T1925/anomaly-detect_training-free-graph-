from pathlib import Path
import json,csv,hashlib,sys
root=Path.cwd();out=root/'results/NeighborDiv/protocol-audit-2026-10-13'
config=json.loads((root/'Repositories/NeighborDiv/configs/paper_experiments.json').read_text())
manifest={}
for line in (root/'data/checksums.sha256').read_text().splitlines():
 digest,path=line.split(None,1);manifest[path.strip()]=digest
checks=[]
for spec in config['datasets']:
 path=root/'data'/spec['path'];d=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):d.update(block)
 checks.append(dict(dataset=spec['name'],file=spec['path'],sha256=d.hexdigest(),matches_manifest=d.hexdigest()==manifest.get(spec['path'])))
assert all(r['matches_manifest'] for r in checks)
(out/'dataset_integrity.json').write_text(json.dumps(checks,indent=2))
from collections import defaultdict
fdata=json.loads((out/'appendix-f-corrected/graph_audit.json').read_text())
groups=defaultdict(list)
for row in fdata: groups[(row['anomaly_type'],row['homophily_target'])].append(row)
summary=[]
for (kind,h),rows in groups.items():
 summary.append(dict(anomaly_type=kind,h=h,**{key:sum(r[key] for r in rows)/len(rows) for key in ['realized_edge_homophily','mean_normal_degree','mean_anomaly_degree','mean_normal_own_community_fraction','mean_anomaly_own_community_fraction','median_normal_pair_variance','median_anomaly_pair_variance']}))
(out/'sbm_realized_summary.json').write_text(json.dumps(summary,indent=2))
# An analytic counterexample to equating cosine variance with community entropy.
toy=[]
for h in [.1,.3,.5,.7,.9]:
 q=h*h+(1-h)**2/4
 toy.append(dict(h=h,background_same_group_pair_probability=q,background_pair_cosine_variance=q*(1-q),uniform_typeD_pair_cosine_variance=.2*.8))
(out/'orthogonal_community_toy.json').write_text(json.dumps(dict(assumption='Large-degree limit; unit orthogonal community centers and no within-community noise. Illustrates variance/entropy distinction, not the paper SBM numerical reproduction.',rows=toy),indent=2))
# AP definitions under the same unchanged labels/scores.
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve,auc,roc_auc_score
full=root/'results/NeighborDiv/paper-audit-tfinance-full-retry/main/T-Finance/full/seed-0/anomaly_scores.csv'
s=np.loadtxt(full,delimiter=',',skiprows=1,usecols=1)
y=np.loadtxt(root/'results/NeighborDiv/paper-audit-appendix-d/labels/T_Finance.csv',delimiter=',',skiprows=1,usecols=1,dtype=int)
p,r,_=precision_recall_curve(y,s)
(out/'metric_definition_check.json').write_text(json.dumps(dict(AP=float(average_precision_score(y,s)),trapezoidal_PR_AUC=float(auc(r,p)),AUROC=float(roc_auc_score(y,s))),indent=2))
print('dataset integrity: 7/7 passed; SBM and metric definition diagnostics saved')
