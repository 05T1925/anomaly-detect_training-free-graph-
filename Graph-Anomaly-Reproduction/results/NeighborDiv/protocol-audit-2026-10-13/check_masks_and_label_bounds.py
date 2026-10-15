from pathlib import Path
import sys,json,csv
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root/'Repositories/NeighborDiv/scripts'))
from audit_score_conventions import calibrated,metrics
out=root/'results/NeighborDiv/protocol-audit-2026-10-13'
rows=[]
for name in ['Disney','Tolokers','Questions']:
 for neighbors in ['preserve','incoming','undirected']:
  stat=np.load(out/(name+'-'+neighbors+'.npz'))
  for degree_source in ['preserve','incoming','undirected']:
   deg=np.load(out/(name+'-'+degree_source+'.npz'))['degree']
   valid=(deg>=2)&np.isfinite(stat['diversity'])
   s=calibrated(stat['diversity'],valid)
   rows.append(dict(dataset=name,neighbors=neighbors,valid_mask_degree_source=degree_source,requested_valid=int((deg>=2).sum()),defined_valid=int(valid.sum()),**metrics(stat['labels'],s)))
with (out/'neighbor_mask_crosscheck.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for row in rows:
 if row['valid_mask_degree_source']=='incoming':print(row)
# Validate closed-form label bounds against brute force with ties.
from audit_score_conventions import single_positive_removal_bounds
rng=np.random.default_rng(17)
for k in range(20):
 y=np.r_[np.ones(5,dtype=int),np.zeros(6,dtype=int)]; s=rng.integers(0,4,size=len(y))
 b=single_positive_removal_bounds(y,s); values=[]
 for i in np.flatnonzero(y):
  changed=y.copy();changed[i]=0; values.append(metrics(changed,s))
 for metric in ['AP','AUROC']:
  observed=[min(r[metric] for r in values),max(r[metric] for r in values)]
  assert np.allclose(observed,b[metric+'_range_after_any_one_positive_removed'],rtol=0,atol=1e-12)
print('20 brute-force checks of label bounds passed')
