"""Predeclared diagnostics for accidentally including undefined diversity.

These variants deliberately depart from Eq. 10-13 except paper-valid-only;
they are tests of implementation-mismatch hypotheses, never new defaults.
"""
from pathlib import Path
import json,csv
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/NeighborDiv/final-audit-2026-10-13'

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    config=json.loads((ROOT/'Repositories/NeighborDiv/configs/paper_experiments.json').read_text())
    for spec in config['datasets']:
        name=spec['name']
        paper_valid={'Cora':(.6142,.0737),'YelpChi':(.5164,.0270),'Reddit':(.5122,.0356),'T-Finance':(.9122,.6084),'Tolokers':(.5676,.2505),'Disney':(.7477,.1851),'Questions':(.6273,.0490)}
        file=ROOT/'results/NeighborDiv/protocol-audit-2026-10-13/moments/main'/name/'full/seed-0/node_diagnostics.csv'
        data=np.genfromtxt(file,delimiter=',',names=True)
        valid=data['valid'].astype(bool);v=np.where(valid,data['diversity'],0.)
        y=np.loadtxt(ROOT/'results/NeighborDiv/protocol-audit-2026-10-13/appendix-d-corrected/labels'/(name.replace('-','_')+'.csv'),delimiter=',',skiprows=1,usecols=1,dtype=int)
        variants=[('paper-valid-only','valid','valid','neutral-zero'),
                  ('median-includes-imputed-zero','all','valid','neutral-zero'),
                  ('zscore-includes-imputed-zero-deviation','valid','all-zero-deviation','neutral-zero'),
                  ('zscore-includes-deviation-of-imputed-zero','valid','all-imputed-diversity','neutral-zero'),
                  ('all-median-and-all-zscore','all','all-imputed-diversity','neutral-zero'),
                  ('paper-reference-with-unstandardized-deviation','valid','none','zero'),
                  ('raw-variance','none','none','zero')]
        for label,ref_scope,std_scope,fallback in variants:
            if ref_scope=='none':delta=v.copy();ref=0.
            else:
                ref=float(np.median(v[valid] if ref_scope=='valid' else v))
                delta=abs(v-ref)
            if std_scope in ('valid','all-zero-deviation'):delta[~valid]=0.
            if std_scope=='none':s=delta.copy()
            else:
                mask=valid if std_scope=='valid' else np.ones(len(v),dtype=bool)
                spread=delta[mask].std();s=(delta-delta[mask].mean())/spread if spread>0 else np.zeros(len(v))
            s[~valid]=0.
            rows.append(dict(dataset=name,variant=label,reference=float(ref),AUROC=float(roc_auc_score(y,s)),AP=float(average_precision_score(y,s)),valid_AUROC=float(roc_auc_score(y[valid],s[valid])),valid_AP=float(average_precision_score(y[valid],s[valid])),paper_valid_AUROC=paper_valid[name][0],paper_valid_AP=paper_valid[name][1],paper_AUROC=spec['paper_auc'],paper_AP=spec['paper_ap'],matches_both_to_4_decimals=round(roc_auc_score(y,s),4)==spec['paper_auc'] and round(average_precision_score(y,s),4)==spec['paper_ap']))
    with (OUT/'calibration_scope.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:
        if r['dataset'] in ('T-Finance','Questions'):print(r,flush=True)

if __name__=='__main__':main()
