"""Post-hoc convention audit; never changes input labels or historical runs."""
from pathlib import Path
import csv
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/NeighborDiv/protocol-audit-2026-10-13'


def calibrated(values, valid, reference='median', lower_median=False, fallback='zero'):
    v = values[valid]
    center = float(np.mean(v)) if reference == 'mean' else (
        float(np.partition(v, (len(v)-1)//2)[(len(v)-1)//2]) if lower_median else float(np.median(v)))
    deviation = np.abs(v-center)
    scores = np.zeros(values.size)
    if deviation.std() > 0:
        scores[valid] = (deviation-deviation.mean())/deviation.std()
    if fallback == 'median': scores[~valid] = np.median(scores[valid])
    return scores


def metrics(y, s):
    return {'AUROC': float(roc_auc_score(y,s)), 'AP': float(average_precision_score(y,s))}


def single_positive_removal_bounds(y, s):
    order = np.argsort(-s, kind='mergesort'); y=y[order]; s=s[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    counts=np.add.reduceat(y,starts).astype(float)
    sizes=np.diff(np.r_[starts,len(y)]).astype(float)
    rank=np.cumsum(sizes); cumulative_p=np.cumsum(counts)
    cumulative_n=rank-cumulative_p
    P=float(y.sum()); N=len(y)-P
    original=metrics(y,s)
    total_ap=original['AP']*P
    tail=np.cumsum((counts/rank)[::-1])[::-1]-counts/rank
    ap=(total_ap-(counts+cumulative_p-1)/rank-tail)/(P-1)
    win_removed=N-cumulative_n+0.5*(sizes-counts)
    win_added=cumulative_p-counts+0.5*(counts-1)
    auc=(original['AUROC']*P*N-win_removed+win_added)/((P-1)*(N+1))
    positive=counts>0
    return {'original':original,'positive_count':int(P),
            'AP_range_after_any_one_positive_removed':[float(ap[positive].min()),float(ap[positive].max())],
            'AUROC_range_after_any_one_positive_removed':[float(auc[positive].min()),float(auc[positive].max())]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'Repositories/NeighborDiv/configs/paper_experiments.json').read_text())
    records=[]
    for spec in config['datasets']:
        name=spec['name']
        if name in ('Disney','Tolokers','Questions'):
            sources=[]
            for policy in ('preserve','incoming','undirected'):
                d=np.load(OUT/(name+'-'+policy+'.npz'))
                sources.append((policy,d['diversity'],d['degree'],d['labels'],d['scores']))
        else:
            base=ROOT/'results/NeighborDiv/paper-audit/main'/name/'full/seed-0'
            if name=='T-Finance':
                base=ROOT/'results/NeighborDiv/paper-audit-tfinance-full-retry/main/T-Finance/full/seed-0'
            values=np.genfromtxt(base/'node_diagnostics.csv',delimiter=',',names=True)
            y=np.loadtxt(ROOT/'results/NeighborDiv/paper-audit-appendix-d/labels'/(name.replace('-','_')+'.csv'),delimiter=',',skiprows=1,usecols=1,dtype=int)
            sources=[('historical',values['diversity'],values['degree'].astype(int),y,values['anomaly_score'])]
            if name=='Cora':
                current=np.genfromtxt(OUT/'cora-freegad-full/node_diagnostics.csv',delimiter=',',names=True)
                current_y=np.loadtxt(OUT/'appendix-d-corrected/labels/Cora.csv',delimiter=',',skiprows=1,usecols=1,dtype=int)
                sources.append(('freegad-table16',current['diversity'],current['degree'].astype(int),current_y,current['anomaly_score']))
        for policy,v,degree,y,saved in sources:
            valid=degree>=2
            reproduced=calibrated(v,valid)
            print(name,policy,'stored metric',metrics(y,saved),'calibration score error',float(np.max(np.abs(saved-reproduced))),flush=True)
            pairs=degree.astype(np.int64)*(degree.astype(np.int64)-1)//2
            for correction in (0,1):
                corrected=v.copy()
                if correction==1:
                    adjust=valid & (pairs>1)
                    corrected[adjust]*=pairs[adjust]/(pairs[adjust]-1.0)
                    corrected[valid & (pairs==1)]=0.0
                for med in ('numpy','lower'):
                    scores=calibrated(corrected,valid,lower_median=med=='lower')
                    records.append(dict(dataset=name,policy=policy,variance_ddof=correction,median=med,
                                        **metrics(y,scores),paper_AUROC=spec['paper_auc'],paper_AP=spec['paper_ap']))
            if name=='T-Finance':
                bounds=single_positive_removal_bounds(y,saved)
                (OUT/'tfinance_one_label_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
                print('T-Finance one-label bounds',bounds,flush=True)
                extra=[]
                for label,raw,ref,fb,scope in (
                    ('variance-mean',v,'mean','zero','all'),('std-median',np.sqrt(v),'median','zero','all'),
                    ('median-fallback',v,'median','median','all'),('valid-only',v,'median','zero','valid')):
                    scores=calibrated(raw,valid,reference=ref,fallback=fb)
                    mask=valid if scope=='valid' else np.ones(len(y),dtype=bool)
                    extra.append(dict(variant=label,**metrics(y[mask],scores[mask]),evaluated_nodes=int(mask.sum()),
                                      provenance='derived from saved Full population variances; no pair sampling'))
                with (OUT/'tfinance_derived_ablations.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(extra[0]));w.writeheader();w.writerows(extra)
    with (OUT/'statistical_conventions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for r in records:
        if r['median']=='numpy':print(r,flush=True)


if __name__=='__main__':main()
