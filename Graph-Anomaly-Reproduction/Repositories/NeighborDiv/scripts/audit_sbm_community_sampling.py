"""Paired diagnostic: uniform endpoint sampling vs balanced community quotas.

Both readings of Appendix F are fixed in advance. Labels, normal graphs,
features, anomaly selections, degree budgets and seed formulas are identical.
Never selects a protocol using evaluation metrics.
"""
from pathlib import Path
import csv,json,sys
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
PROJECT=Path(__file__).resolve().parents[1];ROOT=PROJECT.parent.parent
sys.path.insert(0,str(PROJECT/'scripts'))
from run_appendix_f_sbm import make_normal_sbm
from final_independent_oracle import literal_scores,metrics
OUT=ROOT/'results/NeighborDiv/final-audit-2026-10-13'


def rewired(a,groups,degree,selected,seed,h,kind,policy):
    h_nodes,d_nodes=selected[:50],selected[50:]
    anomaly_nodes=d_nodes if kind=='Type-D' else selected
    mask=np.zeros(len(groups),bool);mask[anomaly_nodes]=True
    original=a.tocoo();keep=~(mask[original.row]|mask[original.col])
    rows=original.row[keep].tolist();cols=original.col[keep].tolist()
    pieces=[(d_nodes,True,1)] if kind=='Type-D' else [(h_nodes,False,0),(d_nodes,True,1)]
    ti=1 if kind=='Type-D' else 2
    coverage=[]
    for nodes,diverse,part in pieces:
        rng=np.random.default_rng(seed*100000+round(h*1000)*10+ti*2+part)
        normal=np.flatnonzero(~mask)
        for node in nodes:
            count=int(degree[node])
            if not diverse:
                candidates=np.flatnonzero((groups==groups[node])&~mask)
                targets=rng.choice(candidates,size=min(count,len(candidates)),replace=False)
            elif policy=='uniform-nodes':
                targets=rng.choice(normal,size=min(count,len(normal)),replace=False)
            else:
                quota=np.full(5,count//5,int)
                extra=rng.permutation(5)[:count%5];quota[extra]+=1
                targets=np.concatenate([rng.choice(normal[groups[normal]==g],size=quota[g],replace=False) for g in range(5)])
            assert len(targets)==count
            rows.extend([int(node)]*count);cols.extend(targets.tolist())
            rows.extend(targets.tolist());cols.extend([int(node)]*count)
            if diverse:coverage.append(len(np.unique(groups[targets])))
    result=sp.coo_matrix((np.ones(len(rows),dtype=bool),(rows,cols)),shape=a.shape).tocsr()
    y=mask.astype(int)
    assert np.array_equal(np.diff(result.indptr)[anomaly_nodes],degree[anomaly_nodes])
    return result,y,float(np.mean(coverage))


def main():
    OUT.mkdir(parents=True,exist_ok=True);runs=[]
    history=list(csv.DictReader((ROOT/'results/NeighborDiv/protocol-audit-2026-10-13/appendix-f-corrected/runs.csv').open()))
    historical={(float(r['homophily']),int(r['generation_seed']),r['anomaly_type']):float(r['AUROC']) for r in history if r['method']=='NeighborDiv'}
    for seed in (0,1,2):
        for h in (.1,.3,.5,.7,.9):
            base,x,groups,degree,_,_=make_normal_sbm(seed*100+round(h*10),h)
            rng=np.random.default_rng(seed*10000+round(h*100))
            selected=rng.choice(2000,size=100,replace=False)
            projected=TruncatedSVD(n_components=8,n_iter=5,random_state=0).fit_transform(x)
            for kind in ('Type-D','Mixed'):
                for policy in ('uniform-nodes','balanced-community-quotas'):
                    a,y,coverage=rewired(base,groups,degree,selected,seed,h,kind,policy)
                    scores,var,valid,_=literal_scores(a,projected)
                    row=dict(seed=seed,h=h,anomaly_type=kind,policy=policy,**metrics(y,scores),mean_typeD_communities_covered=coverage,normal_median_variance=float(np.median(var[valid&(y==0)])),anomaly_median_variance=float(np.median(var[valid&(y==1)])))
                    if policy=='uniform-nodes':
                        row['AUROC_error_vs_historical']=abs(row['AUROC']-historical[(h,seed,kind)])
                        # Independent BLAS pair dots can split previously tied
                        # scores. Allow at most two pair-order units in AUROC.
                        tolerance=2.0/(int(y.sum())*int((y==0).sum()))
                        if row['AUROC_error_vs_historical']>tolerance:
                            print('historical mismatch',row,'historical',historical[(h,seed,kind)],flush=True)
                            raise AssertionError('Reconstructed uniform protocol differs from historical scores')
                    runs.append(row)
    with (OUT/'sbm_sampling_semantics.csv').open('w',newline='') as f:
        fields=list(runs[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(runs)
    summary=[]
    paper={'Type-D':[.785,.778,.880,.847,.362],'Mixed':[.859,.864,.915,.876,.522]}
    for kind in ('Type-D','Mixed'):
        for i,h in enumerate((.1,.3,.5,.7,.9)):
            for policy in ('uniform-nodes','balanced-community-quotas'):
                values=[r for r in runs if r['anomaly_type']==kind and r['h']==h and r['policy']==policy]
                s=dict(anomaly_type=kind,h=h,policy=policy,AUROC_mean=float(np.mean([r['AUROC'] for r in values])),AUROC_std=float(np.std([r['AUROC'] for r in values])),paper_AUROC=paper[kind][i],mean_communities_covered=float(np.mean([r['mean_typeD_communities_covered'] for r in values])))
                summary.append(s);print(s,flush=True)
    (OUT/'sbm_sampling_semantics_summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'sbm_sampling_semantics_metadata.json').write_text(json.dumps(dict(interpretation='Appendix F uniform across communities is ambiguous: IID uniform endpoints vs approximately equal quotas. Diagnostic only; author code unknown.',graph_seed_formula='seed_index*100+round(h*10)',selection_seed_formula='seed_index*10000+round(h*100)',n_runs=len(runs),scorer='independent Eq.4/9/10-13 oracle',feature_projection='sklearn randomized r8 n_iter5 seed0',historical_uniform_runs_independently_verified=30,historical_max_AUROC_error=max(r.get('AUROC_error_vs_historical',0.) for r in runs),historical_tie_tolerance='2/(positive_count*negative_count) AUROC pair-order units'),indent=2))

if __name__=='__main__':main()
