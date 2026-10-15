from pathlib import Path
import json,csv,hashlib
root=Path.cwd();out=root/'results/NeighborDiv/final-audit-2026-10-13'
oracle=json.loads((out/'independent_oracle.json').read_text())
with (out/'projection_controls.csv').open('w',newline='') as f:
 fields=['dataset','variant','policy','AUROC','AP','paper_AUROC','paper_AP'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(oracle['runs'])
main=list(csv.DictReader((out/'live-production/main/summary.csv').open(encoding='utf-8-sig')))
old=list(csv.DictReader((root/'results/NeighborDiv/protocol-audit-2026-10-13/moments/main/summary.csv').open(encoding='utf-8-sig')))
checks=[]
for row,prior in zip(main,old):
 assert row['dataset']==prior['dataset'] and row['status']=='completed'
 checks.append(dict(dataset=row['dataset'],AUROC=float(row['AUROC']),AP=float(row['AP']),AUROC_error_vs_previous=abs(float(row['AUROC'])-float(prior['AUROC'])),AP_error_vs_previous=abs(float(row['AP'])-float(prior['AP']))))
assert len(checks)==7 and max(r['AUROC_error_vs_previous'] for r in checks)==0 and max(r['AP_error_vs_previous'] for r in checks)==0
prov=json.loads((out/'data_provenance.json').read_text());assert len(prov['current_public_sources'])==6 and all(r['byte_identical'] for r in prov['current_public_sources'])
assert len(prov['local_repository_copies'])==8 and all(r['byte_identical'] for r in prov['local_repository_copies'])
assert len(oracle['inputs'])==7 and len(oracle['runs'])==44 and len(oracle['independent_metric_checks'])==7
paper=json.loads((out/'paper_identity.json').read_text());assert paper['byte_identical'] and paper['all_page_text_equal']
scopes=list(csv.DictReader((out/'calibration_scope.csv').open()));assert len(scopes)==49
sbm=list(csv.DictReader((out/'sbm_sampling_semantics.csv').open()));assert len(sbm)==60
summary=dict(status='final-audit-completed; unresolved paper metric discrepancies remain',live_production_checks=checks,public_data_byte_identity_passed=6,repository_copy_identity_passed=8,paper_byte_identity=True,independent_oracle_graphs=7,independent_projection_controls=44,independent_metric_formula_checks=7,calibration_scope_controls=49,paired_SBM_controls=60,additional_DGL_inputs=2,production_scoring_code_modified_this_round=False)
previous_snapshot=json.loads((root/'results/NeighborDiv/protocol-audit-2026-10-13/completion_manifest.json').read_text())['source_hashes_at_audit_completion']
production=['src/data.py','src/neighbordiv.py','src/heuristic_baselines.py','src/amen_ego_native.cpp','scripts/run_neighbordiv.py','configs/paper_experiments.json']
unchanged={}
for item in production:
 path=root/'Repositories/NeighborDiv'/item;key='Repositories/NeighborDiv/'+item
 unchanged[item]=hashlib.sha256(path.read_bytes()).hexdigest()==previous_snapshot[key]
assert all(unchanged.values())
summary['production_source_hashes_unchanged']=unchanged
summary['audit_scripts_sha256']={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'Repositories/NeighborDiv/scripts').glob('*.py') if p.name in ['final_independent_oracle.py','audit_upstream_data.py','audit_calibration_scope.py','audit_sbm_community_sampling.py']}
(out/'final_audit_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
