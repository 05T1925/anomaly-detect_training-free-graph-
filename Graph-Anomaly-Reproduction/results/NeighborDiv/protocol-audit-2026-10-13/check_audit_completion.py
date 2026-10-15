from pathlib import Path
import json,csv,ast,hashlib,xml.etree.ElementTree as ET
root=Path.cwd();out=root/'results/NeighborDiv/protocol-audit-2026-10-13';repo=root/'Repositories/NeighborDiv'
counts={}
expected={'main':7,'sampling':44,'rank':8,'statistic':16,'low-degree':12}
for suite,wanted in expected.items():
 path=out/'moments'/suite/'summary.csv'
 rows=list(csv.DictReader(path.open()));assert len(rows)==wanted and all(r['status']=='completed' for r in rows)
 counts['moments/'+suite]=len(rows)
for suite,wanted in [('statistic',4),('low-degree',3)]:
 rows=list(csv.DictReader((out/'tfinance-ablations'/suite/'summary.csv').open()));assert len(rows)==wanted and all(r['status']=='completed' for r in rows)
 counts['tfinance/'+suite]=len(rows)
for file,wanted in [('appendix-d-corrected/metrics.csv',28),('appendix-f-corrected/runs.csv',135),('appendix-g-corrected/metrics.csv',11),('entropy-corrected/metrics.csv',16),('statistical_conventions.csv',56)]:
 rows=list(csv.DictReader((out/file).open()));assert len(rows)==wanted;counts[file]=len(rows)
for path in repo.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
for path in (out/'appendix-g-corrected').glob('*.svg'):ET.parse(path)
for row in csv.DictReader((out/'moments/main/summary.csv').open()):
 run=json.loads((Path(row['output_dir'])/'run.json').read_text())
 assert run['metrics']['evaluated_nodes']==run['graph']['n_nodes']
 assert run['graph']['n_valid_nodes_degree_ge_2']==int(row['paper_valid_nodes'])
 assert run['graph']['pair_mode']=='full' and run['graph']['enumerated_neighbor_pairs']==0
 assert run['parameters']['full_backend']=='moments'
checks=json.loads((out/'regression_checks.json').read_text());assert checks['status']=='passed'
status=json.loads((out/'affected_suites_status.json').read_text());assert len(status)==5 and all(r['exit_code']==0 for r in status)
status=json.loads((out/'tfinance_ablations_status.json').read_text());assert len(status)==2 and all(r['exit_code']==0 for r in status)
manifest=dict(status='completed for the audit scope; paper metrics remain unresolved except specified matches',counts=counts,regression_checks=len(checks['checks']),projection_runs=len(json.loads((out/'projection_moment_audit.json').read_text())['runs']),main_degree_counts_match_Table16=True,python_syntax='passed',svg_xml='passed',source_hashes_at_audit_completion={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.json','.sh')})
(out/'completion_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='source_hashes_at_audit_completion'},indent=2))
