from pathlib import Path
import subprocess,sys,json
root=Path.cwd()
repo=root/'Repositories/NeighborDiv'
out=root/'results/NeighborDiv/protocol-audit-2026-10-13'
commands=[]
for suite,datasets in [('sampling',['Cora','Tolokers','Disney','Questions']),('rank',['Cora','Questions']),('statistic',['Cora','Tolokers','Disney','Questions']),('low-degree',['Cora','Tolokers','Disney','Questions'])]:
 commands.append([sys.executable,'-u',str(repo/'scripts/run_paper_experiments.py'),'--suite',suite,'--datasets',*datasets,'--full-backend','moments','--output-root',str(out/'moments'),'--execute'])
commands.append([sys.executable,'-u',str(repo/'scripts/run_entropy_bin_sensitivity.py'),'--datasets','Cora','Tolokers','Disney','Questions','--output-dir',str(out/'entropy-corrected')])
status=[]
for command in commands:
 print('RUN',command,flush=True)
 code=subprocess.call(command)
 status.append(dict(command=command,exit_code=code))
 (out/'affected_suites_status.json').write_text(json.dumps(status,indent=2))
 if code: raise SystemExit(code)
