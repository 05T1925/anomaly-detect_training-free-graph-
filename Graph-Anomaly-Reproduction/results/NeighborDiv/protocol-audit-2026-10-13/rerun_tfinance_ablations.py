from pathlib import Path
import subprocess,sys,json
root=Path.cwd();out=root/'results/NeighborDiv/protocol-audit-2026-10-13';status=[]
for suite in ['statistic','low-degree']:
 command=[sys.executable,'-u',str(root/'Repositories/NeighborDiv/scripts/run_paper_experiments.py'),'--suite',suite,'--datasets','T-Finance','--full-backend','moments','--output-root',str(out/'tfinance-ablations'),'--execute']
 code=subprocess.call(command);status.append(dict(suite=suite,exit_code=code));(out/'tfinance_ablations_status.json').write_text(json.dumps(status,indent=2))
 if code:raise SystemExit(code)
