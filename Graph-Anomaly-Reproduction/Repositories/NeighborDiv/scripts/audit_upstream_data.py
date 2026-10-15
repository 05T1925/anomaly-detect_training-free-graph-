"""Read-only provenance check. Downloads go only into the new audit folder."""
from pathlib import Path
import hashlib,json,urllib.request,subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/NeighborDiv/final-audit-2026-10-13'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=[
      ('Cora','data/Cora/cora-freegad.mat','Repositories/FreeGAD/FreeGAD/dataset/cora.mat'),
      ('Cora','data/Cora/cora-freegad.mat','Repositories/AlignGAD/Dataset/Test/cora.mat'),
      ('Cora-CoLA','data/Cora/cora-cola.mat','Repositories/CoLA/dataset/cora.mat'),
      ('YelpChi','data/YelpChi/yelpchi-freegad.mat','Repositories/FreeGAD/FreeGAD/dataset/YelpChi.mat'),
      ('YelpChi','data/YelpChi/yelpchi-freegad.mat','Repositories/AlignGAD/Dataset/Test/YelpChi.mat'),
      ('Reddit','data/Reddit/reddit-freegad.mat','Repositories/FreeGAD/FreeGAD/dataset/Reddit.mat'),
      ('Reddit','data/Reddit/reddit-freegad.mat','Repositories/AlignGAD/Dataset/Test/Reddit.mat'),
      ('Disney','data/Disney/disney-dominant.mat','Repositories/DOMINANT/gae/data/Disney.mat')]
    local=[]
    for name,a,b in pairs:
        row=dict(dataset=name,shared_path=a,repository_path=b,shared_sha256=sha(ROOT/a),repository_sha256=sha(ROOT/b))
        row['byte_identical']=row['shared_sha256']==row['repository_sha256'];local.append(row)
    sources=[
      ('Cora','data/Cora/cora-freegad.mat','https://raw.githubusercontent.com/yunf-zhao/FreeGAD/main/FreeGAD/dataset/cora.mat'),
      ('YelpChi','data/YelpChi/yelpchi-freegad.mat','https://raw.githubusercontent.com/yunf-zhao/FreeGAD/main/FreeGAD/dataset/YelpChi.mat'),
      ('Reddit','data/Reddit/reddit-freegad.mat','https://raw.githubusercontent.com/yunf-zhao/FreeGAD/main/FreeGAD/dataset/Reddit.mat'),
      ('Disney','data/Disney/disney-dominant.mat','https://raw.githubusercontent.com/kaize0409/GCN_AnomalyDetection/master/gae/data/Disney.mat'),
      ('Tolokers','data/Tolokers/tolokers.npz','https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/tolokers.npz'),
      ('Questions','data/Questions/questions.npz','https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/questions.npz')]
    online=[]
    downloads=OUT/'upstream';downloads.mkdir(exist_ok=True)
    for name,path,url in sources:
        row=dict(dataset=name,url=url,shared_sha256=sha(ROOT/path))
        target=downloads/(name+Path(path).suffix)
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'NeighborDiv-reproducibility-audit'})
            with urllib.request.urlopen(req,timeout=25) as response,target.open('wb') as f:
                total=0
                for block in iter(lambda:response.read(1<<20),b''):
                    total+=len(block)
                    if total>256*1024**2:raise ValueError('Unexpected source size exceeds audit limit')
                    f.write(block)
            row.update(status='downloaded',downloaded_bytes=total,upstream_sha256=sha(target))
            row['byte_identical']=row['shared_sha256']==row['upstream_sha256']
        except Exception as e:row.update(status='unavailable',error=str(e))
        online.append(row)
        (OUT/'data_provenance.json').write_text(json.dumps(dict(local_repository_copies=local,current_public_sources=online),indent=2))
        print(row,flush=True)
    commits={}
    for name in ['FreeGAD','AlignGAD','DOMINANT','CoLA']:
        commits[name]=subprocess.check_output(['git','-C',str(ROOT/'Repositories'/name),'rev-parse','HEAD'],text=True).strip()
    (OUT/'repository_commit_snapshots.json').write_text(json.dumps(commits,indent=2))
    print('provenance complete',flush=True)

if __name__=='__main__':main()
