from pathlib import Path
import urllib.request,hashlib,json,re
import pypdfium2 as pdfium
root=Path.cwd();out=root/'results/NeighborDiv/final-audit-2026-10-13'
local=next((root/'Papers/methods').glob('NeighborDiv*.pdf'))
url='https://arxiv.org/pdf/2605.20879v1'
row=dict(local_path=str(local),public_url=url,local_sha256=hashlib.sha256(local.read_bytes()).hexdigest())
try:
 target=out/'upstream/NeighborDiv-paper.pdf'
 request=urllib.request.Request(url,headers={'User-Agent':'NeighborDiv-reproducibility-audit'})
 with urllib.request.urlopen(request,timeout=25) as response:
  data=response.read(8*1024*1024)
 assert data.startswith(b'%PDF-')
 target.write_bytes(data)
 row['public_sha256']=hashlib.sha256(data).hexdigest();row['byte_identical']=row['public_sha256']==row['local_sha256']
 lp=pdfium.PdfDocument(str(local));rp=pdfium.PdfDocument(str(target))
 row['local_pages']=len(lp);row['public_pages']=len(rp)
 equal=[]
 if len(lp)==len(rp):
  for i in range(len(lp)):
   a=lp[i].get_textpage().get_text_range();b=rp[i].get_textpage().get_text_range()
   equal.append(re.sub(r'\s+','',a)==re.sub(r'\s+','',b))
 row['text_equal_by_page']=equal;row['all_page_text_equal']=len(equal)==len(lp) and all(equal)
 row['status']='verified'
except Exception as e:row.update(status='unavailable',error=str(e))
(out/'paper_identity.json').write_text(json.dumps(row,indent=2,ensure_ascii=False))
print(json.dumps(row,ensure_ascii=False))
