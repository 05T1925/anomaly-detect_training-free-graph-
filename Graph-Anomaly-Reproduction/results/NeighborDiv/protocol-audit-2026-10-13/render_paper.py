from pathlib import Path
import pypdfium2 as pdfium
root=Path.cwd()
pdfpath=next((root/'Papers/methods').glob('NeighborDiv*.pdf'))
out=root/'results/NeighborDiv/protocol-audit-2026-10-13/paper-pages'
out.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(str(pdfpath))
for p in [3,4,5,17,21,22,24,25,26,27,28]:
 page=pdf[p]
 (out/('page-%02d.txt'%(p+1))).write_text(page.get_textpage().get_text_range(),encoding='utf-8')
 page.render(scale=1.5).to_pil().save(str(out/('page-%02d.png'%(p+1))))
print(len(pdf),out)
