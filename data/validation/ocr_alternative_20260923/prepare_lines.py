import json
from pathlib import Path
from PIL import Image
out=Path('data/validation/ocr_alternative_20260923');source=Path('data/validation/transcript_final3_assets')
orig=json.load(open('data/validation/ocr_line_localization_oracle_20260923.json'))['results']
expected=['MUSCAT OTTONEL','BRUT','КРАСНОЕ ПОЛУСЛАДКОЕ','ПОРТВЕЙН БЕЛЫЙ','КРЫМСКИЙ','РОЗОВОЕ ПОЛУСУХОЕ']
rows=[]
for r,t in zip(orig,expected):rows.append(dict(image=r['image'],roi=r['roi'],expected=t,group='target',source=str(source/r['image']),language='latin' if t.isascii() else 'cyrillic'))
controls=[('10390039_0.jpg',(68,650,148,673),'КРАСНОЕ'),('10390039_0.jpg',(69,666,149,686),'ПОЛУСЛАДКОЕ'),('13168313_0.jpg',(27,749,191,778),'ПОРТВЕЙН БЕЛЫЙ'),('13168313_0.jpg',(45,773,177,800),'КРЫМСКИЙ'),('15536279_1.jpg',(149,124,214,148),'BRUT'),('15536279_1.jpg',(87,76,269,106),'FANAGORIA'),('7445929_1.jpg',(230,301,666,366),'БАЛАКЛАВА'),('7445929_1.jpg',(321,405,585,466),'Brut Rose')]
for n,box,t in controls:rows.append(dict(image=n,roi=box,expected=t,group='control',source=str(out/n),language='latin' if t.isascii() else 'cyrillic'))
for i,r in enumerate(rows):
 im=Image.open(r['source']).convert('RGB');crop=im.crop(r['roi']);crop=crop.resize((crop.width*4,crop.height*4),Image.Resampling.LANCZOS);p=out/'lines'/f'{i:02d}.png';crop.save(p);r['file']=str(p)
(out/'line_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
