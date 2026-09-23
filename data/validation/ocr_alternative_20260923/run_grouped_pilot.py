import json,time
from pathlib import Path
from PIL import Image
from paddleocr import TextRecognition
p=Path('data/validation/ocr_alternative_20260923');manifest=json.load(open(p/'line_manifest.json'));sources={r['image']:r['source'] for r in manifest};rows=json.load(open(p/'paddle_serverdet.json'))['results'];model=TextRecognition(model_name='cyrillic_PP-OCRv5_mobile_rec',device='cpu',enable_mkldnn=False,cpu_threads=4);out=[]
for r in rows:
 im=Image.open(sources[r['image']]).convert('RGB');boxes=[w['box'] for w in r['words']];parent=list(range(len(boxes)))
 def find(i):
  while parent[i]!=i:i=parent[i]
  return i
 for i,a in enumerate(boxes):
  for j,b in enumerate(boxes[:i]):
   ha,hb=a[3]-a[1],b[3]-b[1];dy=abs(a[1]+a[3]-b[1]-b[3])/2;gap=max(0,max(a[0],b[0])-min(a[2],b[2]))
   if min(ha,hb)>0 and min(ha,hb)/max(ha,hb)>=.5 and dy<=.3*min(ha,hb) and gap<=max(ha,hb):parent[find(i)]=find(j)
 groups={}
 for i in range(len(boxes)):groups.setdefault(find(i),[]).append(i)
 words=[];start=time.perf_counter()
 for ids in groups.values():
  if len(ids)<2:continue
  bs=[boxes[i] for i in ids];box=[min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)];pad=max(1,round((box[3]-box[1])*.1));roi=[max(0,box[0]-pad),max(0,box[1]-pad),min(im.width,box[2]+pad),min(im.height,box[3]+pad)];crop=im.crop(roi);crop=crop.resize((crop.width*4,crop.height*4),Image.Resampling.LANCZOS)
  import numpy as np
  res=list(model.predict(np.array(crop)[:,:,::-1].copy(),batch_size=1))[0]
  words.append(dict(text=res['rec_text'],conf=float(res['rec_score']),box=box,roi=roi,members=ids))
 out.append(dict(image=r['image'],words=r['words']+words,grouped=words,seconds=r['seconds']+time.perf_counter()-start));print(r['image'],words,flush=True)
(p/'paddle_grouped.json').write_text(json.dumps({'results':out,'complete':True,'note':'Server detection + fixed geometric grouping of adjacent fragments, recognition on union ROI with 10% height padding, x4; no manual boxes or catalog hints.'},ensure_ascii=False,indent=2)+'\n')
