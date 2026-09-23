import json,time,importlib.metadata
from pathlib import Path
from paddleocr import TextRecognition
out=Path('data/validation/ocr_alternative_20260923');rows=json.load(open(out/'line_manifest.json'));results=[]
for lang in ['cyrillic','latin']:
 name=f'{lang}_PP-OCRv5_mobile_rec'
 model=TextRecognition(model_name=name,device='cpu',enable_mkldnn=False,cpu_threads=4)
 # Warmup excluded from timing.
 _=list(model.predict(input=rows[0]['file'],batch_size=1))
 for i,r in enumerate(rows):
  start=time.perf_counter();res=list(model.predict(input=r['file'],batch_size=1))[0]
  results.append(dict(index=i,model=name,text=res['rec_text'],conf=float(res['rec_score']),seconds=time.perf_counter()-start))
  print(results[-1],flush=True)
 (out/'paddle_lines.json').write_text(json.dumps({'complete':False,'results':results},ensure_ascii=False,indent=2))
(out/'paddle_lines.json').write_text(json.dumps({'complete':True,'results':results,'versions':{x:importlib.metadata.version(x) for x in ['paddleocr','paddlex','paddlepaddle']},'device':'cpu','cpu_threads':4},ensure_ascii=False,indent=2)+'\n')
