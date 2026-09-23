import json,time
from pathlib import Path
from paddleocr import PaddleOCR
out=Path('data/validation/ocr_alternative_20260923');rows=json.load(open(out/'line_manifest.json'));sources={r['image']:r['source'] for r in rows}
model=PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det',text_recognition_model_name='cyrillic_PP-OCRv5_mobile_rec',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',enable_mkldnn=False,cpu_threads=4)
results=[]
for name,path in sources.items():
 t=time.perf_counter();r=list(model.predict(path))[0]
 words=[dict(text=txt,conf=float(conf),box=[int(x) for x in box]) for txt,conf,box in zip(r['rec_texts'],r['rec_scores'],r['rec_boxes'])]
 results.append(dict(image=name,words=words,seconds=time.perf_counter()-t))
 print(name,words,flush=True)
(out/'paddle_automatic.json').write_text(json.dumps({'results':results,'complete':True,'note':'CPU; native pipeline detector mobile + Cyrillic recognizer; orientation/unwarping disabled; no manual ROIs.'},ensure_ascii=False,indent=2)+'\n')
