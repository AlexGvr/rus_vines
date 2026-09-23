import sys,json,time
from pathlib import Path
sys.path.insert(0,'pipeline')
from PIL import Image
import numpy as np
from ocr import load
out=Path('data/validation/ocr_alternative_20260923');rows=json.load(open(out/'line_manifest.json'));reader=load();res=[]
for i,r in enumerate(rows):
 im=Image.open(r['file']).convert('RGB');t=time.perf_counter();blocks=reader.recognize(np.array(im),horizontal_list=[[0,im.width,0,im.height]],free_list=[],detail=1,paragraph=False)
 res.append(dict(index=i,text=blocks[0][1],conf=float(blocks[0][2]),seconds=time.perf_counter()-t))
(out/'easy_lines.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
