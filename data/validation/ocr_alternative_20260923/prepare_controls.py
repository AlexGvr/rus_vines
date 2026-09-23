import sys,json
from pathlib import Path
sys.path.insert(0,'pipeline')
from PIL import Image
import numpy as np
from eval_field import locate
from imageprep import normalize_batch
from embed import embed_images
from searchcore import pick_view
out=Path('data/validation/ocr_alternative_20260923');out.mkdir(exist_ok=True)
v=np.load('data/index/clean_mv.npy');meta={}
for n in ['10390039_0.jpg','13168313_0.jpg','15536279_1.jpg','7445929_1.jpg']:
 im=Image.open(locate(n)).convert('RGB');vs=[im,normalize_batch([im],steps=('detect',))[0]];j=pick_view(vs,v@embed_images(vs).T)
 vs[j].save(out/n);meta[n]={'view':j,'size':vs[j].size}
(out/'controls.json').write_text(json.dumps(meta,indent=2))
