"""Selected query views of the regression part (final3_test.json), lossless PNG.
Same selection as the service: raw vs detect view by cosine (pick_view)."""
import sys, json
from pathlib import Path
sys.path.insert(0, 'pipeline')
from PIL import Image
import numpy as np
from eval_field import locate
from imageprep import normalize_batch
from embed import embed_images
from searchcore import pick_view
out = Path('data/validation/ocr_grouped_max2000_test_20260923')
rows = json.load(open('data/validation/final3_test.json'))['results']
v = np.load('data/index/clean_mv.npy'); meta = []
for i, r in enumerate(rows):
    im = Image.open(locate(r['image'])).convert('RGB')
    vs = [im, normalize_batch([im], steps=('detect',))[0]]
    j = pick_view(vs, v @ embed_images(vs).T)
    p = out / 'crops' / f'{i:03d}.png'; vs[j].save(p)
    meta.append(dict(image=r['image'], source=str(p), view=j, size=vs[j].size))
    if i % 20 == 0: print(i, flush=True)
(out / 'views.json').write_text(json.dumps(meta, indent=2) + '\n')
print('views written', len(meta))
