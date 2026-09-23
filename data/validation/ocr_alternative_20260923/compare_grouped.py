import sys,json
sys.path.insert(0,'pipeline')
from PIL import Image
from ocr import Word
from experiment_ocr_strips import rank,merge
from text_match import TextChannel
p='data/validation/ocr_alternative_20260923/';r=json.load(open(p+'paddle_grouped.json'))['results'];base={x['image']:x for x in json.load(open('data/validation/final3_tune.json'))['results']};w=json.load(open('data/catalog/catalog.json'))['wines'];ch=TextChannel(w);by={x['slug']:x for x in w};out=[]
for a in r:
 b=base[a['image']];orig=[Word(x['text'],x['conf'],tuple(x['box'])) for x in b['words']];res={'image':a['image'],'gold':b['gold'],'before':b['top1']}
 for mode,key in [('all','words'),('groups_only','grouped')]:
  extra=[Word(x['text'],x['conf'],tuple(x['box'])) for x in a[key] if x['conf']>=.85 and x['text'].strip()];res[mode]=rank(b,Image.new('RGB',(1,1)),merge(orig,extra),ch,by)[0]
 out.append(res)
json.dump(out,open(p+'grouped_comparison.json','w'),ensure_ascii=False,indent=2)
for mode in ['all','groups_only']:
 print(mode,'fixed',sum(x['before']!=x['gold'] and x[mode]==x['gold'] for x in out),'broken',sum(x['before']==x['gold'] and x[mode]!=x['gold'] for x in out))
