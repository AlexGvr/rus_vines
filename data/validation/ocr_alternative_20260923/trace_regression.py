import sys,json
sys.path.insert(0,'pipeline')
from PIL import Image
from text_match import TextChannel
from ocr import Word
from experiment_ocr_strips import merge
import searchcore as c
p='data/validation/ocr_alternative_20260923/'
r=next(x for x in json.load(open('data/validation/final3_tune.json'))['results'] if x['image']=='7445929_1.jpg');w=json.load(open('data/catalog/catalog.json'))['wines'];ch=TextChannel(w);by={x['slug']:x for x in w};im=Image.open(p+r['image'])
a=next(x for x in json.load(open(p+'paddle_automatic.json'))['results'] if x['image']==r['image'])
orig=[Word(x['text'],x['conf'],tuple(x['box'])) for x in r['words']];extra=[Word(x['text'],x['conf'],tuple(x['box'])) for x in a['words'] if x['conf']>=.85];words=merge(orig,extra)
cs=[c.Candidate(x['slug'],x['cv'],x['inliers'],x['coverage']) for x in r['candidates']];cs.sort(key=lambda x:-x.cv);geo=sorted(cs,key=lambda x:(-x.inliers,-x.cv));cs.sort(key=(lambda x:(-x.inliers,-x.cv)) if geo[0].inliers>=geo[1].inliers*1.5 else lambda x:-x.cv)
trace={'geometry':cs[0].slug,'words':[dict(text=x.text,conf=x.conf,box=x.box) for x in words]}
c.prefer_brand(cs,words,ch,.6);trace['brand']=cs[0].slug
c.resolve(cs,im,ch,lambda *_:words,.8,.6,by);trace['resolve']=cs[0].slug;trace['reasons']={x.slug:x.contradictions for x in cs if x.contradictions}
c.demote_contradicted(cs,ch,words,.6);trace['final']=cs[0].slug
open(p+'regression_trace.json','w').write(json.dumps(trace,ensure_ascii=False,indent=2)+'\n');print(json.dumps(trace,ensure_ascii=False,indent=2))
