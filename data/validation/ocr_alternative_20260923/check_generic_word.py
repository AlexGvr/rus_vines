import sys,json
sys.path.insert(0,'pipeline')
from PIL import Image
import text_match as tm
from ocr import Word
from experiment_ocr_strips import rank,merge
p='data/validation/ocr_alternative_20260923/'
w=json.load(open('data/catalog/catalog.json'))['wines'];by={x['slug']:x for x in w};out={}
for split in ['tune','test']:
 rows=json.load(open('data/validation/final3_'+split+'.json'))['results'];results=[]
 for patched in [False,True]:
  if patched:tm.STOPWORDS.add('игристое')
  else:tm.STOPWORDS.discard('игристое')
  ch=tm.TextChannel(w);orders={}
  for r in rows:
   words=[Word(x['text'],x['conf'],tuple(x['box'])) for x in r['words']]
   orders[r['image']]=rank(r,Image.new('RGB',(1,1)),words,ch,by)[0]
  results.append(orders)
 out[split]={'baseline_replay_mismatches':[r['image'] for r in rows if results[0][r['image']]!=r['top1']], 'changes':[{'image':r['image'],'gold':r['gold'],'before':results[0][r['image']],'after':results[1][r['image']]} for r in rows if results[0][r['image']]!=results[1][r['image']]]}
tm.STOPWORDS.add('игристое');ch=tm.TextChannel(w)
rows={x['image']:x for x in json.load(open('data/validation/final3_tune.json'))['results']};auto=json.load(open(p+'paddle_automatic.json'))['results'];check=[]
for a in auto:
 r=rows[a['image']];orig=[Word(x['text'],x['conf'],tuple(x['box'])) for x in r['words']];extra=[Word(x['text'],x['conf'],tuple(x['box'])) for x in a['words'] if x['conf']>=.85 and x['text'].strip()]
 answer=rank(r,Image.new('RGB',(1,1)),merge(orig,extra),ch,by)[0];check.append({'image':r['image'],'gold':r['gold'],'answer':answer})
out['automatic_with_generic_word_excluded']=check
open(p+'generic_word_diagnostic.json','w').write(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
