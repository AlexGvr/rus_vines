import sys,json,re,unicodedata
from pathlib import Path
sys.path.insert(0,'pipeline')
from PIL import Image
from ocr import Word
from experiment_ocr_strips import rank,merge
from text_match import TextChannel
p=Path('data/validation/ocr_alternative_20260923');m=json.load(open(p/'line_manifest.json'));pd=json.load(open(p/'paddle_lines.json'))['results'];easy=json.load(open(p/'easy_lines.json'))
norm=lambda s:''.join(c for c in unicodedata.normalize('NFKC',s).upper() if c.isalnum())
lines=[]
for i,r in enumerate(m):
 vals=[x for x in pd if x['index']==i]
 appropriate=next(x for x in vals if x['model'].startswith(r['language']))
 selected=max((x for x in vals if x['text'].strip()),key=lambda x:x['conf'],default={'text':'','conf':0})
 lines.append(dict(index=i,**r,easy=easy[i],paddle_script_oracle=appropriate,paddle_max_conf=selected,easy_exact=norm(easy[i]['text'])==norm(r['expected']),paddle_exact=norm(appropriate['text'])==norm(r['expected'])))
b={r['image']:r for r in json.load(open('data/validation/final3_tune.json'))['results']};w=json.load(open('data/catalog/catalog.json'))['wines'];ch=TextChannel(w);by={x['slug']:x for x in w};records=[]
auto={x['image']:x for x in json.load(open(p/'paddle_automatic.json'))['results']} if (p/'paddle_automatic.json').exists() else {}
for name in dict.fromkeys(r['image'] for r in m):
 rs=[r for r in lines if r['image']==name];im=Image.open(rs[0]['source']).convert('RGB');r=b[name];orig=[Word(x['text'],x['conf'],tuple(x['box'])) for x in r['words']];before=rank(r,im,orig,ch,by)
 assert before[0]==r['top1'],name
 variants={}
 for mode in ['paddle_script_oracle','paddle_max_conf']:
  extra=[Word(x[mode]['text'],x[mode]['conf'],tuple(x['roi'])) for x in rs if x[mode]['conf']>=.85 and x[mode]['text'].strip()]
  variants[mode]=rank(r,im,merge(orig,extra),ch,by)
 if name in auto:
  extra=[Word(x['text'],x['conf'],tuple(x['box'])) for x in auto[name]['words'] if x['conf']>=.85 and x['text'].strip()]
  variants['automatic']=rank(r,im,merge(orig,extra),ch,by)
 records.append(dict(image=name,gold=r['gold'],group=rs[0]['group'],before=before,variants=variants))
summary={group:{'lines':sum(x['group']==group for x in lines),'easy_exact':sum(x['easy_exact'] for x in lines if x['group']==group),'paddle_exact_script_oracle':sum(x['paddle_exact'] for x in lines if x['group']==group)} for group in ['target','control']}
for group in ['target','control']:
 sub=[r for r in records if r['group']==group];summary[group]['baseline_correct']=sum(r['before'][0]==r['gold'] for r in sub);summary[group]['frames']=len(sub)
 for mode in records[0]['variants']:summary[group][mode+'_correct']=sum(r['variants'][mode][0]==r['gold'] for r in sub)
(p/'comparison.json').write_text(json.dumps(dict(summary=summary,lines=lines,records=records,note='Manual line coordinates and script routing are oracle diagnostics; automatic uses no manual line coordinates. Added words threshold .85 fixed. No calibration or held-out acceptance test.'),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
