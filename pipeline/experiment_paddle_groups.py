"""Offline automatic text-fragment grouping. Run in isolated Paddle environment.

Consumes lossless selected query views; never uses catalog labels or manual
line boxes. Emits grouped recognition observations for frozen-geometry replay.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def line_groups(boxes):
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[:i]):
            ha, hb = a[3]-a[1], b[3]-b[1]
            dy = abs(a[1]+a[3]-b[1]-b[3])/2
            gap = max(0, max(a[0],b[0])-min(a[2],b[2]))
            if (min(ha,hb)>0 and min(ha,hb)/max(ha,hb)>=.5
                    and dy<=.3*min(ha,hb) and gap<=max(ha,hb)):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return [ids for ids in groups.values() if len(ids)>=2]


def main():
    from PIL import Image
    import numpy as np
    from paddleocr import PaddleOCR, TextRecognition

    ap = argparse.ArgumentParser()
    ap.add_argument('--views', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--det-limit-type', choices=['min', 'max'], default=None,
                    help='detector input limit; default: PaddleOCR defaults (native size, max side 4000)')
    ap.add_argument('--det-limit-side', type=int, default=None)
    args = ap.parse_args()
    rows = json.loads(Path(args.views).read_text())
    det_kwargs = {}
    if args.det_limit_type:
        det_kwargs = dict(text_det_limit_type=args.det_limit_type,
                          text_det_limit_side_len=args.det_limit_side)
    output = Path(args.output)
    completed = []
    if args.resume and output.exists():
        completed = [json.loads(line) for line in output.read_text().splitlines()]
        if [r['image'] for r in completed] != [r['image'] for r in rows[:len(completed)]]:
            raise ValueError('Saved observations do not match the requested prefix')
        for saved, row in zip(completed, rows):
            if hashlib.sha256(Path(row['source']).read_bytes()).hexdigest() != saved['source_sha256']:
                raise ValueError(f'Input changed: {row["image"]}')
    elif output.exists():
        raise FileExistsError('Use a new output path or --resume')
    detector = PaddleOCR(text_detection_model_name='PP-OCRv5_server_det',
                         text_recognition_model_name='cyrillic_PP-OCRv5_mobile_rec',
                         use_doc_orientation_classify=False, use_doc_unwarping=False,
                         use_textline_orientation=False, device='cpu',
                         enable_mkldnn=False, cpu_threads=4)
    recognizer = TextRecognition(model_name='cyrillic_PP-OCRv5_mobile_rec',
                                 device='cpu',enable_mkldnn=False,cpu_threads=4)
    with output.open('a' if args.resume else 'w') as fh:
        for i, row in enumerate(rows[len(completed):], start=len(completed)):
            print(f'Start {i+1}/{len(rows)} {row["image"]}', flush=True)
            start = time.perf_counter()
            image = Image.open(row['source']).convert('RGB')
            detected = list(detector.predict(row['source'], **det_kwargs))[0]
            boxes = [[int(v) for v in b] for b in detected['rec_boxes']]
            words = []
            for ids in line_groups(boxes):
                bs = [boxes[j] for j in ids]
                box = [min(b[0] for b in bs),min(b[1] for b in bs),
                       max(b[2] for b in bs),max(b[3] for b in bs)]
                pad = max(1, round((box[3]-box[1])*.1))
                roi = [max(0,box[0]-pad),max(0,box[1]-pad),
                       min(image.width,box[2]+pad),min(image.height,box[3]+pad)]
                crop = image.crop(roi)
                crop = crop.resize((crop.width*4,crop.height*4),Image.Resampling.LANCZOS)
                res = list(recognizer.predict(np.array(crop)[:,:,::-1].copy(),batch_size=1))[0]
                words.append(dict(text=res['rec_text'],conf=float(res['rec_score']),
                                  box=box,roi=roi,members=ids))
            record = dict(image=row['image'],words=words,boxes=boxes,
                          seconds=time.perf_counter()-start,
                          source_sha256=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest())
            fh.write(json.dumps(record,ensure_ascii=False)+'\n')
            fh.flush()
            print(f'{i+1}/{len(rows)} {row["image"]} {record["seconds"]:.2f}s',flush=True)
    output.with_suffix('.complete.json').write_text(json.dumps({
        'n':len(rows),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'views_sha256':hashlib.sha256(Path(args.views).read_bytes()).hexdigest(),
        'detector_limit':det_kwargs,
        'note':'CPU offline experiment. Only automatically grouped fragments emitted; no manual ROIs.'},indent=2)+'\n')


if __name__=='__main__':
    main()
