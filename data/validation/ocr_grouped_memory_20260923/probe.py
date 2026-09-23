"""Memory/time probe of the experiment's detector+recognizer call on one view.

Same PaddleOCR constructor as pipeline/experiment_paddle_groups.py; optional
detector input limit passed through to predict(). Prints one JSON line.
"""
import argparse, json, resource, time
from paddleocr import PaddleOCR

ap = argparse.ArgumentParser()
ap.add_argument('source'); ap.add_argument('--limit-type', default=None)
ap.add_argument('--limit-side', type=int, default=None); ap.add_argument('--tag', default='')
a = ap.parse_args()
kw = {}
if a.limit_type: kw = dict(text_det_limit_type=a.limit_type, text_det_limit_side_len=a.limit_side)
ocr = PaddleOCR(text_detection_model_name='PP-OCRv5_server_det',
                text_recognition_model_name='cyrillic_PP-OCRv5_mobile_rec',
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, device='cpu', enable_mkldnn=False, cpu_threads=4)
after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t = time.perf_counter()
res = list(ocr.predict(a.source, **kw))[0]
dt = time.perf_counter() - t
print(json.dumps(dict(tag=a.tag, source=a.source, limit=kw, seconds=round(dt, 2),
                      boxes=len(res['rec_boxes']), maxrss_mb_after_load=round(after_load),
                      maxrss_mb_peak=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))), flush=True)
