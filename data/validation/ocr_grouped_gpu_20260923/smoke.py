import time, json, sys
from paddleocr import PaddleOCR, TextRecognition
import numpy as np
from PIL import Image
t=time.perf_counter()
ocr = PaddleOCR(text_detection_model_name='PP-OCRv5_server_det', text_recognition_model_name='cyrillic_PP-OCRv5_mobile_rec',
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                device='gpu', enable_mkldnn=False, cpu_threads=4)
rec = TextRecognition(model_name='cyrillic_PP-OCRv5_mobile_rec', device='gpu', enable_mkldnn=False, cpu_threads=4)
print('load s', round(time.perf_counter()-t,2), flush=True)
for src in sys.argv[1:]:
    for k in range(2):
        t=time.perf_counter(); r=list(ocr.predict(src, text_det_limit_type='max', text_det_limit_side_len=2000))[0]; dt=time.perf_counter()-t
        print(json.dumps(dict(src=src, run=k, seconds=round(dt,2), boxes=len(r['rec_boxes']), texts=[x for x in r['rec_texts']][:6]), ensure_ascii=False), flush=True)
im=np.array(Image.open(sys.argv[1]).convert('RGB'))[:200,:200][:,:,::-1].copy()
t=time.perf_counter(); rr=list(rec.predict(im,batch_size=1))[0]; print('rec s', round(time.perf_counter()-t,3), rr['rec_text'], round(float(rr['rec_score']),3))
