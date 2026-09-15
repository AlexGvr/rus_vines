FROM python:3.12-slim

WORKDIR /app

# torch под Blackwell (sm_120) требует сборку с CUDA 12.8
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision \
 && pip install --no-cache-dir \
      transformers fastapi uvicorn python-multipart pillow numpy

COPY pipeline/embed.py pipeline/imageprep.py /app/pipeline/
COPY service/main.py /app/service/

EXPOSE 8080
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
