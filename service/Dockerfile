# Образ сервиса распознавания.
#
# По умолчанию ставится сборка torch под CUDA 12.8 — она нужна видеокарте
# Blackwell, на которой сервис разрабатывался. Для машины без GPU сборку
# можно переключить на процессорную:
#   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# libGL и libglib нужны OpenCV, без них падает импорт cv2
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url "$TORCH_INDEX" torch torchvision \
 && pip install --no-cache-dir \
      transformers \
      ultralytics \
      opencv-python-headless \
      fastapi \
      "uvicorn[standard]" \
      python-multipart \
      pillow \
      numpy

# Веса детектора кладём в образ: иначе ultralytics тянет их при первом
# запросе, и первый скан ждёт загрузку вместо ответа.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" \
 && mkdir -p /app/weights && mv yolo11m.pt /app/weights/yolo11m.pt
ENV YOLO_WEIGHTS=/app/weights/yolo11m.pt

COPY pipeline/embed.py pipeline/imageprep.py pipeline/rerank.py /app/pipeline/
COPY service/main.py service/recommend.py /app/service/

EXPOSE 8080
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
