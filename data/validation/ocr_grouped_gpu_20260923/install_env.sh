#!/bin/bash
# Separate GPU environment for PaddleOCR timing; the project's .venv is not touched.
set -x
python3 -m venv /tmp/rus-vines-paddle-gpu
/tmp/rus-vines-paddle-gpu/bin/pip install --upgrade pip
/tmp/rus-vines-paddle-gpu/bin/pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
/tmp/rus-vines-paddle-gpu/bin/pip install paddleocr==3.7.0 paddlex==3.7.2 opencv-contrib-python==4.10.0.84 numpy==2.3.5 pillow==12.3.0
/tmp/rus-vines-paddle-gpu/bin/python -c "import paddle; print('compiled_with_cuda', paddle.device.is_compiled_with_cuda()); print('gpu count', paddle.device.cuda.device_count()); paddle.utils.run_check()"
echo "=== install done $?"
