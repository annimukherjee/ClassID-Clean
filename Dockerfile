# --- Stage 1: Base Environment ---
FROM python:3.8-slim-bullseye
WORKDIR /app

# --- Stage 2: Install System Dependencies & Fix SSL Certificates ---
# This step must happen first to enable secure downloads in subsequent steps.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      pkg-config \
      git \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
      wget && \
    update-ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 3: Pre-download Model Checkpoints ---
# With the certificates fixed, this step will now succeed.
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN wget --no-check-certificate -O /root/.cache/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth

# --- Stage 4: The Finicky Python Installation ---
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install cython
RUN pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install opencv-python
RUN pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html
RUN pip install mmdet==2.28.2 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN git clone https://github.com/jin-s13/xtcocoapi.git && \
    cd xtcocoapi && \
    python setup.py install && \
    cd .. && rm -rf xtcocoapi
RUN pip install mmpose==0.29.0
RUN pip install mmtrack==0.14.0
RUN pip install facenet-pytorch --no-deps
RUN pip install requests tqdm Pillow scikit-learn

# The final, critical fix for the numpy/lap import error
RUN pip uninstall -y numpy lap && \
    pip install numpy==1.23.5 && \
    pip install --no-cache-dir --force-reinstall --no-binary :all: lap

# --- Stage 5: Copy Application Code ---
COPY . .

# --- Stage 6: Define the Default Command ---
CMD ["python3", "main.py"]