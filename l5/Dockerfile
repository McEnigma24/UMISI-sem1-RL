# syntax=docker/dockerfile:1

FROM python:3.12-bookworm

# Montowanie GPU przez NVIDIA Container Toolkit (``--gpus all`` w ``enter_dev_container.sh``).
# ``utility`` — m.in. ``nvidia-smi`` w kontenerze; ``compute`` — CUDA dla PyTorcha.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

# MuJoCo / OpenGL oraz xvfb (używane przez ``./start.sh`` przy renderze „human”).
RUN apt-get update -y && apt-get upgrade -y

RUN pip install --upgrade pip
RUN pip install gymnasium==1.2.2
RUN pip install gymnasium-robotics==1.4.2
RUN pip install mujoco==3.8.1
RUN pip install numpy==2.2.4
RUN pip install panda-gym==3.0.7
RUN pip install tensorboard==2.19.0
RUN pip install setuptools>=65,<81
RUN pip install standard-imghdr>=1.3.0
RUN pip install torch==2.6.0
RUN pip install tqdm==4.67.1

RUN apt-get install -y --no-install-recommends libegl1
RUN apt-get install -y --no-install-recommends libgl1
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN apt-get install -y --no-install-recommends libgomp1
RUN apt-get install -y --no-install-recommends xauth
RUN apt-get install -y --no-install-recommends xvfb
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
