# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Warsaw

# BASE #
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends tzdata
RUN ln -snf /usr/share/zoneinfo/Europe/Warsaw /etc/localtime
RUN echo "Europe/Warsaw" > /etc/timezone
RUN apt-get install -y --no-install-recommends libegl1
RUN apt-get install -y --no-install-recommends libgl1
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN apt-get install -y --no-install-recommends libgomp1
RUN apt-get install -y --no-install-recommends xauth
RUN apt-get install -y --no-install-recommends xvfb
RUN apt-get install -y --no-install-recommends ca-certificates
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends build-essential
RUN rm -rf /var/lib/apt/lists/*

# PYTHON # arguments are passed when building -> set in ./config
ARG PY_TAG
ARG PY_VERSION
RUN test -n "${PY_TAG}" || (echo >&2 'ERROR: brak --build-arg PY_TAG. Buduj przez ./docker/enter_dev_container.sh (wartość z l5/config).' && exit 1)
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends software-properties-common gnupg
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends ${PY_VERSION}
RUN apt-get install -y --no-install-recommends ${PY_VERSION}-venv
RUN apt-get install -y --no-install-recommends ${PY_VERSION}-dev
RUN rm -rf /var/lib/apt/lists/*
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN ${PY_VERSION} /tmp/get-pip.py --no-warn-script-location
RUN rm -f /tmp/get-pip.py
RUN ${PY_VERSION} -m pip install --upgrade pip

# REQUIREMENTS #
RUN ${PY_VERSION} -m pip install gymnasium==1.2.2
RUN ${PY_VERSION} -m pip install gymnasium-robotics==1.4.2
RUN ${PY_VERSION} -m pip install mujoco==3.8.1
RUN ${PY_VERSION} -m pip install numpy==2.2.4
RUN ${PY_VERSION} -m pip install panda-gym==3.0.7
RUN ${PY_VERSION} -m pip install tensorboard==2.19.0
RUN ${PY_VERSION} -m pip install 'setuptools>=65,<81'
RUN ${PY_VERSION} -m pip install torch==2.6.0
RUN ${PY_VERSION} -m pip install tqdm==4.67.1

# SYMLINKS #
RUN ln -sf "/usr/bin/${PY_VERSION}" /usr/local/bin/python
RUN ln -sf "/usr/bin/${PY_VERSION}" /usr/local/bin/python3

WORKDIR /workspace
