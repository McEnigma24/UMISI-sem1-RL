# syntax=docker/dockerfile:1
# Środowisko labu l5: Python + zależności z requirements.txt (bez venva na hoście).
FROM python:3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

# Biblioteki systemowe pod MuJoCo / OpenGL oraz xvfb na końcowy render „human”.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

# Mniejszy obraz: PyTorch tylko CPU (wiersz torch== z requirements pomijamy).
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt

COPY docker/entrypoint.sh /usr/local/bin/l5-entrypoint.sh
RUN chmod +x /usr/local/bin/l5-entrypoint.sh

# Snapshot kodu w obrazie; na co dzień montujesz katalog labu w ``l5/docker/run.sh``.
COPY . /workspace

ENTRYPOINT ["/usr/local/bin/l5-entrypoint.sh"]
CMD ["python", "train.py"]
