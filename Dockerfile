FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

# Host CUDA 12.9 driver is backwards-compatible with container CUDA 12.4
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /workspace

# PyTorch with CUDA 12.4 (compatible with host driver 12.9)
RUN pip install --no-cache-dir \
    torch>=2.3.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data checkpoints splits logs eval_output

CMD ["/bin/bash", "run_training.sh"]
