FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    git wget curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch wheel bundles its own CUDA runtime — no CUDA base image needed
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data checkpoints splits logs eval_output

CMD ["/bin/bash", "run_training.sh"]
