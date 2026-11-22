FROM python:3.12-slim

# Avoid writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# PyTorch con CUDA para ejecutar el modelo en la GPU (requiere docker-compose.gpu.yml
# y NVIDIA Container Toolkit). Sin GPU funciona igual en CPU.
# Para una imagen más pequeña solo-CPU: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir 'torch>=2.7,<2.12' 'torchvision>=0.22,<0.27' --index-url ${TORCH_INDEX}
RUN pip install --no-cache-dir -r requirements.txt
# Pesos de ResNet18 que usa la política ACT: así no se descargan en cada arranque.
RUN python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"
# Configuración y tokenizador del VLM de SmolVLA (los pesos van en el propio modelo).
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', allow_patterns=['*.json', '*.txt', '*.jinja', '*.model'])"

# Copy application files
COPY server/ ./server/
COPY static/ ./static/

# Expose web port
EXPOSE 8080

# Run FastAPI with uvicorn
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
