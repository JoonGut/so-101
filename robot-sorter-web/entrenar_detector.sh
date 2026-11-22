#!/usr/bin/env bash
# Entrena el detector de la mesa (cámara de arriba -> instrucción) con el dataset de
# instrucciones de Hugging Face. Descarga solo los vídeos de la cámara frontal (~1,8 GB).
#   ./entrenar_detector.sh                 # dataset Oierst/instuction
#   ./entrenar_detector.sh usuario/dataset # otro dataset con las mismas instrucciones
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
REPO="${1:-Oierst/instuction}"
DATA="$DIR/data/$(echo "$REPO" | tr '/' '_')"
IMAGE=robot-sorter-web-robot-sorter
GPU=()
command -v nvidia-smi > /dev/null && GPU=(--gpus all)

docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "[!] Falta la imagen: ejecuta ./abrir_estacion.sh"; exit 1; }
mkdir -p "$DATA"
echo "[*] Descargando $REPO (metadatos + vídeos de la cámara frontal)…"
docker run --rm -v "$DATA:/ds" "$IMAGE" python -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', repo_type='dataset', local_dir='/ds', max_workers=8,
                  allow_patterns=['meta/*', 'meta/*/*', 'meta/*/*/*', 'videos/observation.images.front/*/*.mp4'])
" 2>&1 | grep -v "unauthenticated"
echo "[*] Entrenando el detector…"
docker run --rm "${GPU[@]}" -v "$DATA:/ds:ro" -v "$DIR/calibration:/app/data/calibration" \
    -v "$DIR/server:/app/server:ro" "$IMAGE" python -m server.scene --dataset /ds
echo "[*] Listo. En la web, paso 3 → «Automático»."
