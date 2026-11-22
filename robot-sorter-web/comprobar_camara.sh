#!/usr/bin/env bash
# Saca de las grabaciones de SenseCraft cómo veía la cámara frontal la mesa al grabar
# y abre la estación para alinear la cámara (paso 1 → «Alinear la cámara con las grabaciones»).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
DATASETS="${1:-$HOME/.config/SenseCraft Robotics/backend-storage/projects}"
IMAGE=robot-sorter-web-robot-sorter

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "[!] Falta la imagen de Docker. Ejecuta antes ./abrir_estacion.sh"
    exit 1
fi
# Todas las carpetas «datasets» de los proyectos de SenseCraft.
mapfile -t FOUND < <(find "$DATASETS" -maxdepth 3 -type d -name datasets 2>/dev/null)
if [ ${#FOUND[@]} -eq 0 ]; then
    echo "[!] No hay grabaciones en: $DATASETS"
    echo "    Uso: $0 [carpeta con las grabaciones]"
    exit 1
fi
echo "[*] Analizando las grabaciones de: ${FOUND[0]}"
docker run --rm \
    -v "${FOUND[0]}:/datasets:ro" \
    -v "$DIR/calibration:/app/data/calibration" \
    -v "$DIR/server:/app/server:ro" \
    "$IMAGE" python -m server.reference --datasets /datasets
echo
echo "[*] Abriendo la estación: paso 1 → «Alinear la cámara con las grabaciones»."
./abrir_estacion.sh
