#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "  LANZANDO SO-101 ROBOT VISION WORKSTATION CON DOCKER    "
echo "=========================================================="

if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        docker compose up --build
    elif sudo docker info &> /dev/null; then
        echo "[*] Ejecutando con permisos sudo para acceder a docker.sock..."
        sudo docker compose up --build
    else
        echo "[!] Error al conectar con Docker daemon. Comprueba los permisos o el servicio."
        exit 1
    fi
else
    echo "[!] Docker no está instalado en el sistema."
    exit 1
fi
