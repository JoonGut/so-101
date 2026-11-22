#!/usr/bin/env bash
# Abre la estación: levanta Docker (robot + cámaras + GPU) si hace falta
# y abre el Firefox especial que da permiso a las dos cámaras sin preguntar.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
URL="http://localhost:8080"
PROFILE="$DIR/.firefox-camera-profile"
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.robot.yml)
if command -v nvidia-smi &> /dev/null; then
    COMPOSE+=(-f docker-compose.gpu.yml)
fi

# 1. Servidor
if ! curl -fs -o /dev/null "$URL/api/policy"; then
    echo "[*] Levantando el servidor con robot, cámaras y GPU…"
    "${COMPOSE[@]}" up -d
    echo -n "[*] Esperando al servidor"
    for _ in $(seq 60); do
        curl -fs -o /dev/null "$URL/api/policy" && break
        echo -n "."
        sleep 1
    done
    echo
fi
curl -fs -o /dev/null "$URL/api/policy" || { echo "[!] El servidor no responde en $URL"; exit 1; }

# 2. Perfil de Firefox con permiso de cámara automático
mkdir -p "$PROFILE"
cat > "$PROFILE/user.js" <<'EOF'
user_pref("media.navigator.permission.disabled", true);
user_pref("permissions.default.camera", 1);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("datareporting.policy.firstRunURL", "");
user_pref("browser.startup.homepage_override.mstone", "ignore");
EOF

if pgrep -f -- "--profile $PROFILE" > /dev/null; then
    echo "[i] El Firefox de la estación ya está abierto."
    exit 0
fi

# 3. Aviso si otro programa tiene ya las cámaras de la estación
for cam in /dev/v4l/by-id/*Wed_Camera*index0 /dev/v4l/by-id/*USB2.0_CAM1*index0; do
    [ -e "$cam" ] || { echo "[!] No está conectada: $(basename "$cam")"; continue; }
    holders=$(fuser "$(readlink -f "$cam")" 2>/dev/null | xargs)
    if [ -n "$holders" ]; then
        names=$(ps -o comm= -p ${holders// /,} | sort -u | xargs)
        echo "[!] $(basename "$cam" | sed 's/-video-index0//') la está usando: $names (PID $holders)."
        echo "    Cierra la pestaña de $URL en tu otro navegador (o SenseCraft) para liberarla."
    fi
done

# 4. Firefox especial
FIREFOX=$(command -v firefox-esr || command -v firefox)
echo "[*] Abriendo $URL…"
nohup "$FIREFOX" --no-remote --profile "$PROFILE" --new-window "$URL" > /dev/null 2>&1 &
