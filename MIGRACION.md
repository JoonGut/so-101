# Migración al portátil nuevo (paso a paso)

Portátil antiguo: usuario `<usuario>`, IP `<IP-portátil-antiguo>`, Debian 13, RTX 3050 Ti.
El portátil nuevo tiene **otro usuario**. Todos los comandos se ejecutan **en el portátil nuevo** salvo el paso 0.

---

## 0. En el portátil ANTIGUO (antes de nada)

```bash
# El robot debe estar parado: tiene que decir "finished" o "idle", no "running"
curl -s localhost:8080/api/policy | head -c 80; echo
# rsync hace falta en los dos portátiles
sudo apt install -y rsync
# Cierra la app SenseCraft Robotics (para que no escriba mientras copias)

# Los modelos y la calibración los crea el contenedor como root y sin permiso de lectura:
# sin esto, el rsync del paso 6 se salta los modelos («Permiso denegado»).
# (Ya hecho el 2026-11-22 a las 05:40; repetir si se ha añadido algún modelo después.)
docker exec robot-sorter-web-robot-sorter-1 chmod -R a+rX /app/data/models /app/data/calibration
```

---

## 1. Sistema base en el portátil nuevo (Debian 13 "trixie")

```bash
sudo apt update
sudo apt install -y rsync curl ca-certificates gnupg psmisc firefox-esr openssh-client v4l-utils
```

## 2. Driver NVIDIA (550, como el antiguo)

```bash
# Habilitar contrib/non-free (Debian 13 puede usar cualquiera de los dos formatos)
[ -f /etc/apt/sources.list.d/debian.sources ] && \
  sudo sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
[ -f /etc/apt/sources.list ] && \
  sudo sed -i 's/^\(deb .*trixie[^ ]* main\)$/\1 contrib non-free non-free-firmware/' /etc/apt/sources.list
sudo apt update
sudo apt install -y linux-headers-amd64 nvidia-driver firmware-misc-nonfree
```
> Si la BIOS tiene **Secure Boot** activo, desactívalo (o firma el módulo) o el driver no cargará.

## 3. Docker (repositorio oficial, como el antiguo)

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian trixie stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 4. NVIDIA Container Toolkit (GPU dentro de Docker)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

## 5. Permisos del usuario y reinicio

```bash
sudo usermod -aG dialout,video,docker $USER
sudo reboot
```

Tras reiniciar, comprueba:
```bash
nvidia-smi                                          # debe mostrar la GPU
docker run --rm --gpus all debian:trixie nvidia-smi # GPU visible dentro de Docker
groups                                              # debe incluir dialout video docker
```

---

## 6. Copiar los datos desde el portátil antiguo

```bash
OLD=<usuario>@<IP-portátil-antiguo>
# Puerto SSH del portátil antiguo (22 si no lo has cambiado)
export RSYNC_RSH="ssh -p <puerto-ssh>"
mkdir -p ~/Descargas ~/.config ~/Jon

# Instalador de SenseCraft Robotics 1.2.3 (5,2 GB)
rsync -a --info=progress2 $OLD:Descargas/SenseCraft-Robotics.deb ~/Descargas/

# Datos de SenseCraft: proyecto, modelo ACT, calibraciones, grabaciones (3,2 GB)
rsync -as --info=progress2 "$OLD:.config/SenseCraft Robotics" ~/.config/

# Proyecto robot-sorter-web + run_fluido.py (2 GB)
#   backups/ (copias de seguridad, ~20 GB) no se copia; .cache es caché de HuggingFace.
rsync -a --info=progress2 --exclude .venv --exclude .firefox-camera-profile \
  --exclude /hackthon/backups --exclude .cache \
  $OLD:Jon/hackthon ~/Jon/

# Regla udev de SenseCraft (permisos del puerto serie del brazo)
scp -P <puerto-ssh> $OLD:/etc/udev/rules.d/99-sensecraft-robotics.rules /tmp/
sudo cp /tmp/99-sensecraft-robotics.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```
> `~/.local/share/SenseCraft Robotics` (14 GB) NO se copia: es el runtime y la app lo regenera.
> `.venv` y `.firefox-camera-profile` tampoco: no se usan / `abrir_estacion.sh` lo recrea.

## 7. Cambiar las rutas del usuario antiguo por el nuevo

SenseCraft guarda rutas absolutas `/home/<usuario>/...` en ~69 `state.json`.

```bash
grep -rIlZ --include='*.json' -e '/home/<usuario>/' ~/.config/"SenseCraft Robotics"/backend-storage \
  | xargs -0 -r sed -i "s|/home/<usuario>/|$HOME/|g"

# Comprobación: debe dar 0
grep -rIl --include='*.json' -e '/home/<usuario>/' ~/.config/"SenseCraft Robotics"/backend-storage | wc -l
```

## 8. Instalar SenseCraft Robotics y primer arranque

```bash
sudo apt install -y ~/Descargas/SenseCraft-Robotics.deb
sensecraft-robotics &
```
En la app: espera a que termine de preparar el runtime, y comprueba que aparecen el
proyecto, el modelo importado y las calibraciones. Luego **cierra la app**.

`run_fluido.py` usa el runtime en una carpeta con hash fijo. Comprueba que existe:
```bash
ls ~/.local/share/"SenseCraft Robotics"/r/
# Debe salir: afb1f21fdf9484b9
# Si sale otro nombre (p.ej. XXXX), corrígelo:
#   sed -i 's/afb1f21fdf9484b9/XXXX/' ~/Jon/hackthon/run_fluido.py
```

---

## 9. Conectar brazo y cámaras y fijar los puertos

Conecta los dos brazos y las dos cámaras USB.

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
```
Debes ver:
- `usb-1a86_USB_Single_Serial_5B8E115432-if00` → **follower**
- `usb-1a86_USB_Single_Serial_5B8E116657-if00` → **leader**
- `usb-Wed_Camera_Wed_Camera_20240420112206-video-index0` → **cámara frontal**
- `usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0` → **cámara de la pinza**

Fija los puertos por número de serie (así da igual en qué orden los detecte Linux), igual que en el antiguo
(leader → `/dev/ttyACM0` del contenedor, follower → `/dev/ttyACM1`):

```bash
cat > ~/Jon/hackthon/robot-sorter-web/.env <<'EOF'
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116657-if00
ROBOT_PORT_2=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115432-if00
EOF
```

## 10. Levantar la web del robot

```bash
cd ~/Jon/hackthon/robot-sorter-web
docker compose -f docker-compose.yml -f docker-compose.robot.yml -f docker-compose.gpu.yml up -d --build
docker compose ls                                   # robot-sorter-web running
curl -s localhost:8080/api/policy | head -c 80; echo
```

> **Cámaras:** desde el 2026-11-22 el contenedor las abre por su nombre fijo (`/hostdev/v4l/by-id/...`),
> así que reconectar una cámara ya no obliga a recrearlo, y si una se corta en mitad de una ejecución
> el programa la reabre. Si usas **otras cámaras** (otro número de serie), indica sus nombres de
> `ls /dev/v4l/by-id/` en el `.env`: `CAMERA_FRONT_ID=...` y `CAMERA_GRIPPER_ID=...` (sin `/dev/v4l/by-id/`).
>
> **Brazos:** si desconectas y reconectas un brazo con el contenedor levantado, sigue apuntando al
> puerto viejo. Solución (con el robot parado): repetir el `up -d` de arriba añadiendo `--force-recreate`.
>
> **Firefox:** si la web dice `No se encuentra la cámara «We[bd] Camera»`, cierra el Firefox de la
> estación entero y vuelve a abrirlo con `./abrir_estacion.sh` (no abras la web en tu Firefox normal).

Abrir la estación (Firefox con permiso de cámaras automático):
```bash
./abrir_estacion.sh
```

## 10b. Foto de los recipientes vacíos (tapado de recipientes)

El modelo `lata1-8papel1-5plastico1-6` ve los recipientes que no son el destino como en una foto
de los recipientes vacíos (`server/bin_mask.py`). Esa foto depende de dónde estén la cámara y los
recipientes, así que **hay que repetirla en la mesa nueva**:

1. Vacía los tres recipientes (gris = lata, azul = plástico, blanco = papel), brazo en reposo.
2. Cierra el Firefox de la estación (para liberar la cámara).
3. Haz la foto y dibuja las zonas para comprobar que encajan con los recipientes:
```bash
docker exec robot-sorter-web-robot-sorter-1 python -m server.bin_mask capture
docker exec robot-sorter-web-robot-sorter-1 python -m server.bin_mask zones --out /tmp/zonas.jpg
docker cp robot-sorter-web-robot-sorter-1:/tmp/zonas.jpg ~/zonas.jpg   # ábrela y mírala
```
Si los recuadros verdes no caen sobre los recipientes (la cámara está colocada distinta), hay que
ajustar `polygon` en `calibration/bin_mask/config.json`. Para desactivar el tapado: `"enabled": false`.

Otros ajustes del 2026-11-22 (en `server/run_policy.py`, cambiar y reconstruir el contenedor):
- `RESET_ON_MISS = True`: si un agarre falla, la instrucción se reinicia entera desde la base.
- `AUTO_EMPTY = True`: el umbral de «pinza vacía» se mide al arrancar (en el registro:
  `Pinza cerrada en vacío: X. Cuenta como vacía por debajo de Y.`).

## 11. run_fluido.py (inferencia directa en GPU, sin la web)

Cierra antes SenseCraft. Los números `/dev/videoN` cambian entre portátiles, así que usa rutas fijas:
```bash
cd ~/Jon/hackthon
python3 run_fluido.py \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115432-if00 \
  --front   /dev/v4l/by-id/usb-Wed_Camera_Wed_Camera_20240420112206-video-index0 \
  --gripper /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
```

---

## 12. Verificación final

- [ ] `nvidia-smi` y `docker run --rm --gpus all debian:trixie nvidia-smi` funcionan
- [ ] SenseCraft muestra proyecto, modelo y calibraciones
- [ ] `http://localhost:8080` abre y se ven las dos cámaras
- [ ] Lanzar la política desde la web: la salida dice `Dispositivo de inferencia: cuda`
- [ ] El brazo se mueve correctamente

**Solo cuando todo esto funcione**, apaga y retira el portátil antiguo.

---

## 13. (Opcional) Prefect server del proyecto notebook13

No es parte del robot. Solo si también lo usas:
```bash
# En el ANTIGUO: exportar el volumen de datos
docker run --rm -v notebook13_prefect_data:/v -v ~/:/b debian:trixie tar czf /b/prefect_data.tgz -C /v .

# En el NUEVO:
OLD=<usuario>@<IP-portátil-antiguo>
mkdir -p ~/IA_DATA/sistemas_bd/b6
rsync -a --info=progress2 $OLD:IA_DATA/sistemas_bd/b6/notebook13 ~/IA_DATA/sistemas_bd/b6/
scp $OLD:prefect_data.tgz ~/
docker volume create notebook13_prefect_data
docker run --rm -v notebook13_prefect_data:/v -v ~/:/b debian:trixie tar xzf /b/prefect_data.tgz -C /v
cd ~/IA_DATA/sistemas_bd/b6/notebook13 && docker compose up -d
```
