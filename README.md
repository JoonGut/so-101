# SO-101 · Estación de clasificación de residuos

Un brazo **SO-101** recoge residuos de la mesa y los deja en su recipiente:

| Recipiente | Residuo |
| --- | --- |
| Gris | Lata |
| Azul | Plástico |
| Blanco | Papel |

La **cámara de arriba** mira la mesa y decide qué hay que recoger. Un modelo **ACT** de
[LeRobot](https://github.com/huggingface/lerobot) mueve el brazo usando esa cámara y la de la
pinza. Todo se controla desde una web local (`http://localhost:8080`) que corre en Docker.

---

## Qué necesitas

**Hardware**
- Brazo SO-101 *follower* (el *leader* solo hace falta para grabar o calibrar).
- Cámara de arriba (`Wed Camera`) y cámara de la pinza (`USB2.0_CAM1`), por USB.
- Los tres recipientes y la alfombrilla, colocados como en las grabaciones.

**Software (Linux; probado en Debian 13)**
- Docker con Docker Compose.
- Opcional, pero recomendado: GPU NVIDIA con su driver y *NVIDIA Container Toolkit*. Sin GPU
  funciona en CPU, pero el brazo va a tirones.
- Firefox ESR (la estación abre un Firefox propio con permiso para las cámaras).
- Internet la primera vez: se construye la imagen de Docker (~12 GB).

**Modelos:** no están en el repositorio porque pesan demasiado (cada uno ocupa 200–400 MB).
Ver el paso 2.

> Para instalar todo desde cero en otro portátil (driver NVIDIA, Docker, SenseCraft…) sigue
> [`MIGRACION.md`](MIGRACION.md).

---

## Estructura

```
so-101/
├── README.md               ← este archivo
├── MIGRACION.md            ← instalación completa en un portátil nuevo
├── run_fluido.py           ← ejecutar el modelo de SenseCraft en GPU sin la web (opcional)
└── robot-sorter-web/
    ├── abrir_estacion.sh   ← ARRANCA TODO (Docker + Firefox de la estación)
    ├── docker-compose*.yml ← base, + robot y cámaras, + GPU
    ├── server/             ← servidor (FastAPI) y control del brazo (run_policy.py)
    ├── static/             ← la web
    ├── calibration/        ← calibración del brazo, detector de la mesa, foto de recipientes
    └── models/             ← modelos entrenados (NO incluidos: hay que añadirlos)
```

---

## Primera puesta en marcha

### 1. Clonar y dar permisos

```bash
git clone https://github.com/JoonGut/so-101.git
cd so-101/robot-sorter-web

# Acceso al puerto serie del brazo, a las cámaras y a Docker (luego cierra sesión o reinicia)
sudo usermod -aG dialout,video,docker $USER
```

### 2. Añadir los modelos

Hay dos formas:

- **Desde la web (lo más fácil):** en el paso **2 · Elegir el modelo**, pulsa **Subir un modelo (.zip)**
  y elige un `.zip` con la carpeta `pretrained_model` de LeRobot (debe usar las cámaras `front`
  y `gripper` y 6 articulaciones).
- **Copiando carpetas:** pon cada modelo en `robot-sorter-web/models/<nombre>/pretrained/`
  (con `config.json`, `model.safetensors` y los procesadores). Por ejemplo, desde otro portátil:
  ```bash
  scp -r usuario@IP-del-otro-portatil:Jon/hackthon/robot-sorter-web/models robot-sorter-web/
  ```

### 3. Comprobar puertos y cámaras

Conecta el brazo y las dos cámaras, y mira sus nombres:

```bash
ls -l /dev/serial/by-id/   # brazos
ls -l /dev/v4l/by-id/      # cámaras
```

Por defecto se usan el brazo en `/dev/ttyACM0` y las dos cámaras originales. Si los tuyos son
distintos, crea `robot-sorter-web/.env`:

```bash
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_XXXXXXXX-if00
CAMERA_FRONT_ID=usb-..._video-index0     # nombre de /dev/v4l/by-id/ (sin la ruta)
CAMERA_GRIPPER_ID=usb-..._video-index0
```

### 4. Arrancar

```bash
cd robot-sorter-web
./abrir_estacion.sh
```

El script levanta Docker (la primera vez construye la imagen y tarda un rato), añade la GPU si
hay `nvidia-smi`, comprueba que las cámaras estén libres y abre el **Firefox de la estación** en
`http://localhost:8080`.

> Abre siempre la web con este script, no con tu Firefox normal: ese no ve las cámaras USB.

---

## Uso diario

1. Ejecuta `./abrir_estacion.sh`.
2. En la web:
   1. **Calibrar la mesa vacía:** quita todo de la mesa, deja el brazo en reposo y pulsa el botón.
      Si los agarres fallan mucho, usa **Alinear la cámara con las grabaciones** y mueve la cámara
      de arriba hasta que la imagen coincida con la referencia.
   2. **Elegir el modelo:** por ejemplo `lata1-8papel1-5plastico1-6 · ACT`.
   3. **Recoger basura:** elige la duración y pulsa **▶ Empezar**.
3. Pon residuos en la mesa. La cámara de arriba detecta qué hay, el brazo lo recoge, lo deja en
   su recipiente, vuelve a la base y espera al siguiente. Si la mesa está vacía, no se mueve.
4. Para terminar, pulsa **■ Parar y volver** (el brazo vuelve a su postura inicial).
   **Parada de emergencia** corta el movimiento al momento.

El **Registro** de la web muestra lo que pasa:

```
▶ Instrucción «Papel» (100%): objetos fuera: papel. Máximo 25 s.
Objeto agarrado (intento 1, corrección +0.0°). Llevándolo al cubo…
✓ Residuo depositado (1 en total).
```

Mientras el modelo trabaja, puedes ver lo que ve el modelo en
`http://localhost:8080/api/policy/preview/front_act.jpg`.

---

## Tapado de recipientes

El modelo `lata1-8papel1-5plastico1-6` se grabó con los recipientes vacíos. Si ya hay algo
dentro de un recipiente, en la imagen que recibe el modelo los recipientes que **no** son el
destino se sustituyen por una foto de los recipientes vacíos (el detector de la mesa sigue
viendo la imagen real). El código está en `server/bin_mask.py` y la configuración en
`calibration/bin_mask/config.json` (`"enabled": false` lo desactiva).

**Hay que repetir la foto** si cambia la posición de la cámara o de los recipientes. Hazlo con
los recipientes vacíos, el brazo en reposo y el Firefox de la estación cerrado:

```bash
docker exec robot-sorter-web-robot-sorter-1 python -m server.bin_mask capture
docker exec robot-sorter-web-robot-sorter-1 python -m server.bin_mask zones --out /tmp/zonas.jpg
docker cp robot-sorter-web-robot-sorter-1:/tmp/zonas.jpg .   # comprueba que los recuadros caen sobre los recipientes
```

---

## Ajustes del comportamiento

Están al principio de `robot-sorter-web/server/run_policy.py`. Si cambias alguno, reconstruye el contenedor.

| Ajuste | Valor | Qué hace |
| --- | --- | --- |
| `RESET_ON_MISS` | `True` | Si un agarre falla, la instrucción se reinicia desde la base en lugar de dejar que el modelo siga desde el fallo. |
| `AUTO_EMPTY` | `True` | Al arrancar mide cuánto marca la pinza cerrada en vacío para distinguir un agarre fallido de uno bueno con papel o plástico. |

---

## Comandos útiles

Desde `robot-sorter-web/`:

```bash
# ¿Está el brazo en marcha? (no reconstruyas mientras diga "running")
curl -s localhost:8080/api/policy | head -c 60; echo

# Reconstruir tras cambiar el código
docker compose -f docker-compose.yml -f docker-compose.robot.yml -f docker-compose.gpu.yml up -d --build

# Registros del servidor / parar todo
docker compose logs -f
docker compose down
```

Sin GPU, quita `-f docker-compose.gpu.yml` de los comandos.

---

## Problemas frecuentes

| Mensaje o síntoma | Solución |
| --- | --- |
| `No se encuentra la cámara «We[bd] Camera». Hay: Internal Camera.` | Cierra la ventana del Firefox de la estación y vuelve a abrirla con `./abrir_estacion.sh`. No uses tu Firefox normal. |
| `Failed to open OpenCVCamera(...)` | La cámara está desconectada u otro programa la está usando. Revisa el cable y cierra otros programas que usen la cámara. |
| `There is no status packet` (motores) | Fallo de comunicación con el brazo. Vuelve a lanzar el modelo. Si se repite, revisa los cables de los motores y no conectes el brazo y las cámaras al mismo hub USB. |
| Has desconectado y reconectado el **brazo** | Con el robot parado, repite el `up -d` de arriba añadiendo `--force-recreate`. (Las cámaras se reconectan solas.) |
| El brazo falla casi todos los agarres | Probablemente la cámara de arriba se ha movido. Usa **Alinear la cámara con las grabaciones** en la web. |

---

## Otros

- **`run_fluido.py`:** ejecuta el modelo de SenseCraft Robotics en GPU sin la web (con la app de
  SenseCraft cerrada). Ver `python3 run_fluido.py --help`.
- **Detector de la mesa:** viene entrenado en `calibration/scene/`. Para reentrenarlo:
  `./entrenar_detector.sh` (descarga ~1,8 GB de vídeos de Hugging Face).
- **Pruebas** (sin hardware): `python -m unittest discover -s tests -v` dentro de `robot-sorter-web/`.
- `robot-sorter-web/README.md` describe la primera versión del panel (simulador y calibración de
  articulaciones con LeRobot).
