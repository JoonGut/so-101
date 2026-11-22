# SO-101 · Estación de clasificación

Panel local del hackathon LeRobot Bilbao 2026. Muestra dos cámaras, un gemelo digital y los controles del **simulador** SO-101. También permite calibrar las articulaciones del **robot físico** mediante la API oficial de LeRobot. Los ciclos de clasificación siguen siendo simulados.

El gemelo digital tiene **tres contenedores**. Sus posiciones y colores se definen en `CONTAINER_COORDINATES`, en `server/robot_controller.py`; la vista 3D recibe esa configuración del servidor.

## Arrancar

### En otro PC con Docker

Instala Docker con Docker Compose (Docker Desktop en Windows/macOS o Docker Engine con el complemento Compose en Linux). Descomprime `robot-sorter-web.zip`, abre una terminal dentro de la carpeta `robot-sorter-web` y ejecuta:

```bash
docker compose up -d
```

Abre **http://localhost:8080** en ese mismo PC. En el primer arranque Docker construye la imagen y descarga Python y las dependencias: necesita conexión a Internet. No necesitas instalar Python ni Node en el PC.

Para ver los registros, detener la aplicación o reconstruirla después de cambiar el código:

```bash
docker compose logs -f
docker compose down
docker compose up -d --build
```

Si el puerto 8080 está ocupado, cambia `"8080:8080"` por `"8081:8080"` en `docker-compose.yml` y abre http://localhost:8081.

Las cámaras se seleccionan desde el navegador; no hace falta pasar dispositivos USB al contenedor. El controlador incluido es un simulador.

### Sin Docker

```bash
./run_local.sh
```

Abre la dirección que indica el script, normalmente http://localhost:8080. Necesita Python **3.12 o superior** con las dependencias de `requirements.txt` instaladas. También puedes usar Docker:

```bash
docker compose up --build
```

## Cámaras

- **Frontal:** vista general de la mesa. **Muñeca:** detalle de la pinza.
- Al abrir la web se muestran las capturas guardadas en `static/assets/test_front.jpg` y `test_wrist.jpg`, identificadas como tales.
- En cada selector, elige **Conectar cámara…** y concede permiso. Después podrás seleccionar un dispositivo por su nombre. Las cámaras deben estar conectadas al ordenador donde abres el navegador; no se leen del servidor.
- Para usar cámaras, abre la web desde `localhost` o HTTPS. El navegador puede bloquearlas si accedes por una dirección HTTP de la red local.
- La imagen conserva sus proporciones. Al cambiar de fuente se libera la cámara anterior; si falla la conexión, aparece un aviso y se muestra la captura guardada.
- En la frontal, vacía la mesa y pulsa **Calibrar mesa**. Se comparan imágenes reducidas a 160 × 120, hasta 15 veces por segundo, para detectar cambios. **Esto no identifica el material del objeto ni inicia su clasificación.**
- **Mesa de pruebas** simula un residuo que puedes arrastrar con ratón o pantalla táctil. Selecciona el material y activa el modo automático para probar la clasificación simulada.

## Controles

Los ajustes, el modo y las articulaciones se sincronizan por WebSocket. Sin conexión, los comandos quedan desactivados y la web intenta reconectar cada dos segundos.

**Probar ciclo** inicia una clasificación simulada de papel. **Ir a reposo** cancela el ciclo y restaura la posición inicial de la simulación. **Parada de emergencia** detiene el simulador; **Rearmar estación** vuelve a habilitarlo. Estos botones no sustituyen una parada física de seguridad.

## Calibrar el robot físico con LeRobot

La sección **Calibración de articulaciones** usa `SO101Follower.calibrate()` de **LeRobot 0.6.0**, la misma rutina de calibración del comando `lerobot-calibrate`. No sustituye este procedimiento por una simulación. Referencia: [guía oficial del SO-101](https://huggingface.co/docs/lerobot/main/so101).

### Acceso al USB

En Linux, conecta el SO-101 follower por USB y con su alimentación. Comprueba su puerto con `python -m serial.tools.list_ports`. Si ejecutas la web directamente, ese usuario debe tener permiso para acceder al puerto.

Con Docker en Linux, usa el archivo adicional que expone **solo el puerto del robot** al contenedor. Ajusta el valor de `ROBOT_PORT` al puerto que hayas identificado:

```bash
ROBOT_PORT=/dev/ttyACM0 docker compose -f docker-compose.yml -f docker-compose.robot.yml up -d --build
```

Dentro del contenedor se muestra como `/dev/ttyACM0`, aunque en el ordenador hayas elegido otro puerto. La imagen ya incluye Python 3.12, LeRobot y el SDK Feetech; usa PyTorch para CPU y no necesita GPU. La primera construcción descarga estas dependencias y tarda más que la versión anterior.

El Compose normal (`docker compose up -d`) sigue arrancando la web sin exigir un robot conectado. Para calibrar por USB hay que utilizar el archivo adicional. En Docker Desktop para Windows/macOS, el contenedor no recibe automáticamente los USB del ordenador: necesitas configurar el acceso USB a su máquina Linux o ejecutar la calibración desde un equipo Linux. El puente interactivo de esta web está preparado para Linux.

### Pasos en la web

1. Cierra SenseCraft u otras aplicaciones que estén utilizando el puerto del brazo.
2. Pulsa **Actualizar puertos**, elige el USB y escribe un nombre para el robot, por ejemplo `so101`. Usa el mismo nombre cuando reutilices esa calibración.
3. Pulsa **Calibrar robot**. La web abre la rutina oficial en un proceso independiente y muestra su registro. El simulador queda detenido y en modo manual.
4. Sujeta el brazo: durante la calibración los motores se quedan sin par. Cuando LeRobot lo indique, coloca las articulaciones en el centro de su recorrido y envía una respuesta vacía para pulsar Enter.
5. Sigue la instrucción de LeRobot para recorrer las articulaciones manualmente. No fuerces los topes. Envía Enter cuando hayas terminado. Si encuentra un archivo previo, Enter lo reutiliza; `c` y Enter inicia una nueva calibración.
6. Solo se muestra **Completada** si LeRobot termina correctamente y existe el JSON de calibración. Después puedes rearmar el simulador.

**Cancelar calibración** interrumpe LeRobot para que desconecte el brazo. La parada de la web también cancela la calibración en curso; no equivale a una parada física certificada. El proceso sigue activo si recargas o cierras el navegador: vuelve a abrir la web para continuar o cancelarlo.

En Docker los archivos se guardan en `calibration/<nombre>.json`, una carpeta del ordenador que permanece al recrear el contenedor. En ejecución directa se guardan en `data/calibration/<nombre>.json` (o en `CALIBRATION_DIR`, si defines esa variable). Estos archivos contienen la calibración de los motores; **Calibrar mesa** sigue siendo únicamente la referencia de imagen para visión.

## Código

Sin framework, compilación ni dependencias frontend nuevas:

| Archivo | Responsabilidad |
| --- | --- |
| `static/index.html` | Estructura y etiquetas de la interfaz |
| `static/css/industrial.css` | Estilos y adaptación a móvil |
| `static/js/vision.js` | Vista reutilizable para ambas cámaras, calibración y demo |
| `static/js/controls.js` | Ajustes y presentación de la telemetría |
| `static/js/robot3d.js` | Escena Three.js del gemelo digital |
| `static/js/app.js` | Conexión WebSocket y coordinación de los módulos |
| `static/js/calibration.js` | Panel de calibración y respuestas al proceso de LeRobot |
| `server/main.py` | Servidor FastAPI, API y WebSocket |
| `server/calibration.py` | Proceso interactivo de calibración y sus endpoints |
| `server/calibrate_robot.py` | Llamada a la API oficial de LeRobot |
| `server/robot_controller.py` | Cinemática y ciclos del simulador |

Para añadir un ajuste, incluye su etiqueta en el HTML y una entrada en `ControlDeck.parameters`. Ambas cámaras usan la misma clase `VisionTracker`; solo la frontal activa el análisis.

## Comprobaciones

Pruebas sin hardware ni dependencias de test adicionales, con las dependencias del proyecto instaladas:

```bash
python -m unittest discover -s tests -v
node --test tests/vision.test.cjs
```

Cubren la cancelación de trayectorias, la parada, los disparos duplicados, la calibración de visión y el cambio de cámara durante una solicitud pendiente. El puente de LeRobot se prueba con un proceso falso: instrucciones sin salto de línea, respuestas, cancelación, puertos inválidos, persistencia y bloqueo de comandos. No se accede a motores reales durante las pruebas.
