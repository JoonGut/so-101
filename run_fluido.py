#!/usr/bin/env python3
"""
Lanza el modelo ACT de SenseCraft Robotics con inferencia en GPU para que el
brazo se mueva fluido.

Diagnostico (logs de la app): la app corre con --policy_device=cpu y cada chunk
tarda ~0.7-0.8 s en inferirse. A 30 fps eso deja la cola de acciones vacia o
con acciones obsoletas ("stale_filtered=45") y el brazo va a tirones. El
runtime de la app ya trae torch 2.10+cu128 con CUDA disponible (RTX 3050 Ti),
solo que no se usa.

Uso:
    1. CIERRA la app SenseCraft Robotics (si no, bloquea puertos y camaras).
    2. python3 run_fluido.py            # usa los valores por defecto
       python3 run_fluido.py --help     # para ajustar puertos, camaras, etc.
    Ctrl+C para detener (cierra cliente y servidor).
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
SC_DATA = HOME / ".local/share/SenseCraft Robotics"
SC_CFG = HOME / ".config/SenseCraft Robotics/backend-storage"

# Rutas tomadas de webui-backend.log de la app
PYTHON = SC_DATA / "r/afb1f21fdf9484b9/runtime/bin/python"
BACKEND = SC_DATA / (
    "product-store/blobs/9f519b98f4d1c3bb253c494ca58d3581f26d0074dac5152d75cf3b662fb6807c/expanded"
)
WORKDIR = SC_CFG / "runtime/robot-client-workdir"
CALIB = SC_CFG / "calibration/lerobot/robots/so_follower"
MODEL = SC_CFG / (
    "projects/proj_7cacd95f/models/model_import_996690c2/bundle/"
    "checkpoints/last/pretrained_model"
)


# Postura de reposo con la que arranca el brazo en las ejecuciones de la app (log de
# robot_client). Si arranca en otra (p.ej. wrist_flex 62 en vez de 46) el modelo no la
# reconoce, duda y se queda quieto en el sitio.
HOME_POSE = {
    "shoulder_pan.pos": 5.0,
    "shoulder_lift.pos": -101.5,
    "elbow_flex.pos": 97.8,
    "wrist_flex.pos": 46.0,
    "wrist_roll.pos": -4.0,
    "gripper.pos": 1.0,
}

HOME_CODE = r"""
import sys, time, json
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
port, calib, target = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
r = SO100Follower(SO100FollowerConfig(port=port, id="so100_follower", calibration_dir=__import__("pathlib").Path(calib)))
r.connect(calibrate=False)
try:
    start = r.get_observation()
    keys = list(target)
    print("[home] desde", {k: round(start[k], 1) for k in keys}, flush=True)
    steps = 120  # 4 s a 30 Hz
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)  # suavizado entrada/salida
        r.send_action({k: start[k] + (target[k] - start[k]) * t for k in keys})
        time.sleep(1 / 30)
    time.sleep(0.5)
    end = r.get_observation()
    print("[home] hasta", {k: round(end[k], 1) for k in keys}, flush=True)
finally:
    r.disconnect()
"""


def go_home(a, env):
    import json
    print("[fluido] llevando el brazo a la postura de inicio ...")
    res = subprocess.run(
        [str(PYTHON), "-c", HOME_CODE, a.robot_port, str(CALIB), json.dumps(HOME_POSE)],
        cwd=WORKDIR, env=env,
    )
    if res.returncode != 0:
        sys.exit("No pude llevar el brazo a la postura de inicio (ver error arriba).")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(MODEL), help="carpeta pretrained_model")
    p.add_argument("--task", default="5", help="texto de tarea (manifest.json: task_text)")
    p.add_argument("--robot-port", default="/dev/ttyACM1", help="puerto serie del follower")
    p.add_argument("--front", default="/dev/video4", help="camara frontal")
    p.add_argument("--gripper", default="/dev/video2", help="camara de la pinza")
    p.add_argument("--device", default="cuda", help="cuda | cpu (cuda es lo que arregla los tirones)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--port", type=int, default=40277, help="puerto del policy server local")
    p.add_argument(
        "--actions-per-chunk", type=int, default=100,
        help="acciones usadas por chunk (el modelo genera 100). Menos = mas reactivo, mas inferencias",
    )
    p.add_argument(
        "--chunk-threshold", type=float, default=0.4,
        help="pide un chunk nuevo cuando la cola baja de esta fraccion. Mas alto = replanifica "
             "mas a menudo; con GPU y valores >0.5 el brazo duda y tarda en arrancar. "
             "Mismo valor que la app: 0.4",
    )
    p.add_argument(
        "--aggregate", default="weighted_average",
        help="como mezclar acciones solapadas entre chunks (weighted_average suaviza)",
    )
    p.add_argument(
        "--smooth", default="off",
        help="suavizado de acciones de la app (interpola a mas Hz): 'auto', un numero "
             "de Hz o 'off'. EXPERIMENTAL: en las pruebas dio 'Port is in use' y paro el brazo",
    )
    p.add_argument(
        "--no-home", action="store_true",
        help="no llevar el brazo a la postura de inicio antes de arrancar",
    )
    return p.parse_args()


def main():
    a = parse()
    for label, path in [("python de la app", PYTHON), ("backend", BACKEND), ("modelo", Path(a.model))]:
        if not Path(path).exists():
            sys.exit(f"No existe {label}: {path}\n(la app pudo actualizarse; revisa rutas arriba)")

    if a.device == "cuda":
        chk = subprocess.run(
            [str(PYTHON), "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
            capture_output=True,
        )
        if chk.returncode != 0:
            sys.exit("CUDA no esta disponible en el runtime de la app. Usa --device cpu o revisa el driver NVIDIA.")

    env = dict(
        os.environ,
        PYTHONPATH=str(BACKEND),
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        WEBUI_RUN_ROBOT_TYPE="so100_follower",
        WEBUI_RS_ROLLOUT_AUTO_HOME="1",
        # igual que la app: las imagenes se envian como JPEG (como el video de
        # entrenamiento). Con "raw" el modelo recibe imagenes distintas y duda.
        WEBUI_CLOUD_OBSERVATION_CODEC="jpeg",
        WEBUI_CLOUD_JPEG_QUALITY="75",
        WEBUI_CLOUD_OBSERVATION_TRANSFER_OPTIMIZATION="1",
    )
    if a.smooth.lower() != "off":
        env["WEBUI_ACTION_SMOOTHING"] = "1"
        env["WEBUI_ACTION_OUTPUT_HZ"] = a.smooth
    cams = (
        f'{{ front: {{ type: opencv, index_or_path: "{a.front}", backend: 200, width: 640, height: 480, fps: {a.fps} }}, '
        f'gripper: {{ type: opencv, index_or_path: "{a.gripper}", backend: 200, width: 640, height: 480, fps: {a.fps} }}}}'
    )

    server_cmd = [
        str(PYTHON), "-m", "backend.run.encoded_policy_server",
        "--host=127.0.0.1", f"--port={a.port}", f"--fps={a.fps}",
    ]
    client_cmd = [
        str(PYTHON), "-m", "backend.run.lerobot_async_robot_client_runtime",
        "--robot.type=so100_follower",
        f"--robot.port={a.robot_port}",
        f"--robot.cameras={cams}",
        "--robot.id=so100_follower",
        f"--task={a.task}",
        f"--server_address=127.0.0.1:{a.port}",
        "--policy_type=act",
        f"--policy_device={a.device}",
        # el cliente solo mueve acciones; la inferencia pesada va en el servidor
        "--client_device=cpu",
        f"--actions_per_chunk={a.actions_per_chunk}",
        f"--chunk_size_threshold={a.chunk_threshold}",
        f"--aggregate_fn_name={a.aggregate}",
        f"--fps={a.fps}",
        f"--pretrained_name_or_path={a.model}",
        f"--robot.calibration_dir={CALIB}",
    ]

    if not a.no_home:
        go_home(a, env)

    procs = []

    def stop(*_):
        for pr in reversed(procs):
            if pr.poll() is None:
                pr.send_signal(signal.SIGINT)
        for pr in procs:
            try:
                pr.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pr.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"[fluido] policy server en {a.device} ...")
    procs.append(subprocess.Popen(server_cmd, cwd=WORKDIR, env=env))
    time.sleep(6)  # la app espera ~6 s a que el servidor este listo
    print("[fluido] cliente del robot ... (Ctrl+C para detener)")
    procs.append(subprocess.Popen(client_cmd, cwd=WORKDIR, env=env))

    while all(pr.poll() is None for pr in procs):
        time.sleep(0.5)
    stop()


if __name__ == "__main__":
    main()
