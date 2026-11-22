"""Ejecuta una política entrenada de LeRobot en el SO-101 con un bucle propio.

Se usa un bucle propio (en lugar de ``lerobot-rollout``) para aplicar en vivo los
ajustes de la estación mientras el modelo trabaja:

- Velocidad   -> frecuencia a la que se ejecutan las acciones (1,0× = 30 Hz, como al grabar).
- Suavidad    -> filtro exponencial sobre la acción (1 = sin filtro, 10 = muy suave).
- Aceleración -> grados máximos que avanza cada articulación por paso.

El servidor escribe los ajustes en ``params.json``; el bucle lo relee al cambiar.
También guarda lo que ven las cámaras y la postura del brazo para la web.

Señales: SIGINT/SIGTERM -> vuelve a la postura inicial y desconecta.
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path

PREVIEW_DIR = Path(os.environ.get("POLICY_PREVIEW_DIR", "/tmp/policy_preview"))
PARAMS_FILE = PREVIEW_DIR / "params.json"
STATE_FILE = PREVIEW_DIR / "state.json"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BASE_FPS = 30.0
PREVIEW_PERIOD_S = 0.2
RETURN_TIME_S = 3.0
# La pinza no recibe un objetivo más de 15 puntos (0-100) por debajo de su posición real:
# sujeta con firmeza (con 8 se le escurrían los objetos) sin apretar a fondo, lo que
# dispararía la protección por sobrecarga del motor.
GRIP_SQUEEZE = 15.0
# El modelo predice 100 acciones de golpe (3,3 s). Con 0 vuelve a mirar las cámaras en
# cada fotograma y promedia las predicciones solapadas (temporal ensembling de ACT): es lo
# que mejor corrige. Con N > 0 replanifica cada N acciones.
REPLAN_STEPS = 0
# Detección de agarres fallidos (pinza en escala 0-100).
GRIP_OPEN = 12.0       # por encima: pinza abierta para agarrar
GRIP_EMPTY = 9.0       # por debajo: cerrada sin nada dentro (en vacío marca 4-7)
# El umbral de «vacía» se mide en cada ejecución: con el brazo en reposo la pinza está
# cerrada sin nada, y se considera vacía si se cierra hasta ese valor + EMPTY_MARGIN.
# En las grabaciones, sujetando papel marca ~5-8 y plástico ~4-5: con 9 fijo, agarres
# buenos contaban como fallidos. False = usar siempre GRIP_EMPTY.
AUTO_EMPTY = True
EMPTY_MARGIN = 1.0
EMPTY_MIN, EMPTY_MAX = 2.5, GRIP_EMPTY   # límites del umbral medido
GRIP_RELEASE = 8.0     # abrirse esto respecto a cómo sujetaba = soltar en el cubo
GRIP_RELEASE_MIN = 2.0 # abrirse al menos esto antes de quedarse vacía = lo ha soltado
GRIP_CLOSING = 5.0     # cerrarse esto respecto a su máxima apertura = está agarrando
MISS_TICKS = 8         # fotogramas seguidos cerrada en vacío para darlo por fallido
HOLD_TICKS = 15        # fotogramas seguidos cerrada y quieta con algo dentro
RETRY_RETURN_S = 2.0
# Modo automático: si un agarre falla, la instrucción termina ahí y se reinicia entera
# (vuelve a la base, reinicia el modelo y vuelve a mirar la mesa). Tras un fallo el
# modelo casi nunca acierta siguiendo desde donde está; tras el reinicio, sí.
# False = comportamiento anterior (anota el fallo y deja seguir al modelo).
RESET_ON_MISS = True
# Tras cada fallo se prueba otra corrección de la base (grados) alrededor de la actual.
SEARCH_PATTERN = [0.0, 3.0, -3.0, 6.0, -6.0, 9.0, -9.0]
PAN = 0  # índice de shoulder_pan en MOTORS
# Modo automático: la cámara de arriba decide la instrucción.
CLASSIFY_PERIOD_S = 0.05   # cada cuánto se mira la mesa mientras el brazo espera (20 Hz; ~13 ms por lectura en GPU)
STABLE_READINGS = 6        # ventana de lecturas (0,3 s): se promedian sus probabilidades
AGREE_READINGS = 5         # ... y en al menos estas la instrucción ganadora debe ser la primera
MIN_CONFIDENCE = 0.6
MIN_OBJECTS = 0.5          # compuerta: por debajo, la mesa se da por vacía y el brazo no se mueve
AWAY_DEG = 25.0            # el brazo se ha ido de la postura base
HOME_DEG = 8.0             # ... y ha vuelto a ella
HOME_STILL_S = 1.0         # quieto en la base este tiempo = instrucción terminada
EPISODE_MARGIN = 1.15      # tiempo máximo = episodio más largo grabado x 1,15
DEFAULT_EPISODE_S = 40.0
# Postura de reposo de las grabaciones (mediana del primer fotograma de los 738 episodios de
# Oierst/instuction): brazo plegado y pinza cerrada. Los modelos solo han visto empezar desde
# aquí, así que el brazo va a ella antes de empezar y vuelve a ella entre instrucciones.
DATASET_REST = [-7.0, -101.9, 92.4, 73.8, -2.6, 1.2]
REST_MOVE_S = 3.0


class GraspMonitor:
    """Sigue la pinza para saber si un agarre ha fallado.

    buscando -> (se abre) -> agarrando -> cerrada en vacío: «fallo»
                                       -> cerrada y quieta con algo: «agarrado» -> llevando
    llevando -> se abre: «soltado» (en el cubo) -> (se cierra) -> buscando
             -> se queda en vacío: «caído»
    El cierre en vacío después de soltar en el cubo no cuenta como fallo.
    """

    def __init__(self):
        self.attempts = 0
        self.misses = 0
        self.deposits = 0
        self.empty = GRIP_EMPTY  # por debajo: cerrada sin nada dentro
        self.reset()

    def reset(self):
        self.phase = "searching"
        self.count = 0
        self.last = None
        self.held = None
        self.peak = 0.0

    def update(self, gripper):
        event = None
        last, self.last = self.last, gripper
        steady = last is not None and abs(gripper - last) < 1.0
        if self.phase == "searching":
            if gripper >= GRIP_OPEN:
                self.phase, self.count, self.peak = "grasping", 0, gripper
        elif self.phase == "grasping":
            self.peak = max(self.peak, gripper)
            if gripper <= self.empty:
                self.count += 1
                if self.count >= MISS_TICKS:
                    self.attempts += 1
                    self.misses += 1
                    self.reset()
                    event = "miss"
            elif gripper <= self.peak - GRIP_CLOSING and steady:
                self.count += 1
                if self.count >= HOLD_TICKS:
                    self.attempts += 1
                    self.phase, self.count, self.held, self.peak = "carrying", 0, gripper, gripper
                    event = "grabbed"
            else:
                self.count = 0
        elif self.phase == "carrying":
            self.peak = max(self.peak, gripper)
            if gripper >= self.held + GRIP_RELEASE:
                self.deposits += 1
                self.phase = "released"
                event = "released"
            elif gripper <= self.empty:
                self.count += 1
                if self.count >= MISS_TICKS:
                    # Si antes de quedarse vacía se abrió algo, lo ha soltado (en el cubo).
                    if self.peak >= self.held + GRIP_RELEASE_MIN:
                        self.deposits += 1
                        event = "released"
                    else:
                        self.misses += 1
                        event = "dropped"
                    self.reset()
            else:
                self.count = 0
        elif self.phase == "released":
            if gripper <= self.empty:
                self.reset()
        return event

    def summary(self):
        return {"attempts": self.attempts, "misses": self.misses, "deposits": self.deposits}


def clamp(value, low, high):
    return max(low, min(high, float(value)))


def motion_settings(params):
    """Traduce los ajustes de la web a parámetros del bucle de control."""
    speed = clamp(params.get("speed_factor", 1.0), 0.2, 2.0)
    smoothness = clamp(params.get("smoothness_factor", 1.0), 1.0, 10.0)
    acceleration = clamp(params.get("max_acceleration", 2.5), 0.5, 10.0)
    return {
        "speed": speed,
        "fps": BASE_FPS * speed,
        # 1 -> sin filtro (alpha 1); 10 -> alpha 0,2.
        "alpha": 1.0 - (smoothness - 1.0) / 9.0 * 0.8,
        "smoothness": smoothness,
        # Aceleración 2,5 (valor por defecto) -> 15° por paso.
        "max_step": acceleration * 6.0,
        "pan_offset": clamp(params.get("pan_offset", 0.0), -15.0, 15.0),
        "auto_correct": bool(params.get("auto_correct", False)),
    }


def limit_command(command, present, max_step):
    """Limita el avance por paso y la fuerza de cierre de la pinza."""
    limited = {}
    for index, motor in enumerate(MOTORS):
        goal = command[index]
        if motor == "gripper":
            goal = max(goal, present[index] - GRIP_SQUEEZE)
        limited[f"{motor}.pos"] = present[index] + clamp(goal - present[index], -max_step, max_step)
    return limited


def safe_disconnect(robot):
    """Afloja la pinza y quita el par aunque algún motor esté en error."""
    bus = robot.bus
    try:
        present = bus.sync_read("Present_Position")
        bus.write("Goal_Position", "gripper", present["gripper"])
    except Exception:
        pass
    try:
        robot.disconnect()
        return
    except Exception as error:
        print(f"Aviso al desconectar: {error}. Se quita el par motor a motor.", flush=True)
    for motor in MOTORS:
        try:
            bus.write("Torque_Enable", motor, 0, num_retry=3)
        except Exception as error:
            print(f"No se ha podido quitar el par de {motor}: {error}", flush=True)
    try:
        bus.port_handler.closePort()
    except Exception:
        pass
    for camera in robot.cameras.values():
        try:
            camera.disconnect()
        except Exception:
            pass


CAMERA_RECONNECT_S = 15.0  # tiempo máximo para que vuelva una cámara USB desconectada


def get_observation(robot, should_stop=lambda: False):
    """Lee brazo y cámaras. Si una cámara USB se ha desconectado (el cable de la pinza
    se mueve con el brazo), el brazo se queda quieto y se reabre la cámara."""
    try:
        return robot.get_observation()
    except (RuntimeError, TimeoutError):
        broken = {name: cam for name, cam in robot.cameras.items()
                  if cam.thread is None or not cam.thread.is_alive()}
        if not broken:
            raise
    print(f"⚠ Se ha perdido la cámara {', '.join(broken)}. Brazo parado mientras se reconecta…", flush=True)
    deadline = time.perf_counter() + CAMERA_RECONNECT_S
    for name, cam in broken.items():
        try:
            cam.disconnect()
        except Exception:
            pass
        while True:
            try:
                cam.connect()
                break
            except Exception as error:
                if should_stop() or time.perf_counter() > deadline:
                    raise RuntimeError(f"La cámara {name} no ha vuelto en {CAMERA_RECONNECT_S:.0f} s. "
                                       "Revisa su cable USB.") from error
                time.sleep(1.0)
        print(f"✓ Cámara {name} reconectada. Sigue la ejecución.", flush=True)
    return robot.get_observation()


def move_smoothly(robot, goal, seconds, should_stop=lambda: False):
    """Lleva el brazo a una postura con una curva suave (sin tirones)."""
    positions = robot.bus.sync_read("Present_Position")
    current = [positions[m] for m in MOTORS]
    steps = max(1, int(seconds * BASE_FPS))
    for i in range(1, steps + 1):
        if should_stop():
            return False
        t = i / steps
        progress = 3 * t * t - 2 * t * t * t
        target = [c + (g - c) * progress for c, g in zip(current, goal)]
        positions = robot.bus.sync_read("Present_Position")
        robot.send_action(limit_command(target, [positions[m] for m in MOTORS], 30.0))
        time.sleep(1 / BASE_FPS)
    return True


def describe(settings):
    return (f"velocidad {settings['speed']:.1f}× ({settings['fps']:.0f} Hz), "
            f"suavidad {settings['smoothness']:.0f}/10, paso máximo {settings['max_step']:.0f}°, "
            f"corrección base {settings['pan_offset']:+.1f}°" + (" (auto)" if settings["auto_correct"] else ""))


def write_atomic(path, data):
    temporary = path.with_name(f".{path.name}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def model_cameras(config):
    """Cámaras y resolución que espera el modelo, a partir de sus entradas."""
    cameras = {}
    for key, feature in config.input_features.items():
        if key.startswith("observation.images."):
            _, height, width = feature.shape
            cameras[key.removeprefix("observation.images.")] = (width, height)
    return cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--camera", action="append", default=[], help="nombre=ruta")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", default="lata-papel-plastico")
    parser.add_argument("--auto", action="store_true", help="la cámara de arriba elige la instrucción")
    parser.add_argument("--no-rest", action="store_true", help="empezar desde donde esté el brazo (sin ir al reposo grabado)")
    parser.add_argument("--replan", type=int, default=REPLAN_STEPS, help="acciones antes de volver a mirar (0 = cada fotograma, promediando)")
    args = parser.parse_args()

    stop = {"requested": False}

    def request_stop(*_):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in PREVIEW_DIR.glob("*.jpg"):
        old.unlink(missing_ok=True)

    import cv2
    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de inferencia: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

    config = PreTrainedConfig.from_pretrained(args.model)
    config.device = device
    if config.type == "act" and args.replan <= 0:
        # Predice en cada fotograma y promedia las predicciones solapadas (artículo de ACT).
        config.temporal_ensemble_coeff = 0.01
        config.n_action_steps = 1
        print("Corrige en cada fotograma (promediando predicciones)", flush=True)
    elif args.replan > 0 and getattr(config, "n_action_steps", None) and getattr(config, "temporal_ensemble_coeff", None) is None:
        config.n_action_steps = max(1, min(args.replan, config.n_action_steps))
        print(f"Replanifica cada {config.n_action_steps} acciones", flush=True)
    policy = get_policy_class(config.type).from_pretrained(args.model, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=args.model, preprocessor_overrides={"device_processor": {"device": device}},
    )
    print(f"Modelo cargado: {config.type}", flush=True)
    scene = None
    if args.auto:
        from collections import deque
        from server.scene import NOTHING, OUT_OBJECTS, load_scene, scene_available

        if not scene_available():
            raise SystemExit("Falta el detector de la mesa: ejecuta ./entrenar_detector.sh")
        scene = load_scene(device=device)
        history = deque(maxlen=STABLE_READINGS)
        print(f"Modo automático: la cámara de arriba elige la instrucción ({type(scene).__name__}).", flush=True)
        if config.type == "act":
            print("  (Este modelo ACT no lee el texto de la instrucción: la cámara solo decide cuándo empezar.)", flush=True)
    elif config.type != "act":
        print(f"Instrucción: «{args.task}»", flush=True)
    bin_mask = None  # bin_mask: tapar los cubos que no son el destino (opcional, server/bin_mask.py)
    if scene:
        from server.bin_mask import CAMERA as BIN_MASK_CAMERA, load_bin_mask
        bin_mask = load_bin_mask(args.model)
    model_view = None  # bin_mask: última imagen tapada, para la vista previa

    paths = dict(item.split("=", 1) for item in args.camera)
    wanted = model_cameras(config)
    missing = set(wanted) - set(paths)
    if missing:
        raise SystemExit(f"El modelo espera cámaras que no existen en la estación: {', '.join(sorted(missing))}")
    cameras = {
        name: OpenCVCameraConfig(index_or_path=Path(paths[name]), width=width, height=height, fps=30)
        for name, (width, height) in wanted.items()
    }
    robot = SO100Follower(SO100FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        cameras=cameras, use_degrees=True,
    ))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.bus.write_calibration(robot.calibration)
    print("Brazo y cámaras conectados.", flush=True)
    rest = None
    if not args.no_rest:
        print(f"Llevando el brazo al reposo de las grabaciones {DATASET_REST}…", flush=True)
        move_smoothly(robot, DATASET_REST, REST_MOVE_S, lambda: stop["requested"])
        time.sleep(0.5)
        rest = list(DATASET_REST)
        closed_empty = robot.bus.sync_read("Present_Position")["gripper"]  # pinza cerrada sin nada
    print("Empieza la ejecución.", flush=True)

    params_mtime = None
    settings = motion_settings({})
    initial = None
    smoothed = None
    last_preview = 0.0
    start = time.perf_counter()
    last_report = start
    ticks = 0
    grasp = GraspMonitor()
    if AUTO_EMPTY and rest is not None:
        grasp.empty = clamp(closed_empty + EMPTY_MARGIN, EMPTY_MIN, EMPTY_MAX)
        print(f"Pinza cerrada en vacío: {closed_empty:.1f}. Cuenta como vacía por debajo de {grasp.empty:.1f}.", flush=True)
    search_index = 0
    found = None
    target, sent, offset = None, None, 0.0
    task = args.task
    phase = "observing" if scene else "running"
    auto_info = {}
    last_classify = 0.0
    episode_start = 0.0
    episode_limit = DEFAULT_EPISODE_S
    moved_away = False
    home_since = None
    episodes_done = 0

    def publish(observation, present, force=False):
        nonlocal last_preview
        now = time.perf_counter()
        if not force and now - last_preview < PREVIEW_PERIOD_S:
            return
        last_preview = now
        for name in wanted:
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(observation[name], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                write_atomic(PREVIEW_DIR / f"{name}.jpg", encoded.tobytes())
        if model_view is not None:  # bin_mask: lo que ve ACT con los cubos tapados
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(model_view, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                write_atomic(PREVIEW_DIR / "front_act.jpg", encoded.tobytes())
        write_atomic(STATE_FILE, json.dumps({
            **{f"{m}.pos": v for m, v in zip(MOTORS, present)},
            **({"model": target, "sent": [sent[f"{m}.pos"] for m in MOTORS]} if sent else {}),
            "grasp": grasp.summary(), "pan_offset": offset,
            **({"found_pan_offset": found} if found is not None else {}),
            **({"auto": auto_info} if scene else {}),
        }).encode())

    def back_home(reason):
        """Fin de una instrucción: vuelve a la base y espera la siguiente."""
        nonlocal phase, episodes_done, smoothed
        episodes_done += 1
        print(f"■ Instrucción «{task}» terminada ({reason}). Vuelve a la base y espera.", flush=True)
        phase = "returning"
        auto_info.update(phase=phase)
        move_smoothly(robot, initial, RETRY_RETURN_S, lambda: stop["requested"])
        time.sleep(0.8)  # que la escena se asiente antes de volver a mirar
        policy.reset()
        history.clear()
        phase = "observing"
        auto_info.update(phase=phase, instruction=None, episodes=episodes_done)

    try:
        while not stop["requested"] and time.perf_counter() - start < args.duration:
            tick = time.perf_counter()
            # Ajustes en vivo.
            try:
                mtime = PARAMS_FILE.stat().st_mtime
                if mtime != params_mtime:
                    params_mtime = mtime
                    new = motion_settings(json.loads(PARAMS_FILE.read_text()))
                    if new != settings:
                        if new["pan_offset"] != settings["pan_offset"]:
                            search_index = 0  # nueva base: la búsqueda empieza de cero
                            found = None      # ya recogida por el servidor (o la ha cambiado el usuario)
                        settings = new
                        print(f"Ajustes aplicados: {describe(settings)}", flush=True)
            except (OSError, ValueError):
                pass

            observation = get_observation(robot, lambda: stop["requested"])
            present = [observation[f"{motor}.pos"] for motor in MOTORS]
            if initial is None:
                initial = rest or present
                smoothed = list(present)
                print(f"Ajustes iniciales: {describe(settings)}", flush=True)

            # Modo automático: con el brazo en la base, la cámara de arriba decide qué hacer.
            # Mientras el modelo trabaja no se mira (no se le cambia la instrucción a medias).
            if scene and phase == "observing":
                now = time.perf_counter()
                if now - last_classify >= CLASSIFY_PERIOD_S:
                    last_classify = now
                    ranking, objects = scene.analyze(observation["front"])
                    reading = dict(ranking)
                    if objects < MIN_OBJECTS:
                        reading = {label: 0.0 for label in reading}
                        reading[NOTHING] = 1.0
                    history.append(reading)
                    # Promedio de las últimas lecturas: más estable que exigir lecturas idénticas.
                    average = {label: sum(h[label] for h in history) / len(history) for label in reading}
                    ranking = sorted(average.items(), key=lambda item: -item[1])
                    top, confidence = ranking[0]
                    stable = sum(1 for h in history if max(h, key=h.get) == top) if confidence >= MIN_CONFIDENCE else 0
                    auto_info = {
                        "phase": phase, "instruction": None, "episodes": episodes_done,
                        "seen": top, "confidence": round(confidence, 3), "stable": stable, "needed": AGREE_READINGS,
                        "objects_probability": round(objects, 3), "out": getattr(scene, "last_out", None),
                        "objects": OUT_OBJECTS.get(top, []),
                        "ranking": [[label, round(p, 3)] for label, p in ranking[:3]],
                    }
                    if len(history) == STABLE_READINGS and stable >= AGREE_READINGS and top != NOTHING:
                        task = top
                        phase = "running"
                        episode_start = time.perf_counter()
                        episode_limit = scene.durations.get(task, DEFAULT_EPISODE_S) * EPISODE_MARGIN
                        moved_away, home_since = False, None
                        policy.reset()
                        smoothed = list(present)
                        grasp.reset()
                        auto_info.update(phase=phase, instruction=task)
                        print(f"▶ Instrucción «{task}» ({confidence:.0%}): objetos fuera: "
                              f"{', '.join(OUT_OBJECTS.get(task, [])) or '-'}. Máximo {episode_limit:.0f} s.", flush=True)
                publish(observation, present)
                time.sleep(1 / 30)
                continue

            # Corrección de la base: el modelo ve y manda en su propio «sistema» de calibración.
            offset = settings["pan_offset"] + (SEARCH_PATTERN[search_index % len(SEARCH_PATTERN)] if settings["auto_correct"] else 0.0)
            model_state = list(present)
            model_state[PAN] -= offset
            batch = {"observation.state": torch.tensor(model_state, dtype=torch.float32), "task": task}
            for name in wanted:
                frame = observation[name]
                if bin_mask and name == BIN_MASK_CAMERA:  # bin_mask
                    frame = model_view = bin_mask.apply(frame, OUT_OBJECTS.get(task, []))
                image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                batch[f"observation.images.{name}"] = image
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(batch)))
            target = action.squeeze(0).tolist()[: len(MOTORS)]
            target[PAN] += offset

            # Suavidad: filtro exponencial. Aceleración: límite de avance por paso.
            alpha = settings["alpha"]
            for index in range(len(MOTORS)):
                smoothed[index] += alpha * (target[index] - smoothed[index])
            sent = limit_command(smoothed, present, settings["max_step"])
            robot.send_action(sent)

            # Orquestación: si la pinza se cierra en vacío, no se lleva nada al cubo;
            # vuelve a la postura de partida y el modelo empieza otro intento.
            event = grasp.update(present[MOTORS.index("gripper")])
            if event == "grabbed":
                print(f"Objeto agarrado (intento {grasp.attempts}, corrección {offset:+.1f}°). Llevándolo al cubo…", flush=True)
                if settings["auto_correct"] and abs(offset - settings["pan_offset"]) > 0.05:
                    found = offset
                    print(f"✓ Corrección de base que funciona: {offset:+.1f}°. Queda guardada.", flush=True)
            elif event == "released":
                print(f"✓ Residuo depositado ({grasp.deposits} en total).", flush=True)
            elif event in ("miss", "dropped") and not settings["auto_correct"]:
                # Sin ayudas: solo se anota; el brazo sigue exactamente lo que manda el modelo.
                reason = "pinza vacía" if event == "miss" else "se le ha caído"
                print(f"✗ Agarre fallido (intento {grasp.attempts}, {reason}).", flush=True)
                if scene and RESET_ON_MISS:
                    back_home(f"agarre fallido, {reason}: se reinicia")
                    continue
            elif event in ("miss", "dropped"):
                reason = "pinza vacía" if event == "miss" else "se le ha caído"
                print(f"✗ Agarre fallido (intento {grasp.attempts}, {reason}, corrección {offset:+.1f}°). Vuelve y reintenta…", flush=True)
                if settings["auto_correct"]:
                    search_index += 1
                    following = settings["pan_offset"] + SEARCH_PATTERN[search_index % len(SEARCH_PATTERN)]
                    print(f"  Siguiente intento con corrección de base {following:+.1f}°", flush=True)
                write_atomic(STATE_FILE, json.dumps({
                    **{f"{m}.pos": v for m, v in zip(MOTORS, present)}, "grasp": grasp.summary(),
                }).encode())
                if not move_smoothly(robot, initial, RETRY_RETURN_S, lambda: stop["requested"]):
                    break
                policy.reset()
                positions = robot.bus.sync_read("Present_Position")
                smoothed = [positions[m] for m in MOTORS]
                continue

            publish(observation, present)
            now = time.perf_counter()

            # Modo automático: ¿ha terminado la instrucción? (se fue de la base y ha vuelto, o tiempo máximo)
            if scene:
                away = max(abs(present[i] - initial[i]) for i in range(len(MOTORS) - 1))
                if away > AWAY_DEG:
                    moved_away = True
                if moved_away and away < HOME_DEG:
                    home_since = home_since or now
                    if now - home_since >= HOME_STILL_S:
                        back_home("ha vuelto a la base")
                        continue
                else:
                    home_since = None
                if now - episode_start > episode_limit:
                    back_home(f"tiempo máximo de {episode_limit:.0f} s")
                    continue

            ticks += 1
            if now - last_report >= 10:
                print(f"{now - start:.0f} s en ejecución · {ticks / (now - last_report):.0f} Hz reales · {describe(settings)}", flush=True)
                last_report = now
                ticks = 0

            # Velocidad: frecuencia del bucle.
            remaining = 1.0 / settings["fps"] - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if robot.is_connected and initial is not None:
            print("Volviendo a la postura inicial…", flush=True)
            try:
                move_smoothly(robot, initial, RETURN_TIME_S)
                time.sleep(0.5)
            except Exception as error:
                print(f"No se ha podido volver a la postura inicial: {error}", flush=True)
        if robot.is_connected:
            safe_disconnect(robot)
        summary = grasp.summary()
        print(f"Resumen: {summary['attempts']} intentos · {summary['deposits']} depositados · "
              f"{summary['misses']} fallidos.", flush=True)
        print("Ejecución terminada.", flush=True)


if __name__ == "__main__":
    main()
