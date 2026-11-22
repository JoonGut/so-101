"""Clasifica los objetos de la mesa repitiendo demostraciones grabadas (ver server/demos.py).

Ciclo: brazo plegado -> la cámara de arriba localiza los objetos -> se elige uno -> se repite
la demostración más cercana (corrigiendo el giro de la base) -> se comprueba que la pinza
lleva algo -> se suelta en su cubo -> vuelve al reposo. Sin objetos, el brazo no se mueve.

Señales: SIGINT/SIGTERM -> vuelve al reposo y desconecta.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from server.demos import (BINS_REFERENCE, DEMOS_DIR, align, find_objects, invert, robot_align,
                          transform)
from server.run_policy import (DATASET_REST, GRIP_EMPTY, MOTORS, PARAMS_FILE, PREVIEW_DIR, STATE_FILE,
                               clamp, limit_command, move_smoothly, safe_disconnect, write_atomic)

FPS = 30.0
LIFT_FRAMES = 30           # tras cerrar la pinza, frames de la demostración que se repiten (levantar)
APPROACH_FRAMES = 45       # frames antes de soltar que se repiten de la demostración del cubo
AFTER_RELEASE = 25
TRANSFER_S = 1.2
REST_S = 2.5
MAX_PAN_FIX = 12.0
NEIGHBOURS = 15
STABLE_SCENES = 2          # observaciones seguidas con los mismos objetos
MAX_TRIES = 3              # intentos por objeto (con demostraciones distintas)
HOLD_MIN = GRIP_EMPTY + 1  # pinza por encima de esto tras levantar = lleva algo


class Library:
    def __init__(self, directory=DEMOS_DIR):
        import cv2

        meta = json.loads((directory / "library.json").read_text())
        self.demos = meta["demos"]
        self.directory = directory
        self.reference = cv2.imread(str(directory / "reference_empty.jpg"))
        self.mask = cv2.imread(str(directory / "reference_mask.png"), 0)
        self.cache = {}

    def actions(self, demo):
        episode = demo["episode"]
        if episode not in self.cache:
            self.cache[episode] = np.load(self.directory / f"ep{episode:04d}.npy")
        return self.cache[episode]

    def of(self, obj):
        return [d for d in self.demos if d["object"] == obj]

    @staticmethod
    def local_pan(demos, key, pose, point):
        """Giro de base previsto en ``point`` (regresión lineal ponderada con los vecinos)."""
        xy = np.array([d[key] for d in demos], float)
        pan = np.array([d[pose][0] for d in demos], float)
        dist = np.linalg.norm(xy - point, axis=1)
        nearest = np.argsort(dist)[:NEIGHBOURS]
        weights = 1.0 / (dist[nearest] + 10.0)
        a = np.c_[xy[nearest], np.ones(len(nearest))] * weights[:, None]
        coef, *_ = np.linalg.lstsq(a, pan[nearest] * weights, rcond=None)
        return float(np.r_[point, 1.0] @ coef)

    def plan(self, obj, point, bin_point, skip=()):
        demos = self.of(obj)
        if not demos:
            return None
        xy = np.array([d["pixel"] for d in demos], float)
        order = [i for i in np.argsort(np.linalg.norm(xy - point, axis=1)) if demos[i]["episode"] not in skip]
        if not order:
            return None
        grasp = demos[order[0]]
        bins = np.array([d["bin"] for d in demos], float)
        drop = demos[int(np.argmin(np.linalg.norm(bins - bin_point, axis=1)))]
        pan_fix = clamp(self.local_pan(demos, "pixel", "grasp", point)
                        - self.local_pan(demos, "pixel", "grasp", np.array(grasp["pixel"])), -MAX_PAN_FIX, MAX_PAN_FIX)
        drop_fix = clamp(self.local_pan(demos, "bin", "drop", bin_point)
                         - self.local_pan(demos, "bin", "drop", np.array(drop["bin"])), -MAX_PAN_FIX, MAX_PAN_FIX)
        return {
            "object": obj, "grasp": grasp, "drop": drop, "pan_fix": pan_fix, "drop_fix": drop_fix,
            "distance": float(np.linalg.norm(np.array(grasp["pixel"]) - point)),
        }


def observe(library, image):
    """-> (objetos [(clase, punto en encuadre del robot, punto en imagen, área)], cubos {clase: punto robot}, info)."""
    import cv2

    warp, score = align(library.reference, image)           # referencia -> en vivo (tapete)
    rwarp, rscore = robot_align(image, library.reference)   # en vivo -> referencia (robot)
    if warp is None or rwarp is None or score < 0.8 or rscore < 0.6:
        return None, None, {"error": "no se reconoce la escena", "score": score, "robot_score": rscore}
    empty = cv2.warpAffine(library.reference, warp, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(library.mask, warp, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
    found, _ = find_objects(image, empty, mask)
    objects = [(kind, transform(rwarp, point)[0], point, area) for kind, point, area, _ in found if kind]
    bins = {obj: transform(rwarp, transform(warp, p)[0])[0] for obj, p in BINS_REFERENCE.items()}
    return objects, bins, {"score": round(score, 3), "robot_score": round(rscore, 3)}


def build_trajectory(library, plan, start_pose):
    """Lista de objetivos (6 articulaciones) a 30 Hz y el índice tras el que se comprueba el agarre."""
    grasp_actions = library.actions(plan["grasp"]).copy()
    drop_actions = library.actions(plan["drop"]).copy()
    close, release = plan["grasp"]["close"], plan["drop"]["release"]

    # A: demostración hasta levantar el objeto, con la corrección de giro entrando poco a poco.
    a = grasp_actions[: min(len(grasp_actions), close + LIFT_FRAMES)]
    ramp = np.clip(np.arange(len(a)) / max(1, close - 30), 0, 1)
    a[:, 0] += plan["pan_fix"] * ramp
    # Comienza exactamente donde está el brazo (la demo empieza en un reposo muy parecido).
    blend = np.clip(np.arange(len(a)) / 15.0, 0, 1)[:, None]
    a = start_pose * (1 - blend) + a * blend
    check = len(a) - 1

    # C: aproximación al cubo y suelta, de la demostración que soltó en un cubo en ese sitio.
    begin = max(0, release - APPROACH_FRAMES)
    c = drop_actions[begin: min(len(drop_actions), release + AFTER_RELEASE)].copy()
    c[:, 0] += plan["drop_fix"]
    # Mientras se aproxima al cubo la pinza sigue cerrada como al levantar.
    hold = a[-1, 5]
    c[: release - begin, 5] = np.minimum(c[: release - begin, 5], hold)

    # B: transición suave de A a C sin soltar.
    steps = int(TRANSFER_S * FPS)
    t = np.linspace(0, 1, steps)[:, None]
    smooth = 3 * t * t - 2 * t * t * t
    b = a[-1] * (1 - smooth) + c[0] * smooth
    b[:, 5] = hold
    return np.vstack([a, b, c]), check


def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--camera", action="append", default=[], help="nombre=ruta")
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--library", type=Path, default=DEMOS_DIR)
    parser.add_argument("--max-objects", type=int, default=0, help="parar tras depositar N (0 = sin límite)")
    args = parser.parse_args()

    stop = {"requested": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in PREVIEW_DIR.glob("*.jpg"):
        old.unlink(missing_ok=True)

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    library = Library(args.library)
    print(f"Biblioteca: {len(library.demos)} demostraciones "
          f"({', '.join(f'{o}: {len(library.of(o))}' for o in BINS_REFERENCE)}).", flush=True)
    paths = dict(item.split("=", 1) for item in args.camera)
    robot = SO100Follower(SO100FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        cameras={"front": OpenCVCameraConfig(index_or_path=Path(paths["front"]), width=640, height=480, fps=30)},
        use_degrees=True,
    ))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.bus.write_calibration(robot.calibration)
    print("Brazo y cámara conectados. Llevando el brazo al reposo…", flush=True)
    move_smoothly(robot, DATASET_REST, 3.0, lambda: stop["requested"])
    time.sleep(0.5)

    speed = {"factor": 1.0, "mtime": None}
    info = {"phase": "observing", "deposits": 0, "attempts": 0, "misses": 0, "objects": [], "target": None}
    last_preview = [0.0]

    def read_speed():
        try:
            mtime = PARAMS_FILE.stat().st_mtime
            if mtime != speed["mtime"]:
                speed["mtime"] = mtime
                speed["factor"] = clamp(json.loads(PARAMS_FILE.read_text()).get("speed_factor", 1.0), 0.3, 1.5)
        except (OSError, ValueError):
            pass
        return speed["factor"]

    def publish(observation, force=False):
        now = time.perf_counter()
        if not force and now - last_preview[0] < 0.2:
            return
        last_preview[0] = now
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(observation["front"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            write_atomic(PREVIEW_DIR / "front.jpg", encoded.tobytes())
        write_atomic(STATE_FILE, json.dumps({
            **{f"{m}.pos": observation[f"{m}.pos"] for m in MOTORS},
            "grasp": {"attempts": info["attempts"], "misses": info["misses"], "deposits": info["deposits"]},
            "auto": {**info, "seen": ", ".join(o for o in info["objects"]) or "Nada",
                     "instruction": info["target"], "episodes": info["deposits"]},
        }).encode())

    def present():
        positions = robot.bus.sync_read("Present_Position")
        return np.array([positions[m] for m in MOTORS], float)

    def run(trajectory, check=None):
        """Sigue la trayectoria; en ``check`` devuelve si la pinza lleva algo (o None si se ha parado)."""
        factor = read_speed()
        for index, target in enumerate(trajectory):
            if stop["requested"]:
                return None
            tick = time.perf_counter()
            current = present()
            robot.send_action(limit_command(list(target), list(current), 20.0))
            if index % 6 == 0:
                publish(robot.get_observation())
            if check is not None and index == check:
                time.sleep(0.3)
                grip = present()[5]
                print(f"  pinza tras levantar: {grip:.1f}", flush=True)
                if grip < HOLD_MIN:
                    return False
            remaining = 1.0 / (FPS * factor) - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
        return True

    def back_to_rest():
        move_smoothly(robot, DATASET_REST, REST_S, lambda: stop["requested"])
        time.sleep(0.6)

    start = time.perf_counter()
    previous, stable = None, 0
    failed = {}  # objeto -> episodios ya probados sin éxito en esta posición
    try:
        while not stop["requested"] and time.perf_counter() - start < args.duration:
            observation = robot.get_observation()
            image = cv2.cvtColor(observation["front"], cv2.COLOR_RGB2BGR)
            objects, bins, scene = observe(library, image)
            info.update(phase="observing", target=None, scene=scene)
            if objects is None:
                info["objects"] = []
                publish(observation)
                time.sleep(0.3)
                continue
            info["objects"] = [kind for kind, *_ in objects]
            publish(observation)
            signature = sorted(info["objects"])
            stable = stable + 1 if signature == previous else 1
            previous = signature
            if not objects:
                time.sleep(0.3)   # mesa vacía: el brazo se queda quieto
                continue
            if stable < STABLE_SCENES:
                time.sleep(0.1)
                continue

            # Se elige el objeto con la demostración más parecida.
            options = []
            for kind, point, pixel, area in objects:
                plan = library.plan(kind, point, bins[kind], skip=failed.get(kind, ()))
                if plan:
                    options.append((plan["distance"], kind, point, pixel, plan))
            if not options:
                print("Ningún objeto tiene demostraciones disponibles; espero.", flush=True)
                failed.clear()
                time.sleep(1.0)
                continue
            _, kind, point, pixel, plan = min(options, key=lambda o: o[0])
            info.update(phase="running", target=kind)
            info["attempts"] += 1
            print(f"▶ {kind} en ({pixel[0]:.0f}, {pixel[1]:.0f}) px · demostración {plan['grasp']['episode']} "
                  f"a {plan['distance']:.0f} px, giro {plan['pan_fix']:+.1f}° · cubo según demo {plan['drop']['episode']} "
                  f"({plan['drop_fix']:+.1f}°)", flush=True)
            trajectory, check = build_trajectory(library, plan, present())
            result = run(trajectory, check)
            if result is None:
                break
            if result:
                info["deposits"] += 1
                failed.pop(kind, None)
                print(f"✓ {kind} depositado ({info['deposits']} en total).", flush=True)
            else:
                info["misses"] += 1
                failed.setdefault(kind, []).append(plan["grasp"]["episode"])
                if len(failed[kind]) >= MAX_TRIES:
                    failed[kind] = failed[kind][-MAX_TRIES:]
                print(f"✗ {kind}: la pinza está vacía; vuelvo al reposo y lo intento con otra demostración.", flush=True)
            info["phase"] = "returning"
            back_to_rest()
            previous, stable = None, 0
            if args.max_objects and info["deposits"] >= args.max_objects:
                break
    finally:
        if robot.is_connected:
            print("Volviendo al reposo…", flush=True)
            try:
                move_smoothly(robot, DATASET_REST, REST_S)
                time.sleep(0.5)
            except Exception as error:
                print(f"No se ha podido volver al reposo: {error}", flush=True)
            safe_disconnect(robot)
        print(f"Resumen: {info['attempts']} intentos · {info['deposits']} depositados · {info['misses']} fallidos.", flush=True)
        print("Ejecución terminada.", flush=True)


if __name__ == "__main__":
    main()
