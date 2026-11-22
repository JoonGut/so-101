"""Clasificación por posición a partir de las demostraciones grabadas.

En lugar de una red que decide cada movimiento, se reutilizan los movimientos humanos
grabados (Oierst/instuction):

1. Con el brazo plegado, la cámara de arriba localiza cada objeto: resta una mesa vacía de
   referencia (sacada del final de las grabaciones) alineada con la vista en vivo, y
   clasifica cada mancha por su color (papel blanco, lata plateada/azul, plástico negro).
2. Se busca la demostración del mismo objeto con la posición más parecida (en un encuadre
   común: las grabaciones se alinean entre sí porque la cámara se movió entre sesiones).
3. Se repite esa trayectoria. Hasta el agarre se le suma la diferencia de postura de agarre
   que predicen las demostraciones vecinas para la nueva posición; el trayecto al cubo y la
   suelta se repiten tal cual (los cubos no se mueven).

Construir la biblioteca:
    python -m server.demos --dataset /ds --out /app/data/calibration/demos
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEMOS_DIR = Path("/app/data/calibration/demos")
REFERENCE_EPISODE = 526        # su mesa vacía final es el encuadre común (cámara bien centrada)
SINGLE = {"Papel": "papel", "Plastico": "plástico", "Lata": "lata"}
MIN_BLOB_PX = 250              # en 640x480
DIFF_THRESHOLD = 40
FPS = 30


# ----------------------------------------------------------------- imagen
def _prep(image_bgr):
    import cv2

    gray = cv2.cvtColor(cv2.resize(image_bgr, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def orange_mask(image_bgr):
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (3, 110, 90), (24, 255, 255))


def align(source_bgr, target_bgr):
    """Afín 2x3 (640x480) que lleva coordenadas de ``source`` a ``target``; None si no encaja."""
    import cv2

    src, dst = _prep(source_bgr), _prep(target_bgr)
    robot = cv2.dilate(cv2.bitwise_or(orange_mask(source_bgr), orange_mask(target_bgr)), np.ones((41, 41), np.uint8))
    mask = cv2.resize(cv2.bitwise_not(robot), (320, 240), interpolation=cv2.INTER_NEAREST)
    (tx, ty), _ = cv2.phaseCorrelate(src, dst)
    warp = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    try:
        score, warp = cv2.findTransformECC(src, dst, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), mask, 5)
    except cv2.error:
        return None, 0.0
    warp = warp.copy()
    warp[:, 2] *= 2.0
    return warp, float(score)


def transform(warp, points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    return points @ warp[:, :2].T + warp[:, 2]


def table_mask(empty_bgr):
    """Zona del tapete gris oscuro (sin cubos, cinta ni brazo), algo encogida para evitar bordes."""
    import cv2

    hsv = cv2.cvtColor(empty_bgr, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 110, 115))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count <= 1:
        return dark
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mat = (labels == biggest).astype(np.uint8) * 255
    # Rellena los agujeros interiores (sin unir con lo de fuera) y encoge para no tocar bordes.
    holes = cv2.bitwise_not(mat)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    h, w = mat.shape
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h and area < 0.02 * h * w:
            mat[labels == i] = 255
    mat = cv2.bitwise_and(mat, cv2.bitwise_not(cv2.dilate(orange_mask(empty_bgr), np.ones((41, 41), np.uint8))))
    return cv2.erode(mat, np.ones((21, 21), np.uint8))


def robot_align(source_bgr, target_bgr):
    """Euclídea 2x3 que lleva la silueta del brazo plegado de ``source`` a la de ``target``."""
    import cv2

    def silhouette(image):
        mask = cv2.resize(orange_mask(image), (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return cv2.GaussianBlur(mask, (0, 0), 3.0)

    src, dst = silhouette(source_bgr), silhouette(target_bgr)
    if src.sum() < 50 or dst.sum() < 50:
        return None, 0.0
    (tx, ty), _ = cv2.phaseCorrelate(src, dst)
    warp = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    try:
        score, warp = cv2.findTransformECC(src, dst, warp, cv2.MOTION_EUCLIDEAN,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        return None, 0.0
    warp = warp.copy()
    warp[:, 2] *= 2.0
    return warp, float(score)


def compose(outer, inner):
    """Afín equivalente a aplicar ``inner`` y luego ``outer``."""
    a = np.vstack([outer, [0, 0, 1]])
    b = np.vstack([inner, [0, 0, 1]])
    return (a @ b)[:2]


def invert(warp):
    import cv2

    return cv2.invertAffineTransform(np.asarray(warp, np.float64))


# Centros de los cubos en la imagen de referencia (encuadre del tapete), medidos sobre
# reference_empty.jpg (episodio 526). Cada cubo recibe su objeto.
BINS_REFERENCE = {"lata": (173.0, 73.0), "plástico": (180.0, 213.0), "papel": (193.0, 353.0)}


def classify_blob(image_bgr, blob):
    """papel (blanco), lata (plateada con azul/rojo) o plástico (negro)."""
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[blob > 0]
    if len(pixels) == 0:
        return None, {}
    value, saturation, hue = pixels[:, 2].astype(float), pixels[:, 1].astype(float), pixels[:, 0].astype(float)
    colored = float(((saturation > 90) & (value > 50)).mean())
    bright = float((value > 150).mean())
    dark = float((value < 45).mean())
    stats = {"colored": round(colored, 3), "bright": round(bright, 3), "dark": round(dark, 3)}
    if dark > 0.45 and bright < 0.15:
        return "plástico", stats
    if colored > 0.06:
        return "lata", stats
    if bright > 0.25:
        return "papel", stats
    return ("plástico" if dark > 0.25 else "lata"), stats


def find_objects(image_bgr, empty_bgr, mask, threshold=DIFF_THRESHOLD, min_px=MIN_BLOB_PX):
    """Manchas que difieren de la mesa vacía dentro del tapete -> [(clase, (x, y), área, stats)]."""
    import cv2

    a = cv2.GaussianBlur(image_bgr, (7, 7), 0).astype(np.int16)
    b = cv2.GaussianBlur(empty_bgr, (7, 7), 0).astype(np.int16)
    diff = np.abs(a - b).max(axis=2).astype(np.uint8)
    robot = cv2.dilate(orange_mask(image_bgr), np.ones((31, 31), np.uint8))
    fg = ((diff > threshold) & (mask > 0) & (robot == 0)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
    found = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        blob = (labels == i).astype(np.uint8)
        kind, color = classify_blob(image_bgr, blob)
        found.append((kind, (float(centroids[i][0]), float(centroids[i][1])), area, color))
    return sorted(found, key=lambda f: -f[2]), fg


# ----------------------------------------------------------------- trayectorias
def grasp_events(actions, states):
    """Índices de cierre de la pinza sobre el objeto y de la suelta en el cubo (o None)."""
    grip = actions[:, 5]
    opened = np.nonzero(grip > 15)[0]
    if len(opened) == 0:
        return None, None
    start = opened[0]
    peak = grip[start]
    close = None
    for i in range(start, len(grip)):
        peak = max(peak, grip[i])
        if grip[i] < 0.5 * peak and grip[i] < 10:
            close = i
            break
    if close is None:
        return None, None
    held = states[min(close + 15, len(states) - 1), 5]
    release = None
    for i in range(close + 15, len(grip)):
        if grip[i] > held + 8:
            release = i
            break
    return close, release


# ----------------------------------------------------------------- biblioteca
def build(dataset, out):
    import cv2
    import pandas as pd

    from .scene import read_frames

    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    episodes = pd.concat([pd.read_parquet(f) for f in sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))])
    episodes = episodes.set_index("episode_index").sort_index()
    data = pd.concat([pd.read_parquet(f) for f in sorted((dataset / "data").glob("*/*.parquet"))])
    key = "videos/observation.images.front"

    def video_of(ep):
        return dataset / "videos" / "observation.images.front" / f"chunk-{int(ep[f'{key}/chunk_index']):03d}" / f"file-{int(ep[f'{key}/file_index']):03d}.mp4"

    def frames(ep):
        start, end = float(ep[f"{key}/from_timestamp"]), float(ep[f"{key}/to_timestamp"])
        got = read_frames(video_of(ep), [start + 0.2, end - 0.2])
        if len(got) < 2:
            return None, None
        first, last = (cv2.cvtColor(got[t], cv2.COLOR_RGB2BGR) for t in sorted(got))
        return first, last

    _, reference = frames(episodes.loc[REFERENCE_EPISODE])
    cv2.imwrite(str(out / "reference_empty.jpg"), reference)
    reference_mask = table_mask(reference)
    cv2.imwrite(str(out / "reference_mask.png"), reference_mask)

    library, rejected = [], {}
    for index, (episode, ep) in enumerate(episodes.iterrows()):
        task = ep["tasks"][0]
        if task not in SINGLE or ep["length"] < 150:
            continue
        rows = data[data["episode_index"] == episode]
        actions = np.stack(rows["action"].values).astype(np.float32)
        states = np.stack(rows["observation.state"].values).astype(np.float32)
        close, release = grasp_events(actions, states)
        if close is None or release is None:
            rejected["sin agarre/suelta"] = rejected.get("sin agarre/suelta", 0) + 1
            continue
        first, last = frames(ep)
        if first is None:
            continue
        warp, score = align(last, reference)              # episodio -> encuadre del tapete
        rwarp, rscore = robot_align(last, reference)       # episodio -> encuadre del robot
        if warp is None or score < 0.85 or rwarp is None or rscore < 0.7:
            rejected["no alinea"] = rejected.get("no alinea", 0) + 1
            continue
        # Objeto = lo que hay al principio y ya no al final (mismo encuadre, misma luz).
        mask = cv2.warpAffine(reference_mask, invert(warp), (640, 480), flags=cv2.INTER_NEAREST)
        found, _ = find_objects(first, last, mask)
        if not found:
            rejected["sin objeto visible"] = rejected.get("sin objeto visible", 0) + 1
            continue
        kind, point, area, color = found[0]
        obj = SINGLE[task]
        robot_point = transform(rwarp, point)[0]
        bin_episode = transform(invert(warp), BINS_REFERENCE[obj])[0]
        bin_robot = transform(rwarp, bin_episode)[0]
        library.append({
            "episode": int(episode), "object": obj, "seen_as": kind, "area": area, "color": color,
            "pixel_raw": [round(float(point[0]), 1), round(float(point[1]), 1)],
            "pixel": [round(float(robot_point[0]), 1), round(float(robot_point[1]), 1)],
            "bin": [round(float(bin_robot[0]), 1), round(float(bin_robot[1]), 1)],
            "close": int(close), "release": int(release), "length": int(len(actions)),
            "grasp": [round(float(v), 2) for v in states[close, :5]],
            "drop": [round(float(v), 2) for v in states[release, :5]],
            "align_score": round(score, 3), "robot_score": round(rscore, 3),
        })
        np.save(out / f"ep{int(episode):04d}.npy", actions)
        if index % 25 == 0:
            print(f"  episodio {episode}: {len(library)} demostraciones", flush=True)
    (out / "library.json").write_text(json.dumps({"demos": library, "rejected": rejected,
                                                  "reference_episode": REFERENCE_EPISODE}, indent=1))
    agree = np.mean([d["seen_as"] == d["object"] for d in library]) if library else 0
    print(f"{len(library)} demostraciones · descartadas {rejected} · color bien clasificado {agree:.0%} "
          f"({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=DEMOS_DIR)
    args = parser.parse_args()
    build(args.dataset, args.out)
