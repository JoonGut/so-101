"""Comparar la cámara frontal actual con la de las grabaciones de entrenamiento.

Si la cámara frontal se ha movido respecto a cuando se grabó, el modelo apunta
desviado. Aquí se saca una imagen de referencia de las grabaciones y se mide
cuánto se ha desplazado/girado la cámara en directo.

Uso como script (lo lanza ``comprobar_camara.sh``):
    python -m server.reference --datasets /datasets --out /app/data/calibration/reference
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path("/app/data/calibration/reference")
REFERENCE_IMAGE = REFERENCE_DIR / "front.jpg"
REPORT_FILE = REFERENCE_DIR / "report.json"
FRAMES_PER_DATASET = 40
ALIGNED_PX = 6       # por debajo: alineada
ALIGNED_DEG = 0.7


# ----------------------------------------------------------------- medición
def _prepare(image_bgr):
    import cv2

    gray = cv2.cvtColor(cv2.resize(image_bgr, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def _mask(reference_bgr, live_bgr):
    """Ignora el brazo (naranja) en cualquiera de las dos imágenes."""
    import cv2

    def orange(image):
        hsv = cv2.cvtColor(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (5, 120, 120), (22, 255, 255))
    robot = cv2.dilate(cv2.bitwise_or(orange(reference_bgr), orange(live_bgr)), np.ones((15, 15), np.uint8))
    return cv2.bitwise_not(robot)


def estimate_shift(reference_bgr, live_bgr):
    """Desplazamiento, giro y escala de la imagen en directo respecto a la referencia.

    dx > 0: lo que se ve está más a la derecha que en las grabaciones.
    dy > 0: más abajo. angle > 0: girada en sentido horario. Devuelve None si no
    se consigue encajar una imagen con la otra.
    """
    import cv2

    reference = _prepare(reference_bgr)
    live = _prepare(live_bgr)
    mask = _mask(reference_bgr, live_bgr)
    # Primero una estimación gruesa del desplazamiento y luego se afina con giro y escala.
    (tx, ty), response = cv2.phaseCorrelate(reference, live)
    best = None
    for motion in (cv2.MOTION_AFFINE, cv2.MOTION_EUCLIDEAN):
        warp = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
        try:
            score, warp = cv2.findTransformECC(
                reference, live, warp, motion,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), mask, 5)
        except cv2.error:
            continue
        if best is None or score > best[0]:
            best = (score, warp)
        break
    if best is None or best[0] < 0.5:
        return None
    score, warp = best
    # warp lleva coordenadas de la referencia a la imagen en directo (escala 320x240 -> x2).
    a, b = float(warp[0, 0]), float(warp[1, 0])
    scale = math.sqrt(abs(float(np.linalg.det(warp[:, :2]))))
    angle = math.degrees(math.atan2(b, a))
    cx, cy = 160.0, 120.0
    mx = warp[0, 0] * cx + warp[0, 1] * cy + warp[0, 2]
    my = warp[1, 0] * cx + warp[1, 1] * cy + warp[1, 2]
    return {
        "dx": round(float(mx - cx) * 2, 1), "dy": round(float(my - cy) * 2, 1),
        "angle": round(angle, 2), "scale": round(scale, 3), "score": round(float(score), 3),
    }


def describe_shift(shift):
    """Indicaciones en lenguaje claro para mover la cámara."""
    if shift is None:
        return {"aligned": False, "message": "No se reconoce la escena: comprueba que la cámara ve la mesa."}
    hints = []
    if abs(shift["dx"]) >= ALIGNED_PX:
        side = "derecha" if shift["dx"] > 0 else "izquierda"
        hints.append(f"la imagen está {abs(shift['dx']):.0f} px a la {side}")
    if abs(shift["dy"]) >= ALIGNED_PX:
        side = "abajo" if shift["dy"] > 0 else "arriba"
        hints.append(f"{abs(shift['dy']):.0f} px más {side}")
    if abs(shift["angle"]) >= ALIGNED_DEG:
        sense = "horario" if shift["angle"] > 0 else "antihorario"
        hints.append(f"girada {abs(shift['angle']):.1f}° en sentido {sense}")
    if abs(shift["scale"] - 1) >= 0.02:
        hints.append("más cerca de la mesa" if shift["scale"] > 1 else "más lejos de la mesa")
    if not hints:
        return {"aligned": True, "message": "✓ Cámara alineada con las grabaciones."}
    return {"aligned": False, "message": "Respecto a las grabaciones, " + ", ".join(hints) + "."}


# ----------------------------------------------------------------- extracción
def sample_frames(video, count):
    """``count`` fotogramas repartidos por todo el vídeo."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if duration is None and container.duration:
            duration = container.duration / 1_000_000
        if not duration:
            return []
        frames = []
        for i in range(count):
            target = duration * (i + 0.5) / count
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is None or frame.time >= target - 0.05:
                    frames.append(frame.to_ndarray(format="bgr24"))
                    break
        return frames


def extract(datasets_dir, out_dir):
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    medians = {}
    episodes = {}
    for info in sorted(datasets_dir.glob("*/meta/info.json")):
        dataset = info.parent.parent
        meta = json.loads(info.read_text())
        if not meta.get("total_episodes"):
            continue
        videos = sorted((dataset / "videos" / "observation.images.front").glob("*/*.mp4"))
        if not videos:
            continue
        per_video = max(4, FRAMES_PER_DATASET // len(videos))
        frames = [f for video in videos for f in sample_frames(video, per_video)]
        if not frames:
            continue
        medians[dataset.name] = np.median(np.stack(frames), axis=0).astype(np.uint8)
        episodes[dataset.name] = meta["total_episodes"]
        print(f"{dataset.name}: {meta['total_episodes']} episodios, {len(frames)} fotogramas", flush=True)
    if not medians:
        raise SystemExit("No se han encontrado grabaciones con cámara frontal.")

    # ¿Se movió la cámara entre sesiones? Se compara cada sesión con todas las demás
    # y la referencia es la que más se parece al resto.
    names = list(medians)
    shifts = {}
    for a in names:
        for b in names:
            if a < b:
                s = estimate_shift(medians[a], medians[b])
                d = math.hypot(s["dx"], s["dy"]) if s else 99.0
                shifts[(a, b)] = shifts[(b, a)] = d
    def spread(name):
        others = [shifts[(name, o)] for o in names if o != name]
        return float(np.median(others)) if others else 0.0
    center = min(names, key=spread)
    consistent = [n for n in names if n == center or shifts[(center, n)] < 8.0]
    reference = np.median(np.stack([medians[n] for n in consistent]), axis=0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "front.jpg"), reference, [cv2.IMWRITE_JPEG_QUALITY, 92])

    report = {"reference_from": consistent, "sessions": {}}
    for name in names:
        s = estimate_shift(reference, medians[name])
        report["sessions"][name] = {"episodes": episodes[name], "shift": s}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    moved = [n for n in names if n not in consistent]
    print(f"\nReferencia hecha con {len(consistent)} de {len(names)} sesiones "
          f"({sum(episodes[n] for n in consistent)} episodios).", flush=True)
    if moved:
        print("Sesiones con la cámara en otra posición:", flush=True)
        for n in moved:
            s = report["sessions"][n]["shift"]
            print(f"  {n}: {describe_shift(s)['message']}", flush=True)
    else:
        print("La cámara estuvo en la misma posición en todas las sesiones.", flush=True)
    print(f"Guardada en {out_dir / 'front.jpg'}", flush=True)


# ----------------------------------------------------------------- API web
def build_router():
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import FileResponse

    router = APIRouter(prefix="/api/reference")

    @router.get("")
    async def info():
        report = None
        try:
            report = json.loads(REPORT_FILE.read_text())
        except (OSError, ValueError):
            pass
        return {"available": REFERENCE_IMAGE.is_file(),
                "sessions": len(report["reference_from"]) if report else 0}

    @router.get("/front.jpg")
    async def image():
        if not REFERENCE_IMAGE.is_file():
            raise HTTPException(404, "Falta la referencia: ejecuta ./comprobar_camara.sh")
        return FileResponse(REFERENCE_IMAGE, headers={"Cache-Control": "no-store"})

    @router.post("/compare")
    async def compare(request: Request):
        import asyncio
        import cv2

        if not REFERENCE_IMAGE.is_file():
            raise HTTPException(404, "Falta la referencia: ejecuta ./comprobar_camara.sh")
        body = await request.body()
        if not body or len(body) > 5_000_000:
            raise HTTPException(400, "Imagen no válida.")
        live = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if live is None:
            raise HTTPException(400, "Imagen no válida.")
        reference = cv2.imread(str(REFERENCE_IMAGE))
        shift = await asyncio.to_thread(estimate_shift, reference, live)
        return {"shift": shift, **describe_shift(shift)}

    return router


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    extract(args.datasets, args.out)
