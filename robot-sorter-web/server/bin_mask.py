"""Tapar los cubos en la imagen de la cámara de arriba que ve el modelo ACT.

El modelo se grabó con los cubos vacíos: si ya hay un objeto dentro de un cubo, la
imagen le resulta nueva y no sabe agarrar el siguiente. Aquí, en la imagen que recibe
ACT (no en la que usa el detector de la mesa), los cubos que no son el destino de la
instrucción se sustituyen por una foto de los cubos vacíos. El cubo de destino se deja
en directo para que el modelo vea su pinza al soltar.

Módulo independiente: si no está configurado o ``enabled`` es false, no hace nada.
Para quitarlo: borrar este archivo y las líneas marcadas «bin_mask» en run_policy.py.

Archivos (en calibration/bin_mask/ del ordenador):
    empty.png    foto de los cubos vacíos (la hace «capture»)
    config.json  zonas de cada cubo y si está activado

Uso (dentro del contenedor):
    python -m server.bin_mask capture               # foto de los cubos vacíos
    python -m server.bin_mask zones   --out z.jpg   # dibuja las zonas sobre la foto
    python -m server.bin_mask preview --out p.jpg [--keep lata] [--image f.jpg]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

MASK_DIR = Path(os.environ.get("BIN_MASK_DIR", "/app/data/calibration/bin_mask"))
EMPTY_IMAGE = MASK_DIR / "empty.png"
CONFIG_FILE = MASK_DIR / "config.json"
CAMERA = "front"

# Zonas en fracciones del ancho/alto de la imagen (polígonos), medidas sobre la cámara
# de arriba tal y como está montada el 2026-11-22. «objects» relaciona cada cubo con
# los nombres de objeto del detector (server/scene.py, OUT_OBJECTS).
DEFAULT_CONFIG = {
    "enabled": False,
    "feather_px": 7,          # borde suavizado entre la foto y el directo
    "match_light": True,      # igualar brillo de la foto con el directo
    "bins": {
        "arriba": {"objects": [], "polygon": [
            [0.200, 0.035], [0.250, 0.005], [0.380, 0.005], [0.425, 0.045],
            [0.430, 0.255], [0.385, 0.295], [0.250, 0.295], [0.205, 0.255]]},
        "medio": {"objects": [], "polygon": [
            [0.178, 0.305], [0.230, 0.262], [0.375, 0.262], [0.420, 0.305],
            [0.422, 0.560], [0.375, 0.612], [0.230, 0.612], [0.182, 0.565]]},
        "abajo": {"objects": [], "polygon": [
            [0.145, 0.610], [0.200, 0.555], [0.355, 0.555], [0.410, 0.610],
            [0.408, 0.870], [0.355, 0.925], [0.215, 0.925], [0.160, 0.880]]},
    },
}


def load_config():
    try:
        return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)


def save_default_config():
    if not CONFIG_FILE.exists():
        MASK_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False))


class BinMask:
    def __init__(self, empty_rgb, config):
        self.empty = empty_rgb
        self.config = config
        self.bins = config["bins"]
        self._cache = {}

    def _polygon_mask(self, name, shape):
        import cv2

        h, w = shape
        points = np.array([[x * w, y * h] for x, y in self.bins[name]["polygon"]], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [points], 255)
        return mask

    def _masks(self, cover, shape):
        """Máscara (0..1) de lo que se tapa y anillo alrededor para igualar la luz."""
        import cv2

        key = (tuple(sorted(cover)), shape)
        if key not in self._cache:
            covered = np.zeros(shape, np.uint8)
            for name in cover:
                covered |= self._polygon_mask(name, shape)
            for name in set(self.bins) - set(cover):       # el cubo de destino siempre en directo
                covered &= ~self._polygon_mask(name, shape)
            feather = int(self.config.get("feather_px", 0))
            alpha = covered.astype(np.float32) / 255
            if feather > 0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), feather / 2)
            everything = np.zeros(shape, np.uint8)
            for name in self.bins:
                everything |= self._polygon_mask(name, shape)
            ring = cv2.dilate(everything, np.ones((31, 31), np.uint8)) & ~cv2.dilate(everything, np.ones((9, 9), np.uint8))
            self._cache[key] = (alpha[..., None], ring > 0)
        return self._cache[key]

    def cover_for(self, task_objects):
        """Cubos que se tapan: todos menos los de los objetos de la instrucción."""
        return [name for name, info in self.bins.items()
                if not set(info.get("objects", [])) & set(task_objects)]

    def apply(self, frame_rgb, task_objects):
        import cv2

        cover = self.cover_for(task_objects)
        if not cover:
            return frame_rgb
        shape = frame_rgb.shape[:2]
        empty = self.empty
        if empty.shape[:2] != shape:
            empty = cv2.resize(empty, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        alpha, ring = self._masks(cover, shape)
        empty = empty.astype(np.float32)
        if self.config.get("match_light", True) and ring.any():
            # La cámara ajusta la exposición sola: se escala la foto para que su
            # alrededor (papel y tapete junto a los cubos) brille como en directo.
            live_ring = frame_rgb[ring].astype(np.float32)
            empty_ring = empty[ring]
            gain = np.median(live_ring + 1, axis=0) / np.median(empty_ring + 1, axis=0)
            empty = empty * np.clip(gain, 0.8, 1.25)
        out = frame_rgb.astype(np.float32) * (1 - alpha) + empty * alpha
        return np.clip(out, 0, 255).astype(np.uint8)


def load_bin_mask(model_path=None):
    """BinMask listo, o None si no está activado o falta algo (y explica por qué)."""
    import cv2

    config = load_config()
    if not config.get("enabled"):
        return None
    allowed = config.get("models") or []
    if allowed and model_path and not any(name in str(model_path) for name in allowed):
        print(f"Tapar cubos: no se usa con este modelo (solo con {', '.join(allowed)}).", flush=True)
        return None
    if not any(info.get("objects") for info in config["bins"].values()):
        print("Tapar cubos: falta decir qué objeto va a cada cubo en config.json. No se usa.", flush=True)
        return None
    image = cv2.imread(str(EMPTY_IMAGE))
    if image is None:
        print("Tapar cubos: falta la foto de los cubos vacíos (python -m server.bin_mask capture). No se usa.", flush=True)
        return None
    print("Tapar cubos: activado (el modelo ve vacíos los cubos que no son el destino).", flush=True)
    return BinMask(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), config)


# ----------------------------------------------------------------- herramientas
def grab_frame(width=640, height=480):
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    path = os.environ.get("POLICY_CAMERA_FRONT", "/dev/cam_front")
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=Path(path), width=width, height=height, fps=30))
    camera.connect()
    try:
        for _ in range(15):          # que se asiente la exposición automática
            frame = camera.read()
        return frame
    finally:
        camera.disconnect()


def draw_zones(image_rgb, config, cover=()):
    import cv2

    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]
    for name, info in config["bins"].items():
        points = np.array([[x * w, y * h] for x, y in info["polygon"]], np.int32)
        color = (0, 0, 255) if name in cover else (0, 200, 0)
        cv2.polylines(out, [points], True, color, 2)
        label = f"{name}: {', '.join(info.get('objects') or ['?'])}".replace("á", "a")  # OpenCV no dibuja tildes
        cv2.putText(out, label, tuple(points[0] + [0, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def write(path, image):
    import cv2

    if not cv2.imwrite(str(path), image):
        raise SystemExit(f"No se ha podido guardar {path} (¿existe la carpeta?)")


def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "zones", "preview"])
    parser.add_argument("--out", help="imagen de salida (zones / preview)")
    parser.add_argument("--image", help="usar esta imagen en vez de la cámara (preview)")
    parser.add_argument("--keep", action="append", default=[], help="objeto de la instrucción (lata, papel, plástico)")
    args = parser.parse_args()
    save_default_config()
    config = load_config()

    if args.action == "capture":
        frame = grab_frame()
        write(EMPTY_IMAGE, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Foto de los cubos vacíos guardada en {EMPTY_IMAGE} ({frame.shape[1]}x{frame.shape[0]})")
        return

    empty = cv2.imread(str(EMPTY_IMAGE))
    if args.action == "zones":
        source = cv2.cvtColor(empty, cv2.COLOR_BGR2RGB) if empty is not None else grab_frame()
        write(args.out, draw_zones(source, config))
        print(f"Zonas dibujadas en {args.out}")
        return

    if empty is None:
        raise SystemExit("Falta la foto de los cubos vacíos: python -m server.bin_mask capture")
    if args.image:
        live = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    else:
        live = grab_frame()
    masker = BinMask(cv2.cvtColor(empty, cv2.COLOR_BGR2RGB), config)
    model_view = masker.apply(live, args.keep)
    cover = masker.cover_for(args.keep)
    side = np.hstack([draw_zones(live, config, cover), cv2.cvtColor(model_view, cv2.COLOR_RGB2BGR)])
    cv2.putText(side, "real", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(side, "lo que ve ACT", (live.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    write(args.out, side)
    print(f"Comparación guardada en {args.out} (tapados: {', '.join(cover) or 'ninguno'})")


if __name__ == "__main__":
    main()
