"""Decide la instrucción mirando la mesa con la cámara de arriba.

Se entrena con el dataset de instrucciones (LeRobot v3):
- el inicio de cada episodio muestra la mesa con su instrucción (p. ej. «Papel_out_Lata_out»);
- el final muestra la mesa ya recogida: clase «Nada» (el brazo se queda quieto).

Modelo: ResNet18 de ImageNet congelada + dos capas lineales entrenadas:
- instrucción: rasgos de la imagen entera y de la zona de la mesa (1024);
- un detector por objeto («¿está fuera el papel? ¿el plástico? ¿la lata?»), solo con la
  zona de la mesa. La probabilidad final de cada instrucción es el producto de las dos
  opiniones: la capa de instrucción (mira la escena entera, desempata los casos «_in») y
  la de que estén fuera justo sus objetos (generaliza mejor a colocaciones nuevas);
- presencia («¿hay objetos en la mesa?»): solo la zona de la mesa, sin los cubos, para
  que los cubos llenos o vacíos no le confundan. Es la compuerta de seguridad: si dice
  que la mesa está vacía, el brazo no se mueve aunque la otra capa dude.
Se entrena con desplazamientos y cambios de luz para tolerar que la cámara se haya
movido un poco.

    python -m server.scene --dataset /ds --out /app/data/calibration/scene
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

SCENE_DIR = Path("/app/data/calibration/scene")
NOTHING = "Nada"
INPUT_W, INPUT_H = 320, 240
TABLE_X0 = 0.38              # a la izquierda de esta fracción del ancho están los cubos
TABLE_W = int(INPUT_W * (1 - TABLE_X0))
MIN_EPISODE_FRAMES = 150     # episodios más cortos (5 s) se consideran fallidos
START_OFFSETS = (0.1, 0.35, 0.6)
END_OFFSETS = (0.1, 0.35, 0.6)
AUGMENTED_COPIES = 3

# Qué objetos están fuera (en la mesa) en cada instrucción.
OUT_OBJECTS = {
    "Papel": ["papel"],
    "Plastico": ["plástico"],
    "Lata": ["lata"],
    "Papel_in_Plastico_out_Lata_out": ["plástico", "lata"],
    "Papel_out_Lata_out": ["papel", "lata"],
    "Papel_out_Plastico_in_Lata_out": ["papel", "lata"],
    "Papel_out_Plastico_out": ["papel", "plástico"],
    "Papel_out_Plastico_out_Lata_in": ["papel", "plástico"],
    "Papel_out_Plastico_out_Lata_out": ["papel", "plástico", "lata"],
    "Plastico_out_Lata_out": ["plástico", "lata"],
    NOTHING: [],
}
OBJECTS = ["papel", "plástico", "lata"]


# ----------------------------------------------------------------- rasgos
class FeatureExtractor:
    def __init__(self, device="cpu"):
        import torch
        import torchvision

        self.torch = torch
        self.device = device
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(self, batch):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        with torch.inference_mode():
            return self.model((x - self.mean) / self.std).float().cpu().numpy()

    def __call__(self, images_rgb):
        """Imágenes RGB uint8 (cualquier tamaño) -> rasgos (N, 1024): imagen entera + zona de la mesa."""
        import cv2

        full = np.stack([cv2.resize(i, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA) for i in images_rgb])
        table = np.stack([cv2.resize(i[:, int(i.shape[1] * TABLE_X0):], (TABLE_W, INPUT_H), interpolation=cv2.INTER_AREA)
                          for i in images_rgb])
        return np.concatenate([self.embed(full), self.embed(table)], axis=1)


def augment(image, rng):
    """Desplaza, escala, gira un poco y cambia la luz (cámara algo movida, otra luz)."""
    import cv2

    h, w = image.shape[:2]
    angle = rng.uniform(-2.5, 2.5)
    scale = rng.uniform(0.95, 1.05)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-30, 30)
    matrix[1, 2] += rng.uniform(-20, 20)
    out = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = out * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    out *= np.array([rng.uniform(0.92, 1.08) for _ in range(3)], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- clasificador
class SceneClassifier:
    """Clasifica un fotograma de la cámara de arriba en una instrucción (o «Nada»)."""

    def __init__(self, directory=SCENE_DIR, device="cpu"):
        data = np.load(directory / "classifier.npz")
        meta = json.loads((directory / "classifier.json").read_text())
        self.labels = meta["labels"]
        self.durations = meta.get("durations", {})
        self.mean, self.std = data["mean"], data["std"]
        self.weight, self.bias = data["weight"], data["bias"]
        self.presence = (data["p_mean"], data["p_std"], data["p_weight"], data["p_bias"]) if "p_weight" in data else None
        self.per_object = (data["o_weight"], data["o_bias"]) if "o_weight" in data else None
        self.features = FeatureExtractor(device)

    @staticmethod
    def available(directory=SCENE_DIR):
        return (directory / "classifier.npz").is_file() and (directory / "classifier.json").is_file()

    def analyze(self, image_rgb):
        """-> ([(instrucción, probabilidad)] de más a menos probable, probabilidad de que haya objetos)."""
        raw = self.features([image_rgb])[0]
        feature = (raw[: self.mean.shape[0]] - self.mean) / self.std
        probs = softmax(self.weight @ feature + self.bias)
        order = np.argsort(-probs)
        ranking = [(self.labels[i], float(probs[i])) for i in order]
        if self.presence is None:
            objects = 1.0 - float(probs[self.labels.index(NOTHING)])
        else:
            mean, std, weight, bias = self.presence
            table = (raw[512:] - mean) / std
            objects = float(softmax(weight @ table + bias)[1])
            if self.per_object is not None:
                weight, bias = self.per_object
                out = sigmoid(weight @ table + bias)
                self.last_out = dict(zip(OBJECTS, out.round(3).tolist()))
                combined = combine(out[None], probs[None], self.labels)[0]
                ranking = [(self.labels[i], float(combined[i])) for i in np.argsort(-combined)]
        return ranking, objects

    def predict(self, image_rgb):
        return self.analyze(image_rgb)[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def combine(out, probs, labels):
    """Producto de opiniones: P(instrucción) x P(estén fuera justo sus objetos), normalizado.

    out: (N, 3) probabilidad de que cada objeto de OBJECTS esté fuera; probs: (N, C).
    """
    wanted = np.array([[obj in OUT_OBJECTS[label] for obj in OBJECTS] for label in labels])  # (C, 3)
    p_set = np.prod(np.where(wanted[None], out[:, None, :], 1 - out[:, None, :]), axis=2)    # (N, C)
    scores = p_set * probs
    return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)


def softmax(logits):
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def train_linear(x, y, classes, epochs=300, weight_decay=1e-3, seed=0):
    import torch

    torch.manual_seed(seed)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    layer = torch.nn.Linear(x.shape[1], classes)
    counts = torch.bincount(y, minlength=classes).float().clamp(min=1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=(counts.sum() / counts) / classes, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(layer(x), y)
        loss.backward()
        optimizer.step()
    return layer.weight.detach().numpy(), layer.bias.detach().numpy()


def softmax_rows(logits):
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probs / probs.sum(axis=1, keepdims=True)


def train_objects(x, y, epochs=300, weight_decay=1e-3, seed=0):
    """Una salida independiente (sigmoide) por objeto: ¿está fuera?"""
    import torch

    torch.manual_seed(seed)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    layer = torch.nn.Linear(x.shape[1], y.shape[1])
    positive = y.mean(0).clamp(0.05, 0.95)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - positive) / positive)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(layer(x), y)
        loss.backward()
        optimizer.step()
    return layer.weight.detach().numpy(), layer.bias.detach().numpy()


# ----------------------------------------------------------------- extracción
def read_frames(video, timestamps):
    """Fotogramas RGB en los instantes pedidos (segundos) de un vídeo."""
    import av

    frames = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for ts in sorted(timestamps):
            container.seek(int(max(0.0, ts - 0.2) / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is not None and frame.time >= ts - 1 / 60:
                    frames[ts] = frame.to_ndarray(format="rgb24")
                    break
    return frames


def collect(dataset):
    import pandas as pd

    episodes = pd.concat([pd.read_parquet(f) for f in sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))])
    key = "videos/observation.images.front"
    samples = []  # (episodio, etiqueta, imagen)
    by_video = {}
    for _, ep in episodes.iterrows():
        if ep["length"] < MIN_EPISODE_FRAMES:
            continue
        task = ep["tasks"][0]
        video = dataset / "videos" / "observation.images.front" / f"chunk-{int(ep[f'{key}/chunk_index']):03d}" / f"file-{int(ep[f'{key}/file_index']):03d}.mp4"
        start, end = float(ep[f"{key}/from_timestamp"]), float(ep[f"{key}/to_timestamp"])
        for offset in START_OFFSETS:
            by_video.setdefault(video, []).append((start + offset, int(ep["episode_index"]), task))
        for offset in END_OFFSETS:
            by_video.setdefault(video, []).append((end - offset, int(ep["episode_index"]), NOTHING))
    for index, (video, wanted) in enumerate(sorted(by_video.items())):
        frames = read_frames(video, [ts for ts, _, _ in wanted])
        for ts, episode, label in wanted:
            if ts in frames:
                samples.append((episode, label, frames[ts]))
        print(f"  vídeo {index + 1}/{len(by_video)}: {len(samples)} fotogramas", flush=True)
    durations = episodes[episodes["length"] >= MIN_EPISODE_FRAMES].groupby(
        episodes["tasks"].apply(lambda t: t[0]))["length"].max()
    return samples, {task: round(float(n) / 30.0, 1) for task, n in durations.items()}


def train(dataset, out, device):
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Leyendo fotogramas del inicio y del final de cada episodio…", flush=True)
    samples, durations = collect(dataset)
    labels = sorted({label for _, label, _ in samples}, key=lambda l: (l == NOTHING, l))
    index = {label: i for i, label in enumerate(labels)}

    # Separación por episodios para medir de verdad (sin mezclar fotogramas del mismo episodio).
    episodes = sorted({e for e, _, _ in samples})
    random.Random(0).shuffle(episodes)
    validation = set(episodes[: len(episodes) // 5])

    extractor = FeatureExtractor(device)
    rng = random.Random(1)
    def features(rows, copies):
        images, targets = [], []
        for _, label, image in rows:
            images.append(image)
            targets.append(index[label])
            for _ in range(copies):
                images.append(augment(image, rng))
                targets.append(index[label])
        feats = np.concatenate([extractor(images[i:i + 64]) for i in range(0, len(images), 64)])
        return feats, np.array(targets)

    print("Calculando rasgos (con variaciones de posición y luz)…", flush=True)
    train_rows = [s for s in samples if s[0] not in validation]
    val_rows = [s for s in samples if s[0] in validation]
    x_train, y_train = features(train_rows, AUGMENTED_COPIES)
    # Validación con variaciones: simula la cámara algo movida respecto a la grabación.
    x_val, y_val = features(val_rows, 1)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    weight, bias = train_linear((x_train - mean) / std, y_train, len(labels))
    predicted = np.argmax(((x_val - mean) / std) @ weight.T + bias, axis=1)
    accuracy = float((predicted == y_val).mean())
    print(f"\nAcierto en episodios no vistos: {accuracy:.1%} ({len(y_val)} fotogramas)", flush=True)

    # Compuerta de presencia: solo la zona de la mesa, ¿hay objetos o no?
    nothing = index[NOTHING]
    p_train, p_val = (y_train != nothing).astype(int), (y_val != nothing).astype(int)
    t_train, t_val = x_train[:, 512:], x_val[:, 512:]
    p_mean, p_std = t_train.mean(0), t_train.std(0) + 1e-6
    p_weight, p_bias = train_linear((t_train - p_mean) / p_std, p_train, 2)
    p_pred = np.argmax(((t_val - p_mean) / p_std) @ p_weight.T + p_bias, axis=1)
    presence_accuracy = float((p_pred == p_val).mean())
    print(f"Detector de presencia (¿hay objetos?): {presence_accuracy:.1%} · "
          f"mesa vacía bien {float((p_pred[p_val == 0] == 0).mean()):.1%} · "
          f"con objetos bien {float((p_pred[p_val == 1] == 1).mean()):.1%}", flush=True)
    o_train = np.array([[obj in OUT_OBJECTS[labels[t]] for obj in OBJECTS] for t in y_train], int)
    o_val = np.array([[obj in OUT_OBJECTS[labels[t]] for obj in OBJECTS] for t in y_val], int)
    o_weight, o_bias = train_objects((t_train - p_mean) / p_std, o_train)
    o_pred = (((t_val - p_mean) / p_std) @ o_weight.T + o_bias) > 0
    for i, obj in enumerate(OBJECTS):
        print(f"  ¿{obj} fuera?: {float((o_pred[:, i] == o_val[:, i]).mean()):.1%}", flush=True)
    # Instrucción combinada: capa de instrucción x objetos fuera.
    val_probs = softmax_rows(((x_val - mean) / std) @ weight.T + bias)
    val_out = sigmoid(((t_val - p_mean) / p_std) @ o_weight.T + o_bias)
    predicted = np.argmax(combine(val_out, val_probs, labels), axis=1)
    accuracy = float((predicted == y_val).mean())
    print(f"Instrucción combinada (instrucción x objetos): {accuracy:.1%}", flush=True)
    combined = np.where(p_pred == 0, nothing, predicted)
    empty = y_val == nothing
    print(f"Con la compuerta: acierto {float((combined == y_val).mean()):.1%} · "
          f"mesa vacía -> quieto {float((combined[empty] == nothing).mean()):.1%}", flush=True)
    for i, label in enumerate(labels):
        mask = y_val == i
        if mask.any():
            wrong = {labels[j]: int(((predicted == j) & mask).sum()) for j in range(len(labels)) if j != i and ((predicted == j) & mask).any()}
            print(f"  {label:34s} {float((predicted[mask] == i).mean()):6.1%}  {wrong if wrong else ''}", flush=True)

    # Modelo final con todos los episodios.
    x_all = np.concatenate([x_train, x_val])
    y_all = np.concatenate([y_train, y_val])
    mean, std = x_all.mean(0), x_all.std(0) + 1e-6
    weight, bias = train_linear((x_all - mean) / std, y_all, len(labels))
    t_all, p_all = x_all[:, 512:], (y_all != nothing).astype(int)
    p_mean, p_std = t_all.mean(0), t_all.std(0) + 1e-6
    p_weight, p_bias = train_linear((t_all - p_mean) / p_std, p_all, 2)
    o_all = np.array([[obj in OUT_OBJECTS[labels[t]] for obj in OBJECTS] for t in y_all], int)
    o_weight, o_bias = train_objects((t_all - p_mean) / p_std, o_all)
    np.savez(out / "classifier.npz", mean=mean, std=std, weight=weight, bias=bias,
             p_mean=p_mean, p_std=p_std, p_weight=p_weight, p_bias=p_bias, o_weight=o_weight, o_bias=o_bias)
    (out / "classifier.json").write_text(json.dumps({
        "labels": labels, "durations": durations, "accuracy": accuracy, "presence_accuracy": presence_accuracy,
        "episodes": len(episodes), "trained_at": time.time(),
    }, indent=1))
    print(f"\nGuardado en {out} ({time.time() - started:.0f} s).", flush=True)



def load_scene(directory=SCENE_DIR, device="cpu"):
    """Detector de la mesa: la red reentrenada si existe; si no, la capa lineal."""
    from .scene_finetune import FinetunedScene

    if FinetunedScene.available(directory):
        return FinetunedScene(directory, device)
    return SceneClassifier(directory, device)


def scene_available(directory=SCENE_DIR):
    from .scene_finetune import FinetunedScene

    return SceneClassifier.available(directory) or FinetunedScene.available(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=SCENE_DIR)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.dataset, args.out, device)
