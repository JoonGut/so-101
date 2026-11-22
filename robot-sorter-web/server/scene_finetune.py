"""Detector de la mesa con la red entera reentrenada (más preciso que la capa lineal).

Una ResNet18 de ImageNet se reentrena de punta a punta con dos salidas:
- instrucción (11 clases, con «Nada» = mesa recogida);
- objetos fuera (papel, plástico, lata), cada uno independiente.
La probabilidad final de cada instrucción es el producto de ambas opiniones (como en
``scene.combine``). Se entrena con variaciones de encuadre, escala, giro y luz para
tolerar que la cámara de la estación no esté exactamente donde se grabó.

    python -m server.scene_finetune --dataset /ds --out /app/data/calibration/scene
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .scene import NOTHING, OBJECTS, OUT_OBJECTS, SCENE_DIR, combine, read_frames

WEIGHTS = "finetuned.pt"
META = "finetuned.json"
INPUT_W, INPUT_H = 320, 240
MIN_EPISODE_FRAMES = 150
START_OFFSETS = (0.05, 0.2, 0.35, 0.5, 0.7, 0.9)   # mesa con su instrucción (antes de que el brazo cambie nada)
END_OFFSETS = (0.1, 0.25, 0.4, 0.6)                # mesa recogida: «Nada»
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(classes):
    import torch
    import torchvision

    class SceneNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            backbone.fc = torch.nn.Identity()
            self.backbone = backbone
            self.dropout = torch.nn.Dropout(0.3)
            self.instruction = torch.nn.Linear(512, classes)
            self.objects = torch.nn.Linear(512, len(OBJECTS))

        def forward(self, x):
            feature = self.dropout(self.backbone(x))
            return self.instruction(feature), self.objects(feature)

    return SceneNet()


class FinetunedScene:
    """Misma interfaz que ``scene.SceneClassifier``: analyze(imagen) -> (ranking, P(hay objetos))."""

    def __init__(self, directory=SCENE_DIR, device="cpu"):
        import torch

        meta = json.loads((directory / META).read_text())
        self.labels = meta["labels"]
        self.durations = meta.get("durations", {})
        self.torch = torch
        self.device = device
        self.model = build_model(len(self.labels))
        self.model.load_state_dict(torch.load(directory / WEIGHTS, map_location=device))
        self.model.eval().to(device)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.last_out = None

    @staticmethod
    def available(directory=SCENE_DIR):
        return (directory / WEIGHTS).is_file() and (directory / META).is_file()

    def analyze(self, image_rgb):
        import cv2

        torch = self.torch
        image = cv2.resize(image_rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        with torch.inference_mode():
            logits, out_logits = self.model((x - self.mean) / self.std)
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        out = torch.sigmoid(out_logits)[0].cpu().numpy()
        self.last_out = dict(zip(OBJECTS, out.round(3).tolist()))
        combined = combine(out[None], probs[None], self.labels)[0]
        ranking = [(self.labels[i], float(combined[i])) for i in np.argsort(-combined)]
        return ranking, 1.0 - float(combined[self.labels.index(NOTHING)])

    def predict(self, image_rgb):
        return self.analyze(image_rgb)[0]


# ----------------------------------------------------------------- datos
def collect(dataset):
    import cv2
    import pandas as pd

    episodes = pd.concat([pd.read_parquet(f) for f in sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))])
    episodes = episodes[episodes["length"] >= MIN_EPISODE_FRAMES]
    key = "videos/observation.images.front"
    by_video = {}
    for _, ep in episodes.iterrows():
        task = ep["tasks"][0]
        video = dataset / "videos" / "observation.images.front" / f"chunk-{int(ep[f'{key}/chunk_index']):03d}" / f"file-{int(ep[f'{key}/file_index']):03d}.mp4"
        start, end = float(ep[f"{key}/from_timestamp"]), float(ep[f"{key}/to_timestamp"])
        for offset in START_OFFSETS:
            by_video.setdefault(video, []).append((start + offset, int(ep["episode_index"]), task))
        for offset in END_OFFSETS:
            by_video.setdefault(video, []).append((end - offset, int(ep["episode_index"]), NOTHING))
    images, episode_ids, tasks = [], [], []
    for index, (video, wanted) in enumerate(sorted(by_video.items())):
        frames = read_frames(video, [ts for ts, _, _ in wanted])
        for ts, episode, label in wanted:
            if ts in frames:
                images.append(cv2.resize(frames[ts], (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA))
                episode_ids.append(episode)
                tasks.append(label)
        print(f"  vídeo {index + 1}/{len(by_video)}: {len(images)} fotogramas", flush=True)
    durations = episodes.groupby(episodes["tasks"].apply(lambda t: t[0]))["length"].max()
    return np.stack(images), np.array(episode_ids), tasks, {t: round(float(n) / 30.0, 1) for t, n in durations.items()}


def augment_batch(x, torch):
    """Variaciones en GPU: encuadre, escala, giro, luz, color y algo de desenfoque."""
    import torch.nn.functional as F

    n = x.shape[0]
    angle = (torch.rand(n, device=x.device) - 0.5) * 2 * np.radians(5)
    scale = 1 + (torch.rand(n, device=x.device) - 0.5) * 2 * 0.15
    tx = (torch.rand(n, device=x.device) - 0.5) * 2 * 0.12
    ty = (torch.rand(n, device=x.device) - 0.5) * 2 * 0.12
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    brightness = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.5
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.5
    color = 1 + (torch.rand(n, 3, 1, 1, device=x.device) - 0.5) * 0.16
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness * color
    blur = torch.rand(n, device=x.device) < 0.3
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, 1, 1)
    return x.clamp(0, 1)


def run_training(images, labels_idx, out_targets, n_classes, device, epochs, log=True, eval_set=None):
    import torch

    torch.manual_seed(0)
    model = build_model(n_classes).to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    counts = np.bincount(labels_idx, minlength=n_classes).clip(min=1)
    class_weight = torch.tensor(counts.sum() / counts / n_classes, dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": list(model.instruction.parameters()) + list(model.objects.parameters()), "lr": 1e-3},
    ], weight_decay=1e-4)
    batch = 48
    steps = epochs * ((len(images) + batch - 1) // batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    y = torch.from_numpy(labels_idx).long()
    o = torch.from_numpy(out_targets).float()
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(epoch).permutation(len(images))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.from_numpy(images[idx]).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = (augment_batch(x, torch) - mean) / std
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits, out_logits = model(x)
                loss = ce(logits.float(), y[idx].to(device)) + bce(out_logits.float(), o[idx].to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss) * len(idx)
        message = f"  época {epoch + 1}/{epochs}: pérdida {total / len(images):.3f}"
        if eval_set is not None and (epoch + 1) % 3 == 0 or (eval_set is not None and epoch + 1 == epochs):
            message += f" · acierto validación {evaluate(model, *eval_set, device)[0]:.1%}"
        if log:
            print(message, flush=True)
    return model


def predict_all(model, images, labels, device):
    import torch

    model.eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    probs, outs = [], []
    with torch.inference_mode():
        for i in range(0, len(images), 64):
            x = torch.from_numpy(images[i:i + 64]).to(device).permute(0, 3, 1, 2).float() / 255.0
            logits, out_logits = model((x - mean) / std)
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            outs.append(torch.sigmoid(out_logits.float()).cpu().numpy())
    return combine(np.concatenate(outs), np.concatenate(probs), labels)


def evaluate(model, images, targets, labels, device):
    combined = predict_all(model, images, labels, device)
    predicted = combined.argmax(1)
    return float((predicted == targets).mean()), predicted


def train(dataset, out, device, epochs):
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Leyendo fotogramas del inicio y del final de cada episodio…", flush=True)
    images, episode_ids, tasks, durations = collect(dataset)
    labels = sorted(set(tasks), key=lambda l: (l == NOTHING, l))
    index = {label: i for i, label in enumerate(labels)}
    targets = np.array([index[t] for t in tasks])
    out_targets = np.array([[obj in OUT_OBJECTS[t] for obj in OBJECTS] for t in tasks], np.float32)

    episodes = sorted(set(episode_ids.tolist()))
    random.Random(0).shuffle(episodes)
    validation = set(episodes[: len(episodes) // 5])
    val = np.array([e in validation for e in episode_ids])
    print(f"{len(images)} fotogramas · {len(episodes)} episodios ({int(val.sum())} de validación)", flush=True)

    print("Reentrenando la red (validación con episodios no vistos)…", flush=True)
    model = run_training(images[~val], targets[~val], out_targets[~val], len(labels), device, epochs,
                         eval_set=(images[val], targets[val], labels))
    accuracy, predicted = evaluate(model, images[val], targets[val], labels, device)
    y_val = targets[val]
    empty = y_val == index[NOTHING]
    print(f"\nAcierto en episodios no vistos: {accuracy:.1%} · mesa vacía -> quieto "
          f"{float((predicted[empty] == index[NOTHING]).mean()):.1%}", flush=True)
    for i, label in enumerate(labels):
        mask = y_val == i
        if mask.any():
            wrong = {labels[j]: int(((predicted == j) & mask).sum()) for j in range(len(labels)) if j != i and ((predicted == j) & mask).any()}
            print(f"  {label:34s} {float((predicted[mask] == i).mean()):6.1%}  {wrong if wrong else ''}", flush=True)

    print("\nModelo final con todos los episodios…", flush=True)
    import torch
    model = run_training(images, targets, out_targets, len(labels), device, epochs, log=False)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / WEIGHTS)
    (out / META).write_text(json.dumps({
        "labels": labels, "durations": durations, "accuracy": accuracy,
        "episodes": len(episodes), "epochs": epochs, "trained_at": time.time(),
    }, indent=1))
    print(f"Guardado en {out} ({time.time() - started:.0f} s).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=SCENE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.dataset, args.out, device, args.epochs)
