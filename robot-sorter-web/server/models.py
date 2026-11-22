"""Biblioteca de modelos entrenados: subir .zip, listar, elegir y borrar.

Cada modelo subido se guarda en MODELS_DIR/<id>/pretrained (la carpeta
``pretrained_model`` de LeRobot: config.json, model.safetensors y los
procesadores). El modelo de SenseCraft montado en POLICY_PATH aparece como
modelo incluido y no se puede borrar.
"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from .calibration import check_origin

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "data" / "models"))
BUILTIN_PATH = Path(os.environ.get("POLICY_PATH", "/app/models/act"))
BUILTIN_ID = "sensecraft-act"
STATION_CAMERAS = {"front", "gripper"}
STATE_DIM = 6
MAX_ZIP_BYTES = 3 * 1024 ** 3
REQUIRED = ("config.json", "model.safetensors")


def describe(path):
    """Lee config.json y comprueba si el modelo encaja con la estación."""
    config = json.loads((path / "config.json").read_text())
    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    cameras = sorted(k.removeprefix("observation.images.") for k in inputs if k.startswith("observation.images."))
    state = (inputs.get("observation.state") or {}).get("shape")
    action = (outputs.get("action") or {}).get("shape")
    problems = []
    if not config.get("type"):
        problems.append("config.json no indica el tipo de política")
    unknown = sorted(set(cameras) - STATION_CAMERAS)
    if unknown:
        problems.append(f"usa cámaras que la estación no tiene: {', '.join(unknown)} (disponibles: front, gripper)")
    if not cameras:
        problems.append("no usa ninguna cámara")
    if state != [STATE_DIM]:
        problems.append(f"el estado debe tener {STATE_DIM} articulaciones (tiene {state})")
    if action != [STATE_DIM]:
        problems.append(f"la acción debe tener {STATE_DIM} articulaciones (tiene {action})")
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    return {
        "type": config.get("type"),
        "cameras": cameras,
        "compatible": not problems,
        "problems": problems,
        "size_mb": round(size / 2 ** 20, 1),
    }


class ModelLibrary:
    def __init__(self, directory=MODELS_DIR, builtin=BUILTIN_PATH):
        self.directory = Path(directory)
        self.builtin = Path(builtin)

    @property
    def active_file(self):
        return self.directory / "active.json"

    def path_of(self, model_id):
        if model_id == BUILTIN_ID:
            return self.builtin
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", model_id or ""):
            raise ValueError("Identificador de modelo no válido.")
        return self.directory / model_id / "pretrained"

    def list(self):
        models = []
        if (self.builtin / "config.json").is_file():
            models.append({"id": BUILTIN_ID, "name": "ACT de SenseCraft (incluido)", "builtin": True,
                           "uploaded_at": None, **describe(self.builtin)})
        if self.directory.is_dir():
            for folder in sorted(self.directory.iterdir()):
                pretrained = folder / "pretrained"
                if not (pretrained / "config.json").is_file():
                    continue
                try:
                    meta = json.loads((folder / "meta.json").read_text())
                except (OSError, ValueError):
                    meta = {}
                try:
                    info = describe(pretrained)
                except (OSError, ValueError) as error:
                    info = {"type": None, "cameras": [], "compatible": False, "problems": [str(error)], "size_mb": 0}
                models.append({"id": folder.name, "name": meta.get("name", folder.name), "builtin": False,
                               "uploaded_at": meta.get("uploaded_at"), **info})
        return models

    def active_id(self):
        models = {m["id"]: m for m in self.list()}
        try:
            chosen = json.loads(self.active_file.read_text()).get("id")
        except (OSError, ValueError):
            chosen = None
        if chosen in models:
            return chosen
        return BUILTIN_ID if BUILTIN_ID in models else next(iter(models), None)

    def active(self):
        model_id = self.active_id()
        if model_id is None:
            return None
        return next(m for m in self.list() if m["id"] == model_id)

    def select(self, model_id):
        model = next((m for m in self.list() if m["id"] == model_id), None)
        if model is None:
            raise ValueError("Ese modelo no existe.")
        if not model["compatible"]:
            raise ValueError("Ese modelo no es compatible con la estación: " + "; ".join(model["problems"]))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active_file.write_text(json.dumps({"id": model_id}))

    def delete(self, model_id):
        if model_id == BUILTIN_ID:
            raise ValueError("El modelo incluido no se puede borrar.")
        folder = self.path_of(model_id).parent
        if not folder.is_dir():
            raise ValueError("Ese modelo no existe.")
        shutil.rmtree(folder)

    def unique_id(self, name):
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "modelo"
        candidate, number = base, 2
        while (self.directory / candidate).exists() or candidate == BUILTIN_ID:
            candidate, number = f"{base}-{number}", number + 1
        return candidate

    def install_zip(self, zip_path, name):
        """Busca la carpeta del modelo dentro del zip y la instala. Devuelve el id."""
        try:
            archive = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile as error:
            raise ValueError("El archivo no es un .zip válido.") from error
        with archive:
            files = {}
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError("El .zip contiene rutas no permitidas.")
                files[member] = info
            # Carpetas que contienen config.json y model.safetensors; la menos profunda gana.
            candidates = sorted(
                (path.parent for path in files if path.name == "config.json"
                 and path.parent / "model.safetensors" in files),
                key=lambda folder: len(folder.parts),
            )
            if not candidates:
                raise ValueError("No se encuentra un modelo de LeRobot en el .zip (hace falta config.json y model.safetensors en la misma carpeta).")
            source = candidates[0]
            self.directory.mkdir(parents=True, exist_ok=True)
            model_id = self.unique_id(name)
            staging = Path(tempfile.mkdtemp(dir=self.directory, prefix=".install-"))
            try:
                target = staging / "pretrained"
                target.mkdir()
                # Solo los archivos de esa carpeta (no optimizadores ni otros checkpoints).
                for path, info in files.items():
                    if path.parent == source:
                        with archive.open(info) as reader, open(target / path.name, "wb") as writer:
                            shutil.copyfileobj(reader, writer, 1024 * 1024)
                info = describe(target)
                (staging / "meta.json").write_text(json.dumps({"name": name, "uploaded_at": time.time()}))
                staging.rename(self.directory / model_id)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return model_id, info


library = ModelLibrary()
router = APIRouter(prefix="/api/models")


class SelectRequest(BaseModel):
    id: str


def busy():
    from .policy import policy
    return policy.running


@router.get("")
async def list_models():
    return {"models": library.list(), "active": library.active_id()}


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), name: str = Form("")):
    check_origin(request)
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Sube un archivo .zip.")
    display = (name.strip() or Path(file.filename).stem)[:80]
    library.directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=library.directory, prefix=".upload-", suffix=".zip")
    try:
        written = 0
        with os.fdopen(handle, "wb") as output:
            while chunk := await file.read(4 * 1024 * 1024):
                written += len(chunk)
                if written > MAX_ZIP_BYTES:
                    raise HTTPException(413, "El .zip supera el límite de 3 GB.")
                output.write(chunk)
        try:
            model_id, info = await asyncio.to_thread(library.install_zip, temporary, display)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"id": model_id, "name": display, **info, "models": library.list(), "active": library.active_id()}


@router.post("/select")
async def select(payload: SelectRequest, request: Request):
    check_origin(request)
    if busy():
        raise HTTPException(409, "Para el modelo en ejecución antes de cambiarlo.")
    try:
        library.select(payload.id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"models": library.list(), "active": library.active_id()}


@router.delete("/{model_id}")
async def delete(model_id: str, request: Request):
    check_origin(request)
    if busy() and model_id == library.active_id():
        raise HTTPException(409, "No se puede borrar el modelo que se está ejecutando.")
    try:
        library.delete(model_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"models": library.list(), "active": library.active_id()}
