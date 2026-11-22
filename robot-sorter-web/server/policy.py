"""Ejecuta la política entrenada (ACT) en el brazo real desde la web.

El modelo necesita el USB del brazo y las dos cámaras en exclusiva: antes de
arrancar se deja el brazo en reposo y se libera el puente de control, y la web
suelta las cámaras.

- Parar: SIGINT. El bucle devuelve el brazo a la postura inicial y lo desconecta.
- Parada de emergencia: SIGKILL. Los motores conservan el último objetivo y el
  brazo se queda quieto donde está, con par.
"""

import asyncio
import codecs
import json
import os
import re
import signal
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .calibration import CALIBRATION_DIR, calibration, check_origin
from .hardware_api import hardware
from .robot_controller import robot
from .models import library
from .run_policy import PARAMS_FILE, PREVIEW_DIR, STATE_FILE

ROBOT_PORT = os.environ.get("POLICY_ROBOT_PORT", "/dev/ttyACM0")
CAMERAS = {
    "front": os.environ.get("POLICY_CAMERA_FRONT", "/dev/cam_front"),
    "gripper": os.environ.get("POLICY_CAMERA_GRIPPER", "/dev/cam_gripper"),
}
ROBOT_ID = os.environ.get("POLICY_ROBOT_ID", "so101")
MOTION_KEYS = ("speed_factor", "smoothness_factor", "max_acceleration", "pan_offset", "auto_correct")
SAVED_PARAMS = CALIBRATION_DIR / "station_params.json"
SCENE_INFO = CALIBRATION_DIR / "scene" / "classifier.json"


def scene_info():
    """¿Está entrenado el detector de la mesa? (para el modo automático)"""
    for path, kind in ((SCENE_INFO.with_name("finetuned.json"), "red reentrenada"), (SCENE_INFO, "capa lineal")):
        try:
            meta = json.loads(path.read_text())
            return {"available": True, "accuracy": meta.get("accuracy"), "labels": meta.get("labels"), "kind": kind}
        except (OSError, ValueError):
            continue
    return {"available": False}


def load_params():
    """Recupera los ajustes de la estación guardados (sobreviven a reinicios)."""
    try:
        robot.update_parameters({k: v for k, v in json.loads(SAVED_PARAMS.read_text()).items() if k in MOTION_KEYS})
    except (OSError, ValueError):
        pass


def save_params():
    try:
        SAVED_PARAMS.write_text(json.dumps({key: getattr(robot, key) for key in MOTION_KEYS}))
    except OSError:
        pass


def build_command(model_path, duration, task=None, auto=False):
    command = [
        sys.executable, "-u", "-m", "server.run_policy",
        f"--model={model_path}",
        f"--port={ROBOT_PORT}",
        f"--robot-id={ROBOT_ID}",
        f"--calibration-dir={CALIBRATION_DIR}",
        f"--duration={duration}",
        *([f"--task={task}"] if task else []),
        *(["--auto"] if auto else []),
        "--device=auto",
    ]
    command += [f"--camera={name}={path}" for name, path in CAMERAS.items()]
    return command


def push_params():
    """Escribe los ajustes de movimiento para que el bucle del modelo los aplique."""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    data = json.dumps({key: getattr(robot, key) for key in MOTION_KEYS}).encode()
    temporary = PARAMS_FILE.with_name(".params.json")
    temporary.write_bytes(data)
    os.replace(temporary, PARAMS_FILE)


class PolicySession:
    def __init__(self):
        self.process = None
        self.task = None
        self.lock = asyncio.Lock()
        self.state = "idle"  # idle, preparing, running, finished, stopped, error
        self.output = ""
        self.stop_requested = False

    @property
    def running(self):
        return self.state in ("preparing", "running")

    def status(self):
        return {
            "state": self.state,
            "output": self.output,
            "model": library.active(),
            "calibrated": (CALIBRATION_DIR / f"{ROBOT_ID}.json").is_file(),
            "previews": sorted(p.stem for p in PREVIEW_DIR.glob("*.jpg")) if self.running else [],
            "auto": self.auto_info() if self.running else None,
            "detector": scene_info(),
        }

    @staticmethod
    def auto_info():
        try:
            return json.loads(STATE_FILE.read_text()).get("auto")
        except (OSError, ValueError):
            return None

    async def start(self, duration, task=None, auto=False):
        async with self.lock:
            if self.running:
                raise ValueError("El modelo ya se está ejecutando.")
            if calibration.running:
                raise ValueError("Termina o cancela la calibración antes de ejecutar el modelo.")
            model = library.active()
            if model is None:
                raise ValueError("No hay ningún modelo. Sube uno en el panel de modelos.")
            if not model["compatible"]:
                raise ValueError("El modelo elegido no es compatible: " + "; ".join(model["problems"]))
            model_path = library.path_of(model["id"])
            if not (CALIBRATION_DIR / f"{ROBOT_ID}.json").is_file():
                raise ValueError(f"Falta la calibración «{ROBOT_ID}».")
            self.state = "preparing"
            self.stop_requested = False
            self.output = "Preparando el brazo…\n"
            try:
                await self.release_arm()
            except Exception as error:
                self.state = "error"
                self.output += f"No se ha podido preparar el brazo: {error}\n"
                raise ValueError(str(error)) from error
            robot.mode = "manual"
            robot.set_estop(True)  # el simulador no compite con el modelo
            self.output += f"Modelo: {model['name']}. Cargando y abriendo cámaras (unos segundos)…\n"
            push_params()
            STATE_FILE.unlink(missing_ok=True)
            hardware.robot_id = ROBOT_ID
            hardware.load_mapping()
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *build_command(model_path, duration, task, auto), stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as error:
                self.state = "error"
                self.output += f"No se ha podido lanzar LeRobot: {error}\n"
                raise ValueError(str(error)) from error
            self.state = "running"
            self.task = asyncio.create_task(self.read_output())
            asyncio.create_task(self.follow_arm())

    async def release_arm(self):
        """Deja el brazo en reposo y libera el USB que usa el puente de control."""
        if hardware.state == "active":
            # Se parte de la postura real (aunque esté en parada) y se va a reposo suavemente.
            if hardware.real:
                joints, gripper = hardware.real_to_sim(hardware.real)
                robot.current_joints = joints
                robot.gripper_percent = gripper
            hardware.holding = False
            robot.set_estop(False)
            self.output += "Llevando el brazo a reposo…\n"
            await robot.return_home()
        if hardware.state != "disconnected":
            await hardware.disconnect()
            robot.smooth_home = False

    async def follow_arm(self):
        """El gemelo 3D copia la postura real que publica el bucle del modelo."""
        while self.state == "running":
            try:
                real = json.loads(STATE_FILE.read_text())
                found = real.get("found_pan_offset")
                if found is not None and abs(found - robot.pan_offset) > 0.05:
                    # El bucle ha encontrado una corrección que agarra: pasa a ser la base.
                    robot.update_parameters({"pan_offset": found})
                    save_params()
                    push_params()
                joints, gripper = hardware.real_to_sim(real)
                robot.current_joints = joints
                robot.gripper_percent = gripper
            except (OSError, ValueError, KeyError):
                pass
            await asyncio.sleep(0.1)

    async def read_output(self):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while chunk := await self.process.stdout.read(4096):
                text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", decoder.decode(chunk)).replace("\r", "")
                self.output = (self.output + text)[-24000:]
            code = await self.process.wait()
            if self.stop_requested:
                self.state = "stopped"
            elif code == 0:
                self.state = "finished"
                self.output += "\nEjecución terminada; el brazo ha vuelto a su postura inicial.\n"
            else:
                self.state = "error"
                self.output += f"\nLeRobot ha terminado con error (código {code}). Revisa el registro.\n"
        except Exception as error:
            self.state = "error"
            self.output += f"\nError al leer LeRobot: {error}\n"

    async def stop(self, emergency=False):
        async with self.lock:
            if self.state != "running" or self.process is None:
                return
            self.stop_requested = True
            try:
                if emergency:
                    self.output += "\nPARADA DE EMERGENCIA: proceso detenido; el brazo se queda quieto con par.\n"
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.output += "\nParando: el brazo vuelve a su postura inicial…\n"
                    os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 20)
            except asyncio.TimeoutError:
                self.output += "\nLeRobot no ha respondido; se fuerza la parada.\n"
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.task


policy = PolicySession()
router = APIRouter(prefix="/api/policy")


class PolicyStart(BaseModel):
    duration: int = Field(default=60, ge=5, le=3600)
    auto: bool = False
    task: str | None = Field(default=None, max_length=120)


@router.get("")
async def status():
    return policy.status()


@router.post("/start")
async def start(payload: PolicyStart, request: Request):
    check_origin(request)
    try:
        await policy.start(payload.duration, (payload.task or "").strip() or None, payload.auto)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return policy.status()


@router.post("/stop")
async def stop(request: Request):
    check_origin(request)
    await policy.stop()
    return policy.status()


@router.get("/preview/{name}.jpg")
async def preview(name: str):
    if name not in ("front", "gripper", "front_act"):  # front_act: bin_mask (cubos tapados)
        raise HTTPException(404)
    path = PREVIEW_DIR / f"{name}.jpg"
    if not path.is_file():
        return Response(status_code=204)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
