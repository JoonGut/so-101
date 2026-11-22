"""Puente entre la web y la calibración oficial de LeRobot; no replica su algoritmo."""

import asyncio
import codecs
import importlib.util
import os
import re
import signal
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .robot_controller import robot

CALIBRATION_DIR = Path(os.environ.get(
    "CALIBRATION_DIR", Path(__file__).resolve().parent.parent / "data" / "calibration"
))


def serial_ports():
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        return []
    return [
        {"port": port.device, "name": port.description}
        for port in sorted(comports(), key=lambda item: item.device)
        if port.vid is not None or "usb" in port.device.lower() or "/ttyACM" in port.device
    ]


class CalibrationSession:
    def __init__(self, directory=CALIBRATION_DIR):
        self.directory = Path(directory)
        self.process = None
        self.task = None
        self.lock = asyncio.Lock()
        self.state = "idle"
        self.output = ""
        self.robot_id = "so101"
        self.cancelled = False

    @property
    def running(self):
        return self.state == "running"

    def status(self):
        return {
            "state": self.state,
            "output": self.output,
            "robot_id": self.robot_id,
            "saved": (self.directory / f"{self.robot_id}.json").is_file(),
            "available": importlib.util.find_spec("lerobot") is not None and os.name != "nt",
        }

    async def start(self, port, robot_id):
        async with self.lock:
            if self.running:
                raise ValueError("Ya hay una calibración en curso.")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", robot_id):
                raise ValueError("El nombre admite letras, números, guiones y guiones bajos.")
            if not self.status()["available"]:
                raise ValueError("Usa Linux con LeRobot instalado, o el contenedor Docker con acceso al USB.")
            from .hardware_api import hardware
            from .policy import policy
            if hardware.state != "disconnected" and hardware.state != "error":
                raise ValueError("Desconecta el robot del panel de control real antes de calibrar.")
            if policy.running:
                raise ValueError("Para el modelo entrenado antes de calibrar.")
            if port not in {device["port"] for device in serial_ports()}:
                raise ValueError("El puerto USB no está disponible. Conecta el robot y actualiza los puertos.")
            self.directory.mkdir(parents=True, exist_ok=True)
            # No se ejecuta ninguna cadena de shell; todos los argumentos son independientes.
            command = [
                sys.executable, "-u", "-m", "server.calibrate_robot",
                "--port", port, "--id", robot_id, "--directory", str(self.directory),
            ]
            robot.set_estop(True)
            robot.mode = "manual"
            self.robot_id = robot_id
            self.cancelled = False
            self.output = "Iniciando la calibración oficial de LeRobot…\n"
            self.state = "running"
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
            except OSError:
                self.state = "error"
                self.output += "No se ha podido abrir el proceso de LeRobot.\n"
                raise
            self.task = asyncio.create_task(self.read_output())

    async def read_output(self):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while chunk := await self.process.stdout.read(4096):
                text = decoder.decode(chunk)
                # LeRobot imprime una tabla que mueve el cursor; en la web se muestra como registro.
                text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text).replace("\r", "")
                self.output = (self.output + text)[-24000:]
            code = await self.process.wait()
            saved = (self.directory / f"{self.robot_id}.json").is_file()
            self.state = "cancelled" if self.cancelled else "completed" if code == 0 and saved else "error"
            if self.state == "error":
                self.output += "\nLa calibración no se ha completado. Revisa el registro y la conexión USB.\n"
        except Exception as error:
            self.state = "error"
            self.output += f"\nError al leer LeRobot: {error}\n"

    async def reply(self, text):
        if not self.running or self.process.returncode is not None:
            raise ValueError("No hay una calibración en curso.")
        if len(text) > 80 or "\n" in text or "\r" in text:
            raise ValueError("Envía una sola respuesta de hasta 80 caracteres.")
        self.process.stdin.write((text + "\n").encode())
        try:
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            raise ValueError("LeRobot ya ha cerrado la entrada de la calibración.")

    async def cancel(self):
        async with self.lock:
            if not self.running:
                return
            self.cancelled = True
            try:
                # SIGINT deja ejecutar el finally de LeRobot, que desconecta los motores.
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.task


calibration = CalibrationSession()
router = APIRouter(prefix="/api/calibration")


class CalibrationStart(BaseModel):
    port: str = Field(min_length=1, max_length=200)
    robot_id: str = Field(default="so101", pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class CalibrationReply(BaseModel):
    text: str = Field(default="", max_length=80)


def check_origin(request):
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
        raise HTTPException(403, "Abre la calibración desde la propia web de la estación.")


@router.get("")
async def status():
    return {**calibration.status(), "ports": serial_ports()}


@router.post("/start")
async def start(payload: CalibrationStart, request: Request):
    check_origin(request)
    try:
        await calibration.start(payload.port, payload.robot_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except OSError as error:
        raise HTTPException(503, f"No se ha podido iniciar LeRobot: {error}") from error
    return calibration.status()


@router.post("/reply")
async def reply(payload: CalibrationReply, request: Request):
    check_origin(request)
    try:
        await calibration.reply(payload.text)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"status": "sent"}


@router.post("/cancel")
async def cancel(request: Request):
    check_origin(request)
    await calibration.cancel()
    return calibration.status()
