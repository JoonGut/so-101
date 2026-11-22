"""Puente entre el simulador y el SO-101 follower real mediante LeRobot.

Fases:
  - monitoring: motores sin par. Se lee la posición real y el gemelo 3D la copia,
    para comprobar que cada articulación se corresponde (signo y origen).
  - active: motores con par. Se envía al brazo la posición del simulador, limitando
    cada paso con max_relative_target de LeRobot.

La correspondencia es real = signo * simulador + desplazamiento, por articulación.
Se guarda en <directorio de calibración>/<id>.web_mapping.json.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

from .robot_controller import robot, HOME_JOINTS

logger = logging.getLogger("so101_web.hardware")

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Grados máximos por paso (a 30 Hz, 4° ≈ 120°/s). El gripper va en 0-100.
MAX_STEP = 4.0
RATE_HZ = 30


def default_mapping():
    return {
        "joints": {motor: {"sign": 1, "offset": 0.0} for motor in MOTORS},
        "gripper_invert": False,
    }


class HardwareBridge:
    def __init__(self, calibration_dir):
        self.calibration_dir = Path(calibration_dir)
        self.follower = None
        self.state = "disconnected"  # disconnected, connecting, monitoring, active, error
        self.message = ""
        self.port = None
        self.robot_id = "so101"
        self.real = {}  # última lectura del brazo en sus unidades
        self.mapping = default_mapping()
        self.task = None
        self.lock = asyncio.Lock()
        self.holding = False  # tras una parada se deja de seguir al simulador

    @property
    def connected(self):
        return self.state in ("monitoring", "active")

    def mapping_path(self):
        return self.calibration_dir / f"{self.robot_id}.web_mapping.json"

    def load_mapping(self):
        mapping = default_mapping()
        try:
            saved = json.loads(self.mapping_path().read_text())
            for motor in MOTORS:
                entry = saved.get("joints", {}).get(motor, {})
                mapping["joints"][motor]["sign"] = -1 if entry.get("sign") == -1 else 1
                mapping["joints"][motor]["offset"] = float(entry.get("offset", 0.0))
            mapping["gripper_invert"] = bool(saved.get("gripper_invert", False))
        except (OSError, ValueError, TypeError):
            pass
        self.mapping = mapping

    def save_mapping(self):
        self.mapping_path().write_text(json.dumps(self.mapping, indent=2))

    def status(self):
        sim = robot.current_joints[:5]
        return {
            "state": self.state,
            "message": self.message,
            "port": self.port,
            "robot_id": self.robot_id,
            "calibrated": (self.calibration_dir / f"{self.robot_id}.json").is_file(),
            "real": {k: round(v, 1) for k, v in self.real.items()},
            "sim_as_real": {m: round(v, 1) for m, v in self.sim_to_real(sim, robot.gripper_percent).items()},
            "mapping": self.mapping,
            "holding": self.holding,
        }

    # --- conversión de unidades -------------------------------------------------
    def sim_to_real(self, joints, gripper):
        action = {}
        for motor, value in zip(MOTORS, joints):
            entry = self.mapping["joints"][motor]
            action[f"{motor}.pos"] = entry["sign"] * value + entry["offset"]
        g = max(0.0, min(100.0, gripper))
        action["gripper.pos"] = 100.0 - g if self.mapping["gripper_invert"] else g
        return action

    def real_to_sim(self, real):
        joints = []
        for motor in MOTORS:
            entry = self.mapping["joints"][motor]
            joints.append((real[f"{motor}.pos"] - entry["offset"]) * entry["sign"])
        g = real["gripper.pos"]
        return joints, (100.0 - g if self.mapping["gripper_invert"] else g)

    # --- ciclo de vida -------------------------------------------------------------
    async def connect(self, port, robot_id):
        async with self.lock:
            if self.connected or self.state == "connecting":
                raise ValueError("El robot ya está conectado.")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", robot_id):
                raise ValueError("El nombre admite letras, números, guiones y guiones bajos.")
            if not (self.calibration_dir / f"{robot_id}.json").is_file():
                raise ValueError(f"No hay calibración guardada para «{robot_id}». Calibra el robot primero.")
            self.robot_id = robot_id
            self.port = port
            self.load_mapping()
            self.state = "connecting"
            self.message = "Conectando con el brazo…"
            try:
                await asyncio.to_thread(self._open, port, robot_id)
            except Exception as error:
                self.follower = None
                self.state = "error"
                self.message = f"No se ha podido conectar: {error}"
                raise ValueError(self.message) from error
            # Parada y modo manual: nada se mueve hasta que se active a propósito.
            robot.set_estop(True)
            robot.mode = "manual"
            self.state = "monitoring"
            self.holding = False
            self.message = "Monitorizando sin par: mueve el brazo a mano y comprueba el gemelo 3D."
            self.task = asyncio.create_task(self._loop())

    def _open(self, port, robot_id):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        follower = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, calibration_dir=self.calibration_dir,
            use_degrees=True, max_relative_target=MAX_STEP,
        ))
        follower.connect(calibrate=False)
        if not follower.is_calibrated:
            # Equivale a pulsar Enter en LeRobot: escribe el archivo guardado en los motores.
            follower.bus.write_calibration(follower.calibration)
        follower.bus.disable_torque()
        self.follower = follower
        self.real = follower.get_observation()

    async def enable(self):
        async with self.lock:
            if self.state != "monitoring":
                raise ValueError("Conecta el robot en modo monitorización antes de activar el movimiento.")
            # El simulador empieza exactamente donde está el brazo: sin saltos.
            real = await asyncio.to_thread(self.follower.get_observation)
            self.real = real
            joints, gripper = self.real_to_sim(real)
            robot.current_joints = joints
            robot.gripper_percent = gripper
            robot.speed_factor = min(robot.speed_factor, 0.5)
            await asyncio.to_thread(self._hold_here)
            self.holding = False
            self.state = "active"
            self.message = "Movimiento activo. El brazo sigue al simulador. Rearma la estación para mover."

    def _hold_here(self):
        bus = self.follower.bus
        present = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", present)
        bus.enable_torque()

    async def disable(self):
        async with self.lock:
            if self.state != "active":
                return
            robot.set_estop(True)
            await asyncio.to_thread(self.follower.bus.disable_torque)
            self.state = "monitoring"
            self.message = "Par desactivado. Sujeta el brazo: puede caer por su peso."

    async def disconnect(self):
        async with self.lock:
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
                self.task = None
            if self.follower is not None:
                robot.set_estop(True)
                try:
                    await asyncio.to_thread(self.follower.disconnect)
                except Exception as error:
                    logger.error("Error al desconectar el brazo: %s", error)
                self.follower = None
            self.state = "disconnected"
            self.message = "Robot desconectado. Los controles vuelven a ser solo simulación."
            self.real = {}

    def set_mapping(self, motor, sign=None, offset=None, gripper_invert=None):
        if self.state == "active":
            raise ValueError("Desactiva el movimiento antes de cambiar la correspondencia.")
        if motor is not None:
            if motor not in MOTORS:
                raise ValueError("Articulación desconocida.")
            if sign is not None:
                self.mapping["joints"][motor]["sign"] = -1 if sign == -1 else 1
            if offset is not None:
                self.mapping["joints"][motor]["offset"] = max(-360.0, min(360.0, float(offset)))
        if gripper_invert is not None:
            self.mapping["gripper_invert"] = bool(gripper_invert)
        self.save_mapping()

    def set_rest_from_current(self):
        """Toma la postura actual del brazo como la posición de reposo del simulador."""
        if self.state != "monitoring" or not self.real:
            raise ValueError("Conecta en modo monitorización y coloca el brazo en reposo a mano.")
        for motor, home in zip(MOTORS, HOME_JOINTS):
            entry = self.mapping["joints"][motor]
            entry["offset"] = round(self.real[f"{motor}.pos"] - entry["sign"] * home, 2)
        self.save_mapping()

    # --- bucle de lectura y envío ---------------------------------------------------
    async def _loop(self):
        period = 1 / RATE_HZ
        while True:
            try:
                await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("Fallo de comunicación con el brazo: %s", error)
                self.message = f"Fallo de comunicación con el brazo: {error}"
                robot.set_estop(True)
            await asyncio.sleep(period)

    def _tick(self):
        self.real = self.follower.get_observation()
        if self.state == "monitoring":
            # El gemelo copia al brazo, salvo que se esté ejecutando un ciclo simulado.
            if robot.state in ("IDLE", "EMERGENCY_STOPPED"):
                joints, gripper = self.real_to_sim(self.real)
                robot.current_joints = joints
                robot.gripper_percent = gripper
            return
        if self.state != "active":
            return
        if robot.e_stop:
            # Parada: se mantiene la posición actual y se deja de seguir al simulador.
            if not self.holding:
                bus = self.follower.bus
                bus.sync_write("Goal_Position", bus.sync_read("Present_Position"))
                self.holding = True
            return
        if self.holding:
            # Al rearmar, el simulador continúa desde donde se ha quedado el brazo.
            joints, gripper = self.real_to_sim(self.real)
            robot.current_joints = joints
            robot.gripper_percent = gripper
            self.holding = False
        self.follower.send_action(self.sim_to_real(robot.current_joints[:5], robot.gripper_percent))
