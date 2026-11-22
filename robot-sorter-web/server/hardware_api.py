"""Endpoints del control del SO-101 real."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .calibration import CALIBRATION_DIR, calibration, check_origin, serial_ports
from .hardware import HardwareBridge
from .robot_controller import robot

hardware = HardwareBridge(CALIBRATION_DIR)
router = APIRouter(prefix="/api/robot")


class ConnectRequest(BaseModel):
    port: str = Field(min_length=1, max_length=200)
    robot_id: str = Field(default="so101", pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class MappingRequest(BaseModel):
    motor: Optional[str] = None
    sign: Optional[int] = None
    offset: Optional[float] = None
    gripper_invert: Optional[bool] = None


def run(action):
    try:
        return action()
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.get("")
async def status():
    return {**hardware.status(), "ports": serial_ports()}


@router.post("/connect")
async def connect(payload: ConnectRequest, request: Request):
    check_origin(request)
    if calibration.running:
        raise HTTPException(409, "Termina o cancela la calibración antes de conectar el robot.")
    from .policy import policy
    if policy.running:
        raise HTTPException(409, "El modelo entrenado está usando el brazo. Páralo antes de conectar.")
    if payload.port not in {device["port"] for device in serial_ports()}:
        raise HTTPException(409, "El puerto USB no está disponible.")
    try:
        await hardware.connect(payload.port, payload.robot_id)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    robot.smooth_home = True
    return hardware.status()


@router.post("/enable")
async def enable(request: Request):
    check_origin(request)
    try:
        await hardware.enable()
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return hardware.status()


@router.post("/disable")
async def disable(request: Request):
    check_origin(request)
    await hardware.disable()
    return hardware.status()


@router.post("/disconnect")
async def disconnect(request: Request):
    check_origin(request)
    await hardware.disconnect()
    robot.smooth_home = False
    return hardware.status()


@router.post("/mapping")
async def mapping(payload: MappingRequest, request: Request):
    check_origin(request)
    run(lambda: hardware.set_mapping(payload.motor, payload.sign, payload.offset, payload.gripper_invert))
    return hardware.status()


@router.post("/rest-from-current")
async def rest_from_current(request: Request):
    check_origin(request)
    run(hardware.set_rest_from_current)
    return hardware.status()
