"""
FastAPI Server for SO-101 Robotic Trash Sorting Station
Provides WebSocket telemetry, REST control APIs, and serves the industrial web UI.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .robot_controller import robot, CONTAINER_COORDINATES
from .calibration import calibration, router as calibration_router
from .hardware_api import hardware, router as hardware_router
from .policy import load_params, policy, push_params, save_params, router as policy_router
from .models import router as models_router

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("so101_web")

app = FastAPI(title="SO-101 Robotics Dispatch & Sorter", version="1.0.0")
app.include_router(calibration_router)
app.include_router(hardware_router)
app.include_router(policy_router)
app.include_router(models_router)
from .reference import build_router as build_reference_router
app.include_router(build_reference_router())

# CORS middleware for local network and dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# -------------------------------------------------------------
# Data models
# -------------------------------------------------------------
class ParameterUpdate(BaseModel):
    speed_factor: float = None
    smoothness_factor: float = None
    max_acceleration: float = None
    pan_offset: float = None
    auto_correct: bool = None
    detection_threshold: int = None
    min_pixel_area: int = None
    tracking_smoothing: float = None
    mode: str = None


class TargetDetection(BaseModel):
    label: str = "papel"
    target_bin: str = "blue"
    confidence: float = 0.95
    x_mm: float = 0.0
    y_mm: float = 180.0
    bbox: List[float] = [0, 0, 100, 100]  # [x, y, w, h]


# -------------------------------------------------------------
# WebSocket Connection Manager
# -------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Active clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Active clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        data = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(data)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@app.on_event("shutdown")
async def shutdown_event():
    await calibration.cancel()
    await policy.stop()
    await hardware.disconnect()


# Background broadcast loop for continuous joint & robot telemetry (30Hz)
@app.on_event("startup")
async def startup_event():
    load_params()
    async def telemetry_loop():
        while True:
            try:
                telemetry = robot.get_telemetry()
                await manager.broadcast({
                    "type": "telemetry",
                    "data": telemetry
                })
            except Exception as e:
                logger.error(f"Error in telemetry loop: {e}")
            await asyncio.sleep(0.033)  # ~30 Hz

    asyncio.create_task(telemetry_loop())


# -------------------------------------------------------------
# REST APIs
# -------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Returns current robot telemetry and system state."""
    return robot.get_telemetry()


@app.get("/api/containers")
async def get_containers():
    """Returns available sorting bins and associated waste categories."""
    return CONTAINER_COORDINATES


@app.post("/api/params")
async def set_parameters(params: ParameterUpdate):
    """Updates motion and vision calibration parameters."""
    if calibration.running:
        return JSONResponse(status_code=409, content={"status": "calibrating"})
    clean_params = {k: v for k, v in params.model_dump().items() if v is not None}
    robot.update_parameters(clean_params)
    save_params()
    if policy.running:
        push_params()
    logger.info(f"Parameters updated: {clean_params}")
    return {"status": "ok", "parameters": robot.get_telemetry()["parameters"]}


@app.post("/api/trigger")
async def trigger_cycle(target: TargetDetection):
    """Manually or externally triggers a sorting cycle."""
    if calibration.running:
        return JSONResponse(status_code=409, content={"status": "calibrating"})
    success = robot.trigger_cycle(target.model_dump())
    if not success:
        return JSONResponse(status_code=409, content={"status": "busy_or_stopped", "state": robot.state})
    return {"status": "started", "cycle_id": robot.current_cycle_id, "target": target.model_dump()}


@app.post("/api/estop")
async def toggle_estop(payload: Dict[str, bool]):
    """Engages or clears E-STOP."""
    active = payload.get("active", True)
    if calibration.running:
        if not active:
            return JSONResponse(status_code=409, content={"status": "calibrating"})
        await calibration.cancel()
    if active:
        await policy.stop(emergency=True)
    robot.set_estop(active)
    logger.warning(f"E-STOP active={active}")
    return {"status": "ok", "e_stop": robot.e_stop, "state": robot.state}


# -------------------------------------------------------------
# WebSocket Handler
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial full state
        await websocket.send_text(json.dumps({
            "type": "init",
            "telemetry": robot.get_telemetry(),
            "containers": CONTAINER_COORDINATES
        }))
        
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
                msg_type = msg.get("type")

                if policy.running and msg_type not in ("set_estop", "update_params"):
                    await websocket.send_json({"type": "error", "message": "El modelo entrenado está controlando el brazo. Páralo antes de usar los controles."})
                    continue

                if calibration.running:
                    if msg_type == "set_estop" and msg.get("data", {}).get("active", True):
                        await calibration.cancel()
                        robot.set_estop(True)
                    else:
                        await websocket.send_json({"type": "error", "message": "Termina o cancela la calibración del robot antes de usar los controles."})
                    continue
                
                if msg_type == "update_params":
                    params = msg.get("data", {})
                    if policy.running:
                        # Durante el modelo no se cambia el modo de la simulación.
                        params = {k: v for k, v in params.items() if k != "mode"}
                    robot.update_parameters(params)
                    save_params()
                    if policy.running:
                        push_params()
                    
                elif msg_type == "object_detected":
                    # Vision module detected an object on table
                    target_data = msg.get("data", {})
                    if robot.mode == "auto" and robot.state == "IDLE" and not robot.e_stop:
                        robot.trigger_cycle(target_data)
                        
                elif msg_type == "trigger_cycle":
                    target_data = msg.get("data", {})
                    if not robot.trigger_cycle(target_data):
                        await websocket.send_json({"type": "error", "message": "El robot está ocupado o detenido."})
                    
                elif msg_type == "set_estop":
                    active = bool(msg.get("data", {}).get("active", True))
                    if active:
                        await policy.stop(emergency=True)
                    robot.set_estop(active)
                    
                elif msg_type == "home":
                    if not await robot.return_home():
                        await websocket.send_json({"type": "error", "message": "Rearma la estación antes de volver a reposo."})

            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# -------------------------------------------------------------
# Static Files & SPA Routing
# -------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"error": "index.html not found"}

@app.get("/favicon.ico")
async def favicon():
    # Return empty 204 to silence browser console 404
    from fastapi import Response
    return Response(status_code=204)

