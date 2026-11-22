"""
SO-101 Robotic Arm Kinematics & Dispatch Controller
Handles trajectory generation, inverse kinematics approximation,
parameter management (speed, smoothness/interpolation), and state machine.
"""

import math
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable


CONTAINER_COORDINATES = {
    "blue": {
        "id": "blue",
        "name": "Contenedor Azul",
        "category": "Papel y Cartón",
        "color_hex": "#2563eb",
        "x": -170.0,
        "y": 190.0,
        "z": 70.0
    },
    "yellow": {
        "id": "yellow",
        "name": "Contenedor Amarillo",
        "category": "Plásticos y Envases",
        "color_hex": "#eab308",
        "x": -60.0,
        "y": 240.0,
        "z": 70.0
    },
    "green": {
        "id": "green",
        "name": "Contenedor Verde",
        "category": "Vidrio",
        "color_hex": "#16a34a",
        "x": 60.0,
        "y": 240.0,
        "z": 70.0
    }
}

# Home resting joint angles for SO-101 (in degrees)
# [J1_base, J2_shoulder, J3_elbow, J4_wrist_pitch, J5_wrist_roll, J6_gripper]
HOME_JOINTS = [0.0, 35.0, -55.0, 20.0, 0.0, 0.0]  # gripper 0% (closed/rest)


class RobotController:
    def __init__(self):
        # Kinematic state
        self.current_joints = list(HOME_JOINTS)
        self.target_joints = list(HOME_JOINTS)
        self.gripper_percent = 0.0  # 0 = closed, 100 = full open
        
        # Operational parameters
        self.speed_factor: float = 1.0        # 0.1x to 3.0x nominal speed
        self.smoothness_factor: float = 1.0   # 1.0 (stiff/linear) to 10.0 (ultra-smooth S-curve)
        self.max_acceleration: float = 2.5    # rad/s^2
        self.pan_offset: float = 0.0          # corrección de la base para el modelo (grados)
        self.auto_correct: bool = False       # ayudas: reintentar y buscar corrección tras fallar
        self.damping_ratio: float = 0.85      # vibration damping
        self.mode: str = "auto"               # "auto", "manual", "step"
        self.e_stop: bool = False             # Emergency stop latch
        
        # Vision calibration parameters
        self.detection_threshold: int = 30
        self.min_pixel_area: int = 3500
        self.tracking_smoothing: float = 0.75  # EMA filter factor
        
        # State machine
        self.state: str = "IDLE"  # IDLE, LOCKING, APPROACHING, DESCENDING, GRASPING, LIFTING, TRANSFERRING, RELEASING, RETRACTING
        self.active_task: Optional[Dict[str, Any]] = None
        self.current_cycle_id: int = 0
        self.total_cycles_completed: int = 0
        self.cycle_start_time: float = 0.0
        self.last_cycle_duration: float = 0.0
        
        # Sorting statistics
        self.sort_stats: Dict[str, int] = dict.fromkeys(CONTAINER_COORDINATES, 0)
        
        # Event callbacks
        self.subscribers: List[Callable[[Dict[str, Any]], Any]] = []
        self._lock = asyncio.Lock()
        self._trajectory_task: Optional[asyncio.Task] = None
        # Con el brazo real conectado, volver a reposo es un movimiento suave, no un salto.
        self.smooth_home: bool = False

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns full telemetry dictionary."""
        return {
            "state": self.state,
            "mode": self.mode,
            "e_stop": self.e_stop,
            "joints": {
                "j1_base": round(self.current_joints[0], 2),
                "j2_shoulder": round(self.current_joints[1], 2),
                "j3_elbow": round(self.current_joints[2], 2),
                "j4_wrist_pitch": round(self.current_joints[3], 2),
                "j5_wrist_roll": round(self.current_joints[4], 2),
                "j6_gripper": round(self.gripper_percent, 1),
            },
            "parameters": {
                "speed_factor": self.speed_factor,
                "smoothness_factor": self.smoothness_factor,
                "max_acceleration": self.max_acceleration,
                "pan_offset": self.pan_offset,
                "auto_correct": self.auto_correct,
                "damping_ratio": self.damping_ratio,
                "detection_threshold": self.detection_threshold,
                "min_pixel_area": self.min_pixel_area,
                "tracking_smoothing": self.tracking_smoothing,
            },
            "active_task": self.active_task,
            "stats": {
                "cycles_completed": self.total_cycles_completed,
                "last_cycle_duration_s": round(self.last_cycle_duration, 2),
                "sort_counts": self.sort_stats,
            },
            "timestamp": time.time()
        }

    def update_parameters(self, params: Dict[str, Any]):
        """Safely updates operational and motion parameters."""
        if "speed_factor" in params:
            self.speed_factor = max(0.1, min(3.5, float(params["speed_factor"])))
        if "smoothness_factor" in params:
            self.smoothness_factor = max(1.0, min(10.0, float(params["smoothness_factor"])))
        if "max_acceleration" in params:
            self.max_acceleration = max(0.5, min(10.0, float(params["max_acceleration"])))
        if "pan_offset" in params:
            self.pan_offset = round(max(-15.0, min(15.0, float(params["pan_offset"]))), 1)
        if "auto_correct" in params:
            self.auto_correct = bool(params["auto_correct"])
        if "detection_threshold" in params:
            self.detection_threshold = max(5, min(100, int(params["detection_threshold"])))
        if "min_pixel_area" in params:
            self.min_pixel_area = max(200, min(25000, int(params["min_pixel_area"])))
        if "tracking_smoothing" in params:
            self.tracking_smoothing = max(0.05, min(0.98, float(params["tracking_smoothing"])))
        if "mode" in params and params["mode"] in ["auto", "manual", "step"]:
            self.mode = params["mode"]

    def set_estop(self, active: bool):
        """Engages or disengages emergency stop."""
        self.e_stop = active
        if active:
            self.state = "EMERGENCY_STOPPED"
            if self._trajectory_task and not self._trajectory_task.done():
                self._trajectory_task.cancel()
        else:
            if self.state == "EMERGENCY_STOPPED":
                self.state = "IDLE"

    def compute_ik_approximation(self, x: float, y: float, z: float) -> List[float]:
        """
        Calculates 6-DOF joint angles for SO-101 follower arm targeting (x, y, z) in mm.
        SO-101 kinematic specs:
          - Base rotation (J1): rotates around vertical axis.
          - Shoulder link (L1 = 115mm): vertical to horizontal pitch.
          - Elbow link (L2 = 135mm): forearm pitch.
          - Wrist pitch (L3 = 70mm): keeps gripper perpendicular to table.
        """
        # 1. Base Angle J1
        j1 = math.degrees(math.atan2(x, y))
        
        # 2. Planar reach in arm plane
        r = math.sqrt(x**2 + y**2) - 25.0  # offset from base rotation axis
        h = z - 45.0                       # offset from shoulder pivot
        
        l1 = 120.0
        l2 = 130.0
        
        # Clamp reach within physical envelope
        d = math.sqrt(r**2 + h**2)
        d_clamped = max(50.0, min(l1 + l2 - 5.0, d))
        
        # Law of cosines for 2-link planar arm
        cos_elbow = (d_clamped**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow_angle_rad = math.acos(cos_elbow)
        
        phi1 = math.atan2(h, r)
        cos_phi2 = (l1**2 + d_clamped**2 - l2**2) / (2 * l1 * d_clamped)
        cos_phi2 = max(-1.0, min(1.0, cos_phi2))
        phi2 = math.acos(cos_phi2)
        
        shoulder_rad = phi1 + phi2
        
        j2 = math.degrees(shoulder_rad) - 45.0
        j3 = -(180.0 - math.degrees(elbow_angle_rad))
        
        # Wrist keeps end effector pointing down toward table
        j4 = -(j2 + j3) - 20.0
        j5 = 0.0  # Wrist roll aligned with target
        
        # Constrain to mechanical limits
        j1 = max(-135.0, min(135.0, j1))
        j2 = max(-30.0, min(95.0, j2))
        j3 = max(-130.0, min(20.0, j3))
        j4 = max(-90.0, min(90.0, j4))
        
        return [j1, j2, j3, j4, j5]

    async def execute_pick_and_place(self, target_info: Dict[str, Any]):
        """
        Orchestrates an autonomous pick & place routine using S-curve trajectory smoothing
        scaled by speed_factor and smoothness_factor.
        """
        if self.e_stop:
            return
            
        async with self._lock:
            self.current_cycle_id += 1
            self.cycle_start_time = time.time()
            self.active_task = target_info
            
            bin_id = target_info.get("target_bin", "blue")
            if bin_id not in CONTAINER_COORDINATES:
                bin_id = "blue"
            container = CONTAINER_COORDINATES[bin_id]
            
            obj_x = float(target_info.get("x_mm", 0.0))
            obj_y = float(target_info.get("y_mm", 180.0))
            
            # Constrain to valid pick table zone
            obj_x = max(-180.0, min(180.0, obj_x))
            obj_y = max(110.0, min(260.0, obj_y))
            
            try:
                # PHASE 1: TARGET LOCK
                self.state = "TARGET_LOCK"
                await asyncio.sleep(0.15)
                
                # Precompute IK points
                # 1. Approach above target (Z = 120 mm)
                approach_joints = self.compute_ik_approximation(obj_x, obj_y, 120.0) + [75.0]  # Gripper opens
                # 2. Descend to object level (Z = 25 mm)
                descend_joints = self.compute_ik_approximation(obj_x, obj_y, 25.0) + [75.0]
                # 3. Grasp (gripper closes)
                grasp_joints = list(descend_joints)
                grasp_joints[5] = 15.0  # Gripper firm grasp
                # 4. Lift vertically (Z = 135 mm)
                lift_joints = self.compute_ik_approximation(obj_x, obj_y, 135.0) + [15.0]
                # 5. Bin overhead (Z = 110 mm)
                bin_x, bin_y, bin_z = container["x"], container["y"], container["z"]
                bin_joints = self.compute_ik_approximation(bin_x, bin_y, bin_z + 40.0) + [15.0]
                # 6. Release (gripper opens over bin)
                release_joints = list(bin_joints)
                release_joints[5] = 80.0
                # 7. Home position
                home_pos = list(HOME_JOINTS)
                home_pos[5] = 0.0
                
                # Execute trajectory stages with smooth interpolation
                stages = [
                    ("APPROACHING", approach_joints, 1.2),
                    ("DESCENDING", descend_joints, 0.9),
                    ("GRASPING", grasp_joints, 0.6),
                    ("LIFTING", lift_joints, 0.9),
                    ("TRANSFERRING", bin_joints, 1.5),
                    ("RELEASING", release_joints, 0.7),
                    ("RETRACTING", home_pos, 1.3),
                ]
                
                for stage_name, target_j, base_duration in stages:
                    if self.e_stop:
                        self.state = "EMERGENCY_STOPPED"
                        return
                        
                    self.state = stage_name
                    # Calculate duration adjusted by speed factor (faster speed = lower duration)
                    duration = max(0.2, base_duration / self.speed_factor)
                    await self._interpolate_to(target_j, duration)
                
                # Successful cycle completion
                self.state = "IDLE"
                self.total_cycles_completed += 1
                self.sort_stats[bin_id] = self.sort_stats.get(bin_id, 0) + 1
                self.last_cycle_duration = time.time() - self.cycle_start_time
                self.active_task = None
                
            except asyncio.CancelledError:
                self.state = "EMERGENCY_STOPPED" if self.e_stop else "IDLE"
                self.active_task = None
                raise

    async def return_home(self):
        """Cancela el ciclo antes de restaurar la posición de la simulación."""
        if self.e_stop:
            return False
        if self._trajectory_task and not self._trajectory_task.done():
            self._trajectory_task.cancel()
            try:
                await self._trajectory_task
            except asyncio.CancelledError:
                pass
        self.active_task = None
        if self.smooth_home:
            self.state = "RETRACTING"
            await self._interpolate_to(list(HOME_JOINTS[:5]) + [0.0], max(1.5, 3.0 / self.speed_factor))
        else:
            self.current_joints = list(HOME_JOINTS)
            self.gripper_percent = 0.0
        self.state = "EMERGENCY_STOPPED" if self.e_stop else "IDLE"
        return True

    async def _interpolate_to(self, target_joints: List[float], duration: float):
        """
        Interpolates from current joints to target joints using Quintic (S-curve) profile.
        smoothness_factor controls the polynomial steepness and damping.
        """
        start_joints = list(self.current_joints)
        start_gripper = self.gripper_percent
        target_gripper = target_joints[5]
        
        # Frame rate for kinematic interpolation: 40 Hz
        fps = 40
        steps = max(4, int(duration * fps))
        step_dt = duration / steps
        
        # Smoothness factor tuning: 1.0 = linear, 10.0 = high-order S-curve
        smooth_exp = 1.0 + (self.smoothness_factor / 10.0) * 2.0  # 1.3 to 3.0
        
        for i in range(1, steps + 1):
            if self.e_stop:
                break
                
            # Normalized progress t from 0.0 to 1.0
            t = i / steps
            
            # Quintic S-Curve interpolation: 6t^5 - 15t^4 + 10t^3
            # or shaped by smoothness factor
            if self.smoothness_factor <= 2.0:
                # Linear / Direct
                progress = t
            elif self.smoothness_factor <= 6.0:
                # Smooth Cubic (Ease in-out)
                progress = 3 * (t**2) - 2 * (t**3)
            else:
                # Quintic S-Curve with zero jerk at start/end
                progress = 6 * (t**5) - 15 * (t**4) + 10 * (t**3)
                
            # Update individual joint positions
            for j in range(5):
                diff = target_joints[j] - start_joints[j]
                self.current_joints[j] = start_joints[j] + diff * progress
                
            # Update gripper
            self.gripper_percent = start_gripper + (target_gripper - start_gripper) * progress
            
            await asyncio.sleep(step_dt)
            
        # Ensure final exact alignment
        self.current_joints = list(target_joints[:5])
        self.gripper_percent = target_gripper

    def trigger_cycle(self, target_info: Dict[str, Any]):
        """Non-blocking start of a pick and place cycle."""
        if self.state not in ["IDLE"]:
            return False
        if self.e_stop:
            return False
            
        # Reserva el ciclo antes de ceder al bucle para evitar disparos duplicados.
        self.state = "TARGET_LOCK"
        loop = asyncio.get_event_loop()
        self._trajectory_task = loop.create_task(self.execute_pick_and_place(target_info))
        return True


# Global controller instance
robot = RobotController()
