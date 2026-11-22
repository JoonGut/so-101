"""Prueba el puente con el brazo real usando un follower falso; nunca abre un USB."""

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from server import hardware as hw
from server.robot_controller import HOME_JOINTS, robot


class FakeBus:
    def __init__(self, arm):
        self.arm = arm
        self.torque = False
        self.goals = []

    def sync_read(self, _name):
        return {k.removesuffix(".pos"): v for k, v in self.arm.position.items()}

    def sync_write(self, _name, values):
        self.goals.append(dict(values))

    def enable_torque(self):
        self.torque = True

    def disable_torque(self):
        self.torque = False


class FakeFollower:
    def __init__(self):
        self.position = {f"{m}.pos": 10.0 for m in hw.MOTORS} | {"gripper.pos": 20.0}
        self.bus = FakeBus(self)
        self.sent = []
        self.disconnected = False

    def get_observation(self):
        return dict(self.position)

    def send_action(self, action):
        self.sent.append(action)
        return action

    def disconnect(self):
        self.disconnected = True


class HardwareTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = TemporaryDirectory()
        Path(self.folder.name, "so101.json").write_text("{}")
        self.bridge = hw.HardwareBridge(self.folder.name)
        self.fake = FakeFollower()

        def fake_open(port, robot_id):
            self.bridge.follower = self.fake
            self.bridge.real = self.fake.get_observation()

        self.bridge._open = fake_open
        robot.set_estop(False)
        robot.state = "IDLE"
        robot.current_joints = list(HOME_JOINTS)

    async def asyncTearDown(self):
        await self.bridge.disconnect()
        robot.set_estop(False)
        robot.mode = "auto"
        self.folder.cleanup()

    async def test_needs_saved_calibration(self):
        Path(self.folder.name, "so101.json").unlink()
        with self.assertRaises(ValueError):
            await self.bridge.connect("/dev/ttyACM0", "so101")

    async def test_monitoring_mirrors_real_arm_without_torque(self):
        await self.bridge.connect("/dev/ttyACM0", "so101")
        self.assertEqual(self.bridge.state, "monitoring")
        self.assertTrue(robot.e_stop)
        self.fake.position["elbow_flex.pos"] = -42.0
        self.bridge._tick()
        self.assertEqual(robot.current_joints[2], -42.0)
        self.assertFalse(self.fake.bus.torque)
        self.assertEqual(self.fake.sent, [])

    async def test_mapping_round_trip_and_rest_offsets(self):
        await self.bridge.connect("/dev/ttyACM0", "so101")
        self.bridge.set_mapping("shoulder_lift", sign=-1)
        self.bridge.set_rest_from_current()
        joints, _ = self.bridge.real_to_sim(self.fake.get_observation())
        for value, home in zip(joints, HOME_JOINTS):
            self.assertAlmostEqual(value, home)
        action = self.bridge.sim_to_real(joints, 20.0)
        self.assertAlmostEqual(action["shoulder_lift.pos"], 10.0)
        self.assertTrue(self.bridge.mapping_path().is_file())

    async def test_enable_starts_from_real_pose_and_follows_simulator(self):
        await self.bridge.connect("/dev/ttyACM0", "so101")
        await self.bridge.enable()
        self.assertEqual(self.bridge.state, "active")
        self.assertTrue(self.fake.bus.torque)
        self.assertEqual(robot.current_joints, [10.0] * 5)
        # Parado tras conectar: no se envía nada hasta rearmar.
        self.bridge._tick()
        self.assertEqual(self.fake.sent, [])
        robot.set_estop(False)
        robot.current_joints[0] = 15.0
        self.bridge._tick()  # al rearmar se sincroniza con el brazo
        self.bridge._tick()
        robot.current_joints[0] = 15.0
        self.bridge._tick()
        self.assertEqual(self.fake.sent[-1]["shoulder_pan.pos"], 15.0)

    async def test_estop_holds_current_position(self):
        await self.bridge.connect("/dev/ttyACM0", "so101")
        await self.bridge.enable()
        robot.set_estop(False)
        self.bridge._tick()
        sent = len(self.fake.sent)
        robot.set_estop(True)
        self.bridge._tick()
        self.bridge._tick()
        self.assertEqual(len(self.fake.sent), sent)
        self.assertEqual(self.fake.bus.goals[-1]["shoulder_pan"], 10.0)

    async def test_mapping_locked_while_active(self):
        await self.bridge.connect("/dev/ttyACM0", "so101")
        await self.bridge.enable()
        with self.assertRaises(ValueError):
            self.bridge.set_mapping("elbow_flex", sign=-1)

    async def test_disconnect_releases_arm(self):
        await self.bridge.connect("/dev/ttyACM0", "so101")
        await self.bridge.disconnect()
        self.assertTrue(self.fake.disconnected)
        self.assertEqual(self.bridge.state, "disconnected")


if __name__ == "__main__":
    unittest.main()
