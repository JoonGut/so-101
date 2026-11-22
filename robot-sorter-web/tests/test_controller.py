"""Regresiones de los comandos usados por la interfaz, sin hardware."""
import asyncio
import unittest

from server.robot_controller import HOME_JOINTS, RobotController


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_one_cycle_can_start_at_once(self):
        robot = RobotController()
        self.assertTrue(robot.trigger_cycle({}))
        self.assertFalse(robot.trigger_cycle({}))
        await robot.return_home()

    async def test_home_cancels_the_trajectory(self):
        robot = RobotController()
        robot.trigger_cycle({})
        await asyncio.sleep(0.2)
        self.assertTrue(await robot.return_home())
        await asyncio.sleep(0.05)
        self.assertEqual(robot.current_joints, HOME_JOINTS)
        self.assertEqual(robot.gripper_percent, 0)
        self.assertEqual(robot.state, "IDLE")
        self.assertIsNone(robot.active_task)
        self.assertTrue(robot._trajectory_task.done())

    async def test_emergency_stop_survives_task_cancellation(self):
        robot = RobotController()
        robot.trigger_cycle({})
        await asyncio.sleep(0.2)
        robot.set_estop(True)
        await asyncio.sleep(0)
        self.assertEqual(robot.state, "EMERGENCY_STOPPED")
        self.assertFalse(robot.trigger_cycle({}))
        self.assertFalse(await robot.return_home())
        robot.set_estop(False)
        self.assertTrue(await robot.return_home())
        self.assertEqual(robot.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
