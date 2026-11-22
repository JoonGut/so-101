"""Prueba el puente interactivo con un proceso falso; nunca abre un USB real."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException, Request

from server.calibration import CalibrationSession, check_origin
from server.robot_controller import RobotController


class CalibrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = TemporaryDirectory()
        self.session = CalibrationSession(self.folder.name)
        self.robot = RobotController()
        self.launch = asyncio.create_subprocess_exec
        self.calls = []
        self.patches = [
            patch("server.calibration.serial_ports", return_value=[{"port": "/dev/test-robot", "name": "Test"}]),
            patch("server.calibration.importlib.util.find_spec", return_value=object()),
            patch("server.calibration.robot", self.robot),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        await self.session.cancel()
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    async def start_fake(self, save=True):
        calibration_path = str(Path(self.folder.name) / "so101.json")
        script = (
            "import sys, pathlib; "
            "print('Coloca el brazo y pulsa ENTER…', end='', flush=True); "
            "answer = input(); "
            + (f"pathlib.Path({calibration_path!r}).write_text('{{}}'); " if save else "")
            + "print('Respuesta: ' + answer, flush=True)"
        )

        async def launch(*args, **kwargs):
            self.calls.append(args)
            return await self.launch(sys.executable, "-u", "-c", script, **kwargs)

        with patch("server.calibration.asyncio.create_subprocess_exec", side_effect=launch):
            await self.session.start("/dev/test-robot", "so101")
        async with asyncio.timeout(5):
            while "ENTER" not in self.session.output:
                await asyncio.sleep(.01)

    async def test_real_pipe_delivers_prompts_and_responses(self):
        await self.start_fake()
        self.assertTrue(self.robot.e_stop)
        self.assertEqual(self.robot.mode, "manual")
        self.assertIn("server.calibrate_robot", self.calls[0])
        self.assertIn("/dev/test-robot", self.calls[0])
        self.assertIn("--directory", self.calls[0])
        await self.session.reply("c")
        await asyncio.wait_for(self.session.task, 5)
        self.assertEqual(self.session.state, "completed")
        self.assertIn("Respuesta: c", self.session.output)
        self.assertTrue(self.session.status()["saved"])

    async def test_exit_without_saved_file_is_not_a_success(self):
        await self.start_fake(save=False)
        await self.session.reply("")
        await asyncio.wait_for(self.session.task, 5)
        self.assertEqual(self.session.state, "error")
        self.assertFalse(self.session.status()["saved"])

    async def test_second_start_and_multiline_reply_are_rejected(self):
        await self.start_fake()
        with self.assertRaises(ValueError):
            await self.session.start("/dev/test-robot", "so101")
        with self.assertRaises(ValueError):
            await self.session.reply("\n\n")
        await self.session.cancel()
        self.assertEqual(self.session.state, "cancelled")
        self.assertIsNotNone(self.session.process.returncode)
        self.assertTrue(self.robot.e_stop)

    async def test_unknown_ports_and_unsafe_names_never_launch(self):
        with patch("server.calibration.asyncio.create_subprocess_exec") as launch:
            for port, name in [("/dev/unknown", "so101"), ("/dev/test-robot", "../other")]:
                with self.assertRaises(ValueError):
                    await self.session.start(port, name)
            launch.assert_not_called()

    async def test_start_failure_can_be_retried(self):
        with patch("server.calibration.asyncio.create_subprocess_exec", side_effect=OSError("No process")):
            with self.assertRaises(OSError):
                await self.session.start("/dev/test-robot", "so101")
        self.assertEqual(self.session.state, "error")
        await self.start_fake()
        self.assertTrue(self.session.running)

    async def test_saved_calibration_survives_a_new_session(self):
        Path(self.folder.name, "so101.json").write_text(json.dumps({"shoulder_pan": {"id": 1}}))
        self.assertTrue(CalibrationSession(self.folder.name).status()["saved"])

    async def test_simulator_cannot_rearm_or_start_during_calibration(self):
        from server.main import TargetDetection, toggle_estop, trigger_cycle
        await self.start_fake()
        with patch("server.main.calibration", self.session):
            self.assertEqual((await trigger_cycle(TargetDetection())).status_code, 409)
            self.assertEqual((await toggle_estop({"active": False})).status_code, 409)

    def test_other_websites_cannot_start_calibration(self):
        request = Request({
            "type": "http", "scheme": "http", "path": "/api/calibration/start",
            "headers": [(b"host", b"localhost:8080"), (b"origin", b"https://other.example")],
        })
        with self.assertRaises(HTTPException) as error:
            check_origin(request)
        self.assertEqual(error.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
