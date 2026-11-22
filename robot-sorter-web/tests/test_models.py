"""Biblioteca de modelos y ajustes en vivo, sin hardware ni modelos reales."""

import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from server.models import BUILTIN_ID, ModelLibrary
from server.run_policy import GraspMonitor, limit_command, motion_settings

COMPATIBLE = {
    "type": "act",
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.gripper": {"type": "VISUAL", "shape": [3, 480, 640]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


class ModelLibraryTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        root = Path(self.folder.name)
        self.builtin = root / "builtin"
        self.builtin.mkdir()
        (self.builtin / "config.json").write_text(json.dumps(COMPATIBLE))
        (self.builtin / "model.safetensors").write_bytes(b"x")
        self.library = ModelLibrary(root / "models", self.builtin)
        self.zip = root / "upload.zip"

    def tearDown(self):
        self.folder.cleanup()

    def test_builtin_is_default(self):
        self.assertEqual(self.library.active_id(), BUILTIN_ID)

    def test_installs_nested_checkpoint_only_with_model_files(self):
        base = "run/checkpoints/last/pretrained_model/"
        make_zip(self.zip, {
            base + "config.json": json.dumps(COMPATIBLE),
            base + "model.safetensors": "pesos",
            base + "policy_preprocessor.json": "{}",
            "run/checkpoints/last/training_state/optimizer.safetensors": "enorme",
        })
        model_id, info = self.library.install_zip(self.zip, "Mi modelo")
        self.assertEqual(model_id, "mi-modelo")
        self.assertTrue(info["compatible"])
        files = sorted(p.name for p in self.library.path_of(model_id).iterdir())
        self.assertEqual(files, ["config.json", "model.safetensors", "policy_preprocessor.json"])
        self.library.select(model_id)
        self.assertEqual(self.library.active_id(), model_id)
        # Un segundo modelo con el mismo nombre no pisa al primero.
        second, _ = self.library.install_zip(self.zip, "Mi modelo")
        self.assertEqual(second, "mi-modelo-2")

    def test_rejects_path_traversal(self):
        make_zip(self.zip, {"../config.json": "{}", "../model.safetensors": "x"})
        with self.assertRaises(ValueError):
            self.library.install_zip(self.zip, "malo")

    def test_rejects_zip_without_model(self):
        make_zip(self.zip, {"leeme.txt": "hola"})
        with self.assertRaises(ValueError):
            self.library.install_zip(self.zip, "vacío")

    def test_incompatible_model_cannot_be_selected(self):
        config = json.loads(json.dumps(COMPATIBLE))
        config["input_features"]["observation.images.top"] = config["input_features"].pop("observation.images.front")
        make_zip(self.zip, {"config.json": json.dumps(config), "model.safetensors": "x"})
        model_id, info = self.library.install_zip(self.zip, "otro robot")
        self.assertFalse(info["compatible"])
        with self.assertRaises(ValueError):
            self.library.select(model_id)

    def test_builtin_cannot_be_deleted_but_uploads_can(self):
        with self.assertRaises(ValueError):
            self.library.delete(BUILTIN_ID)
        make_zip(self.zip, {"config.json": json.dumps(COMPATIBLE), "model.safetensors": "x"})
        model_id, _ = self.library.install_zip(self.zip, "borrable")
        self.library.delete(model_id)
        self.assertEqual([m["id"] for m in self.library.list()], [BUILTIN_ID])


class MotionSettingsTests(unittest.TestCase):
    def test_defaults_match_recorded_speed(self):
        settings = motion_settings({"speed_factor": 1.0, "smoothness_factor": 3, "max_acceleration": 2.5})
        self.assertEqual(settings["fps"], 30)
        self.assertAlmostEqual(settings["max_step"], 15)

    def test_limits(self):
        slow = motion_settings({"speed_factor": 0.01, "smoothness_factor": 10, "max_acceleration": 0})
        self.assertAlmostEqual(slow["fps"], 6)
        self.assertAlmostEqual(slow["alpha"], 0.2)
        self.assertEqual(motion_settings({"speed_factor": 9})["fps"], 60)
        self.assertEqual(motion_settings({"smoothness_factor": 1})["alpha"], 1.0)


class CommandLimitTests(unittest.TestCase):
    def test_gripper_does_not_squeeze_past_margin(self):
        present = [0, 0, 0, 0, 0, 30.0]
        command = limit_command([0, 0, 0, 0, 0, 1.0], present, 50)
        self.assertEqual(command["gripper.pos"], 15.0)  # 30 - 15, no 1

    def test_joints_advance_at_most_max_step(self):
        command = limit_command([40, -40, 5, 0, 0, 50], [0, 0, 0, 0, 0, 50], 15)
        self.assertEqual(command["shoulder_pan.pos"], 15)
        self.assertEqual(command["shoulder_lift.pos"], -15)
        self.assertEqual(command["elbow_flex.pos"], 5)

    def test_opening_gripper_is_not_limited_by_squeeze(self):
        command = limit_command([0, 0, 0, 0, 0, 60], [0, 0, 0, 0, 0, 30], 50)
        self.assertEqual(command["gripper.pos"], 60)


if __name__ == "__main__":
    unittest.main()


class GraspMonitorTests(unittest.TestCase):
    def feed(self, monitor, values):
        return [event for value in values if (event := monitor.update(value))]

    def test_closing_empty_after_opening_is_a_miss(self):
        monitor = GraspMonitor()
        events = self.feed(monitor, [2, 10, 30, 50, 40, 20, 8] + [3] * 8)
        self.assertEqual(events, ["miss"])
        self.assertEqual(monitor.summary(), {"attempts": 1, "misses": 1, "deposits": 0})

    def test_resting_closed_without_opening_is_not_a_miss(self):
        self.assertEqual(self.feed(GraspMonitor(), [2] * 50), [])

    def test_grab_and_deposit_then_closing_empty_is_not_a_miss(self):
        monitor = GraspMonitor()
        grab = [40, 30, 20] + [14] * 16
        deposit = [25, 40] + [3] * 20
        events = self.feed(monitor, grab + deposit)
        self.assertEqual(events, ["grabbed", "released"])
        self.assertEqual(monitor.summary(), {"attempts": 1, "misses": 0, "deposits": 1})

    def test_brief_dip_to_empty_is_not_a_miss(self):
        monitor = GraspMonitor()
        self.assertEqual(self.feed(monitor, [40, 3, 3, 3, 30, 40]), [])

    def test_small_opening_and_squeezed_close_is_a_miss(self):
        monitor = GraspMonitor()
        self.assertEqual(self.feed(monitor, [3, 10, 18, 19, 12, 8] + [7] * 8), ["miss"])

    def test_object_slipping_while_carrying_is_detected(self):
        monitor = GraspMonitor()
        self.assertEqual(self.feed(monitor, [30] + [14] * 16 + [3] * 8), ["grabbed", "dropped"])


class PanOffsetSettingsTests(unittest.TestCase):
    def test_pan_offset_defaults_and_limits(self):
        self.assertEqual(motion_settings({})["pan_offset"], 0.0)
        self.assertFalse(motion_settings({})["auto_correct"])
        self.assertEqual(motion_settings({"pan_offset": 40})["pan_offset"], 15.0)
        self.assertTrue(motion_settings({"auto_correct": True})["auto_correct"])


class GraspReleaseTests(unittest.TestCase):
    def test_small_opening_before_empty_counts_as_deposit(self):
        monitor = GraspMonitor()
        events = [e for v in [30] + [13] * 16 + [15, 17, 16, 11] + [5] * 8 if (e := monitor.update(v))]
        self.assertEqual(events, ["grabbed", "released"])
        self.assertEqual(monitor.summary(), {"attempts": 1, "misses": 0, "deposits": 1})
