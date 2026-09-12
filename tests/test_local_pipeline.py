from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi05_local import environment as env
from pi05_local.dataset import choose_recovery, dataset_features, safe_remove_tree
from pi05_local.teacher import run_episode


class LocalPipelineUnitTests(unittest.TestCase):
    def test_task_spec_is_deterministic(self) -> None:
        first = env.make_task_spec(123)
        second = env.make_task_spec(123)
        self.assertEqual(first.positions, second.positions)
        self.assertEqual(first.tray_xy, second.tray_xy)
        self.assertNotEqual(first.positions, env.make_task_spec(124).positions)

    def test_target_diversity_and_box_height(self) -> None:
        target_types = [env.make_task_spec(seed).target_type for seed in range(4)]
        self.assertEqual(target_types, ["apple", "orange", "can", "box"])
        self.assertGreaterEqual(float(env.task_config()["tray"]["wall_height"]), 0.06)
        prompt = env.task_prompt(env.make_task_spec(1))
        self.assertIn("orange", prompt)

    def test_recovery_round_robin(self) -> None:
        actual = [choose_recovery("recovery", index) for index in range(8)]
        self.assertEqual(actual, list(env.RECOVERY_TYPES)*2)
        self.assertTrue(all(choose_recovery("nominal", index) == "none" for index in range(8)))

    def test_lerobot_schema(self) -> None:
        features = dataset_features("image")
        self.assertEqual(features["observation.state"]["shape"], (8,))
        self.assertEqual(features["action"]["shape"], (8,))
        self.assertEqual(features["observation.images.image"]["shape"], (3, 256, 256))
        self.assertEqual(features["observation.images.wrist_image"]["dtype"], "image")
        with self.assertRaises(ValueError):
            dataset_features("jpeg")

    def test_bezier_endpoints_and_joint_step(self) -> None:
        points = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0]),
                  np.array([2.0, 0.0, 1.0]), np.array([3.0, 0.0, 0.0])]
        np.testing.assert_allclose(env.cubic_bezier(*points, 0.0), points[0])
        np.testing.assert_allclose(env.cubic_bezier(*points, 1.0), points[-1])
        segment = env.joint_segment(np.zeros(7), np.ones(7)*0.3, seconds=0.1, max_step=0.04)
        jumps = np.diff(np.vstack([np.zeros(7), *segment]), axis=0)
        self.assertLessEqual(float(np.max(np.abs(jumps))), 0.04 + 1e-9)

    def test_safe_delete_scope(self) -> None:
        with self.assertRaises(ValueError):
            safe_remove_tree(Path("/tmp"))
        direct_child = Path(tempfile.mkdtemp(prefix="pi05-delete-test-", dir="/tmp"))
        safe_remove_tree(direct_child)
        self.assertFalse(direct_child.exists())
        with tempfile.TemporaryDirectory(dir="/tmp") as parent:
            child = Path(parent)/"deletable"
            child.mkdir()
            safe_remove_tree(child)
            self.assertFalse(child.exists())


class LocalPipelineIntegrationTests(unittest.TestCase):
    def test_nominal_teacher_episode(self) -> None:
        result = run_episode(env.make_task_spec(0), record_images=False)
        self.assertTrue(result.success)
        self.assertGreater(len(result.frames), 100)
        self.assertLess(result.metadata["max_action_jump"], 0.05)
        self.assertEqual(result.frames[0]["observation.state"].shape, (8,))
        self.assertEqual(result.frames[0]["action"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
