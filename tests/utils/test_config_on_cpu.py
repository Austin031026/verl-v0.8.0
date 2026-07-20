# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils import omega_conf_to_dataclass
from verl.utils.config import validate_config


@dataclass
class TestDataclass(BaseConfig):
    hidden_size: int = 0
    activation: str = "relu"


@dataclass
class TestTrainConfig(BaseConfig):
    batch_size: int = 0
    model: TestDataclass = field(default_factory=TestDataclass)
    override_config: dict = field(default_factory=dict)


_cfg_str = """train_config:
  _target_: tests.utils.test_config_on_cpu.TestTrainConfig
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu
  override_config: {}"""


class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    """

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)


class TestPrintCfgCommand(unittest.TestCase):
    """Test suite for the print_cfg.py command-line tool."""

    def test_command_with_override(self):
        """Test that the command runs without error when overriding config values."""
        import subprocess

        # Run the command
        result = subprocess.run(
            ["python3", "scripts/print_cfg.py"],
            capture_output=True,
            text=True,
        )

        # Verify the command exited successfully
        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")

        # Verify the output contains expected config information
        self.assertIn("critic", result.stdout)
        self.assertIn("profiler", result.stdout)


class TestOPSDConfigValidation(unittest.TestCase):
    @staticmethod
    def _make_pure_opsd_config(*, trainer_overrides=None, reward_overrides=None):
        trainer = {"n_gpus_per_node": 1, "nnodes": 1, "val_before_train": False, "test_freq": -1}
        trainer.update(trainer_overrides or {})
        reward = {
            "reward_model": {"enable": False, "enable_resource_pool": False},
            "custom_reward_function": {"path": None},
        }
        if reward_overrides:
            reward.update(reward_overrides)
        return OmegaConf.create(
            {
                "trainer": trainer,
                "actor_rollout_ref": {
                    "actor": {"use_dynamic_bsz": True},
                    "rollout": {"n": 1, "skip": {"enable": False}},
                    "model": {},
                },
                "data": {"train_batch_size": 1},
                "reward": reward,
                "opsd": {
                    "enabled": True,
                    "loss": {"rl_coupling": "none"},
                    "test": {"enabled": False},
                },
            }
        )

    @staticmethod
    def _actor_config():
        return SimpleNamespace(
            ppo_mini_batch_size=1,
            ppo_epochs=1,
            shuffle=False,
            ulysses_sequence_parallel_size=1,
            fsdp_config=SimpleNamespace(ulysses_sequence_parallel_size=1),
            validate=lambda *args: None,
        )

    def test_opsd_rejects_legacy_rollout_dump(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1, "rollout_data_dir": "/tmp/rollouts"},
                "actor_rollout_ref": {
                    "actor": {"use_dynamic_bsz": True},
                    "rollout": {"n": 1, "skip": {"enable": False}},
                    "model": {},
                },
                "data": {"train_batch_size": 1},
                "opsd": {"enabled": True, "test": {"enabled": False}},
            }
        )
        actor_config = SimpleNamespace(
            ppo_mini_batch_size=1,
            ppo_epochs=1,
            shuffle=False,
            ulysses_sequence_parallel_size=1,
            fsdp_config=SimpleNamespace(ulysses_sequence_parallel_size=1),
            validate=lambda *args: None,
        )

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=actor_config):
            with self.assertRaisesRegex(ValueError, "trainer.rollout_data_dir"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_opsd_rejects_rollout_skip(self):
        config = OmegaConf.create(
            {
                "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                "actor_rollout_ref": {
                    "actor": {"use_dynamic_bsz": True},
                    "rollout": {"n": 1, "skip": {"enable": True}},
                    "model": {},
                },
                "data": {"train_batch_size": 1},
                "opsd": {"enabled": True, "test": {"enabled": False}},
            }
        )
        actor_config = SimpleNamespace(
            ppo_mini_batch_size=1,
            ppo_epochs=1,
            shuffle=False,
            ulysses_sequence_parallel_size=1,
            fsdp_config=SimpleNamespace(ulysses_sequence_parallel_size=1),
            validate=lambda *args: None,
        )

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=actor_config):
            with self.assertRaisesRegex(ValueError, "rollout skip/cache"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_pure_opsd_rejects_validation_before_training(self):
        config = self._make_pure_opsd_config(trainer_overrides={"val_before_train": True})

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=self._actor_config()):
            with self.assertRaisesRegex(ValueError, "trainer.val_before_train=False"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_pure_opsd_rejects_periodic_validation(self):
        config = self._make_pure_opsd_config(trainer_overrides={"test_freq": 10})

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=self._actor_config()):
            with self.assertRaisesRegex(ValueError, "trainer.test_freq=-1"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_pure_opsd_rejects_reward_model(self):
        config = self._make_pure_opsd_config(reward_overrides={"reward_model": {"enable": True}})

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=self._actor_config()):
            with self.assertRaisesRegex(ValueError, "reward.reward_model.enable=False"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_pure_opsd_rejects_reward_model_resource_pool(self):
        config = self._make_pure_opsd_config(
            reward_overrides={"reward_model": {"enable": False, "enable_resource_pool": True}}
        )

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=self._actor_config()):
            with self.assertRaisesRegex(ValueError, "reward.reward_model.enable_resource_pool=False"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_pure_opsd_rejects_custom_reward_function(self):
        config = self._make_pure_opsd_config(
            reward_overrides={"custom_reward_function": {"path": "/tmp/fake_reward.py"}}
        )

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=self._actor_config()):
            with self.assertRaisesRegex(ValueError, "reward.custom_reward_function.path"):
                validate_config(config, use_reference_policy=False, use_critic=False)

    def test_opsd_rejects_unimplemented_rl_coupling_before_worker_startup(self):
        config = self._make_pure_opsd_config()
        config.opsd.loss.rl_coupling = "grpo"

        with patch("verl.utils.config.omega_conf_to_dataclass", return_value=self._actor_config()):
            with self.assertRaisesRegex(NotImplementedError, "rl_coupling='grpo'"):
                validate_config(config, use_reference_policy=False, use_critic=False)


if __name__ == "__main__":
    unittest.main()
