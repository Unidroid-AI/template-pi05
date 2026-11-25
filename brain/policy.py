# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""pi0.5 DROID joint-position brain reference implementation.

``weights/`` is the released ``pi05_droid_jointpos`` checkpoint. The OpenPI
runtime lives in the brain image. The policy reads the left exterior and wrist
cameras, the seven arm joints, and the gripper, then returns an 8-D
absolute joint command.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from adapt import BATCH_SIZES, IMAGE_HW, joint_chunk, padded_rows, resize_with_pad, row_slices, stack_padded
from brain_comm.brain import Brain

# Flow-matching noise is drawn at the checkpoint width, before the 8-D output slice.
_NOISE_SHAPE = (15, 32)

# OpenPI's ``Policy`` only exposes a one-example ``infer``. Batching reuses its pieces.
_POLICY_PARTS = ("_input_transform", "_output_transform", "_sample_actions", "_sample_kwargs", "_rng")


class Pi05DroidJointposBrain(Brain):
    """Load a pi0.5 DROID joint-position checkpoint and return absolute joint actions."""

    # The only per-episode state is the instruction list. Noise comes from the
    # generator the server reseeds before each act, so workers can share this replica.
    session_safe = True

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        # JAX follows CUDA_VISIBLE_DEVICES, which the server sets for this replica.
        del device
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        _limit_gpu_memory(config.get("max_gpu_mem_gb"))
        self._policy = _load_policy(weights_dir)
        if not _can_batch(self._policy):
            print("pi0.5: this OpenPI build cannot batch; answering one row per call", file=sys.stderr)
        # Compile every batch size now, so no control step waits for XLA.
        _warm_up(self._policy)
        self._instructions: list[str] = []

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        del episode_ids, seeds
        self._instructions = list(instructions)

    def end_episode(self, episode_ids: list[str]) -> None:
        del episode_ids
        self._instructions = []

    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        del step
        joints = np.asarray(obs["joint_pos"], dtype=np.float32)
        grip = np.asarray(obs["gripper_pos"], dtype=np.float32)
        exterior = np.asarray(obs["cam_exterior_left"])
        wrist = np.asarray(obs["cam_wrist"])
        count = joints.shape[0]
        if len(self._instructions) != count:
            raise ValueError(f"{len(self._instructions)} instructions for {count} envs")
        # Noise is drawn per row in row order, from the generator the server reseeded.
        noises = [np.random.standard_normal(_NOISE_SHAPE).astype(np.float32) for _ in range(count)]
        examples = [
            _example(exterior[index], wrist[index], joints[index], grip[index], self._instructions[index])
            for index in range(count)
        ]
        chunks = [joint_chunk(actions, self.horizon) for actions in _infer(self._policy, examples, noises)]
        stacked = np.stack(chunks, axis=0).astype(np.float32)
        if stacked.shape[-1] != self.action_dim:
            raise ValueError(f"action width {stacked.shape[-1]} != {self.action_dim}")
        return stacked


def _load_policy(weights_dir: Path):
    if not (weights_dir / "params" / "_METADATA").is_file():
        raise FileNotFoundError(f"pi0.5 checkpoint is missing params/_METADATA in {weights_dir}")
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.droid_policy as droid_policy
    import openpi.transforms as transforms
    from openpi.models.model import ModelType
    from openpi.policies import policy_config
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    # Same decode as the simeval ``pi05_droid_jointpos`` config: joint deltas
    # become absolute positions, and the gripper stays absolute.
    train_config = TrainConfig(
        name="pi05_droid_jointpos",
        exp_name="inference",
        model=pi0_config.Pi0Config(action_horizon=_NOISE_SHAPE[0], pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda _model: transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[
                    transforms.AbsoluteActions(transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        wandb_enabled=False,
    )
    return policy_config.create_trained_policy(train_config, weights_dir)


def _limit_gpu_memory(max_gpu_mem_gb: float | None) -> None:
    """Reserve at most ``max_gpu_mem_gb`` [GiB] of this GPU for JAX.

    JAX takes ``XLA_PYTHON_CLIENT_MEM_FRACTION`` of the whole GPU, and the brain
    image presets 0.5. A fixed budget wastes less on a large GPU. Without a known
    GPU size the environment is left as it is. Call this before JAX starts.
    """
    if max_gpu_mem_gb is None:
        return
    total_gib = _gpu_total_gib()
    if total_gib is None:
        return
    fraction = min(0.9, float(max_gpu_mem_gb) / total_gib)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.4f}"


def _gpu_total_gib() -> float | None:
    """Return the memory [GiB] of the first visible GPU, without starting CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    command = ["nvidia-smi", f"--id={visible}", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout
        return float(output.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _can_batch(policy) -> bool:
    return not getattr(policy, "_is_pytorch_model", False) and all(hasattr(policy, name) for name in _POLICY_PARTS)


def _infer(policy, examples: list[dict], noises: list[np.ndarray]) -> list[np.ndarray]:
    """Return the policy's actions for each example, in order."""
    if not _can_batch(policy):
        return [policy.infer(example, noise=noise)["actions"] for example, noise in zip(examples, noises)]
    actions: list[np.ndarray] = []
    for rows in row_slices(len(examples)):
        actions.extend(_infer_batch(policy, examples[rows], noises[rows]))
    return actions


def _infer_batch(policy, examples: list[dict], noises: list[np.ndarray]) -> list[np.ndarray]:
    """Run one padded forward pass; same steps as ``Policy.infer`` on a batch."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as openpi_model

    # Transforms may modify their input, so give each one its own shallow copy.
    inputs = [policy._input_transform(dict(example)) for example in examples]
    size = padded_rows(len(inputs))
    batch = stack_padded(inputs, size)
    noise = stack_padded(noises, size)
    policy._rng, key = jax.random.split(policy._rng)
    observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
    sampled = np.asarray(
        policy._sample_actions(key, observation, noise=jnp.asarray(noise), **policy._sample_kwargs)
    )
    state = np.asarray(batch["state"])
    return [
        policy._output_transform({"state": state[row], "actions": sampled[row]})["actions"]
        for row in range(len(examples))
    ]


def _warm_up(policy) -> None:
    blank = np.zeros((*IMAGE_HW, 3), dtype=np.uint8)
    example = _example(blank, blank, np.zeros(7, dtype=np.float32), np.zeros(1, dtype=np.float32), "warm up")
    noise = np.random.RandomState(0).standard_normal(_NOISE_SHAPE).astype(np.float32)
    for size in BATCH_SIZES if _can_batch(policy) else BATCH_SIZES[:1]:
        _infer(policy, [example] * size, [noise] * size)


def _example(exterior: np.ndarray, wrist: np.ndarray, joints: np.ndarray, grip: np.ndarray, prompt: str) -> dict:
    gripper = np.asarray(grip, dtype=np.float32).reshape(-1)
    return {
        "observation/exterior_image_1_left": resize_with_pad(exterior),
        "observation/wrist_image_left": resize_with_pad(wrist),
        "observation/joint_position": np.asarray(joints, dtype=np.float32),
        "observation/gripper_position": gripper,
        "prompt": prompt,
    }

