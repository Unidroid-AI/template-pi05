# unidroid-template-pi05

Submission template for the `pi05` brain (pi0.5 VLA, OpenPI runtime). It is not tied to a task: it serves any pi0.5 checkpoint. The reference code loads the released `pi05_droid_jointpos` checkpoint. Copy your trained weights into `weights/` and submit.

## Quickstart

1. Put the weights in `weights/`. For the released checkpoint, run `scripts/fetch_weights.sh`. For your own, replace `weights/params/` and `weights/assets/` with your checkpoint.
2. Update the hashes in `agent.yaml`:

   ```bash
   udcli submission hash --path .
   ```

3. Check the folder and run the brain locally:

   ```bash
   brain-comm check --submission .
   mkdir -p /tmp/brain
   brain-comm serve --submission . --socket-dir /tmp/brain --replicas 1
   ```

4. Submit. See the Miner quick start in the `unidroid-cli` README (`udcli submission check`, then `udcli submission submit`). Do not change the pushed weights after submitting. A copy of the unmodified released weights is rejected, but a fine-tune is accepted.

## What you may modify

Everything except the file layout that `udcli submission check` enforces.

| Path | Purpose |
| --- | --- |
| `brain/policy.py` | Inference: loads the checkpoint and answers each observation. |
| `brain/adapt.py` | Observation and action conversion. |
| `train/` | Training notes or scripts. Uploaded in the code bundle, not evaluated. |
| `tests/` | Unit tests. Run with `pytest`. |
| `agent.yaml` | Brain config. The `weights:` block is rewritten by `udcli submission hash`. |
| `weights/` | Checkpoint. Ignored by Git and pushed to your Hugging Face repo at submit time. |

## Brain specifics

`agent.yaml` sets `policy_config` to `pi05_droid_jointpos` and horizon 15.

The reference policy reads an exterior camera, a wrist camera, the arm joints, and the gripper. `agent.yaml` lists those two cameras, so the simulator does not render the others. Camera names, joint count and action dimension are robot-specific; change them in `agent.yaml` and `brain/adapt.py` for your robot and task. OpenPI turns the model's joint-position deltas into absolute joint targets. The gripper closes when its command is above 0.5. Loading the checkpoint requires the pi0.5 runtime from the brain image.

All envs of one simulator scene go through the model in one forward pass, padded to 1, 4, or 16 rows. The three shapes are compiled while the brain starts, and a row's padded size depends only on how many envs the scene has, so results do not depend on timing. `observation.image_hw` is `126x224`, so the simulator resizes frames once on the GPU instead of sending 720x1280 images. JAX reserves `max_gpu_mem_gb` (16 GiB) of the GPU, not a fraction of it, so a larger GPU leaves the rest to the simulator.

The released checkpoint is `gs://openpi-assets-simeval/pi05_droid_jointpos`, also published at `gs://openpi-assets/checkpoints/pi05_droid_jointpos`. After the download, `weights/params/` and `weights/assets/droid/norm_stats.json` exist.
