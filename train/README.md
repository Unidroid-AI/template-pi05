# Fine-tuning

Evaluation does not run this directory.

`weights/` is the released `pi05_droid_jointpos` checkpoint (JAX orbax params plus DROID norm stats), downloaded as described in the project README. `agent.yaml` lists each file's sha256. `policy_config` is `pi05_droid_jointpos`.

To fine-tune further, start from that checkpoint with OpenPI and the dataset from `task-franka-tabletop/scripts/record_demos.py`. Replace `weights/` with the new checkpoint and update the hashes in `agent.yaml`. Keep the pi0.5 flow architecture: `time_mlp_in` and a 32-D action projection.
