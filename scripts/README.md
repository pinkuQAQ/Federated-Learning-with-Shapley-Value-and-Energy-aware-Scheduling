# Experiment Scripts

This folder keeps run scripts out of the project root.

- `windows/`: local Windows `.bat` scripts. Each experiment writes to its own subdirectory under `save/`, for example `save/main/<RUN_TAG>/seed42`.
- `slurm/`: Linux/GPU-platform `.sh` scripts intended for SLURM-style runs.

Run Windows scripts from the project root or by double-clicking them; each script resolves the project root relative to its own location.
