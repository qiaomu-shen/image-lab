# image-lab

Plotting workspace for RL, IL, and RL+IL locomotion experiments.

This repository intentionally includes the experiment logs and generated figures so
that a fresh clone can inspect the existing results and regenerate the plots.

## Quick Start

Install `uv` first if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and prepare the environment:

```bash
git clone https://github.com/qiaomu-shen/image-lab.git
cd image-lab
uv sync --frozen
```

Verify the bundled TensorBoard logs are readable:

```bash
uv run python scripts/plot_teacher_compare.py --list-tags logs/teacher-student消融/16ppo
```

Regenerate the default figures:

```bash
make figures
```

Generated plots are written under `outputs/figures/`.

## Repository Layout

- `scripts/plot_teacher_compare.py`: TensorBoard scalar plotting script.
- `logs/`: bundled experiment logs, checkpoints, ONNX exports, config snapshots, and notes.
- `outputs/figures/`: generated PNG/PDF figures.
- `pyproject.toml` and `uv.lock`: reproducible Python environment.

## Useful Commands

List scalar tags in a run:

```bash
uv run python scripts/plot_teacher_compare.py --list-tags logs/teacher-student消融/16ppo
```

Plot one metric manually:

```bash
uv run python scripts/plot_teacher_compare.py \
  --metric Train/mean_reward \
  --name teacher_student_ablation_mean_reward \
  --runs \
    PPO=logs/teacher-student消融/16ppo \
    PPO=logs/teacher-student消融/42ppo \
    PPO=logs/teacher-student消融/56ppo \
    TeacherKL=logs/teacher-student消融/16kl \
    TeacherKL=logs/teacher-student消融/42kl \
    TeacherKL=logs/teacher-student消融/56kl \
    PureIL=logs/teacher-student消融/16il \
    PureIL=logs/teacher-student消融/42il \
    PureIL=logs/teacher-student消融/56il \
  --colors 'PPO=#4C78A8' 'TeacherKL=#54A24B' 'PureIL=#E45756' \
  --xlabel "Training iteration" \
  --ylabel "Mean reward"
```

The same script accepts `--artifact-intervals match:start_step:end_step` to
linearly interpolate known resume artifacts for matching run paths.
