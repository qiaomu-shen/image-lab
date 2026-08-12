.PHONY: sync figures figures-reward figures-episode-length

RUNS = \
	PPO=logs/teacher-student消融/16ppo \
	PPO=logs/teacher-student消融/42ppo \
	PPO=logs/teacher-student消融/56ppo \
	TeacherKL=logs/teacher-student消融/16kl \
	TeacherKL=logs/teacher-student消融/42kl \
	TeacherKL=logs/teacher-student消融/56kl \
	PureIL=logs/teacher-student消融/16il \
	PureIL=logs/teacher-student消融/42il \
	PureIL=logs/teacher-student消融/56il

COLORS = PPO=\#4C78A8 TeacherKL=\#54A24B PureIL=\#E45756
ARTIFACT_INTERVALS ?=

sync:
	uv sync --frozen

figures: figures-reward figures-episode-length

figures-reward:
	uv run python scripts/plot_teacher_compare.py \
		--metric Train/mean_reward \
		--name teacher_student_ablation_mean_reward \
		--runs $(RUNS) \
		--colors $(COLORS) \
		--xlabel "Training iteration" \
		--ylabel "Mean reward"
	uv run python scripts/plot_teacher_compare.py \
		--metric Train/mean_reward \
		--name teacher_student_ablation_mean_reward_artifact_corrected \
		--runs $(RUNS) \
		--colors $(COLORS) \
		--xlabel "Training iteration" \
		--ylabel "Mean reward" \
		--artifact-intervals $(ARTIFACT_INTERVALS)

figures-episode-length:
	uv run python scripts/plot_teacher_compare.py \
		--metric Train/mean_episode_length \
		--name teacher_student_ablation_mean_episode_length \
		--runs $(RUNS) \
		--colors $(COLORS) \
		--xlabel "Training iteration" \
		--ylabel "Mean episode length"
	uv run python scripts/plot_teacher_compare.py \
		--metric Train/mean_episode_length \
		--name teacher_student_ablation_mean_episode_length_artifact_corrected \
		--runs $(RUNS) \
		--colors $(COLORS) \
		--xlabel "Training iteration" \
		--ylabel "Mean episode length" \
		--artifact-intervals $(ARTIFACT_INTERVALS)
