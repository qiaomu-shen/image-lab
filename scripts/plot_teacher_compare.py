#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_dirs(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return [root.parent]

    event_dirs = sorted({p.parent for p in root.rglob("events.out.tfevents.*")})
    return event_dirs if event_dirs else [root]


def list_scalar_tags(run_dir: str | Path) -> list[str]:
    tags = set()
    for d in find_event_dirs(run_dir):
        try:
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
            tags.update(ea.Tags().get("scalars", []))
        except Exception:
            pass
    return sorted(tags)


def choose_tag(run_dir: str | Path, query: str) -> str:
    tags = list_scalar_tags(run_dir)

    if query in tags:
        return query

    matches = [t for t in tags if query.lower() in t.lower()]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        msg = "\n".join(f"  - {m}" for m in matches[:30])
        raise RuntimeError(
            f"Metric query '{query}' matched multiple TensorBoard tags in:\n"
            f"{run_dir}\n\n"
            f"Please use a more exact --metric.\n\n"
            f"Matches:\n{msg}"
        )

    available = "\n".join(f"  - {t}" for t in tags[:80])
    raise RuntimeError(
        f"Cannot find metric query '{query}' in:\n{run_dir}\n\n"
        f"Available scalar tags:\n{available}"
    )


def load_scalar(run_dir: str | Path, metric_query: str) -> pd.DataFrame:
    run_dir = Path(run_dir).expanduser().resolve()
    tag = choose_tag(run_dir, metric_query)

    frames = []
    for d in find_event_dirs(run_dir):
        try:
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
            if tag not in ea.Tags().get("scalars", []):
                continue
            events = ea.Scalars(tag)
            frames.append(
                pd.DataFrame(
                    {
                        "step": [e.step for e in events],
                        "value": [e.value for e in events],
                        "wall_time": [e.wall_time for e in events],
                    }
                )
            )
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"No scalar data found for tag '{tag}' in {run_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["step"], keep="last")
    df = df.sort_values("step").reset_index(drop=True)
    df["tag"] = tag
    return df


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return (
        pd.Series(y)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def parse_run_arg(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise ValueError(
            f"Invalid run argument: {s}\n"
            f"Use format: Label=/path/to/tensorboard/run"
        )
    label, path = s.split("=", 1)
    return label.strip(), path.strip()


def plot_grouped_runs(args: argparse.Namespace) -> None:
    grouped: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for item in args.runs:
        label, path = parse_run_arg(item)
        df = load_scalar(path, args.metric)

        if args.x_max is not None:
            df = df[df["step"] <= args.x_max]

        if df.empty:
            raise RuntimeError(f"Run became empty after filtering: {label}={path}")

        grouped[label].append(df)

    out_dir = Path(args.output_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.labelsize": args.font_size,
            "axes.titlesize": args.font_size + 1,
            "legend.fontsize": args.font_size - 1,
            "xtick.labelsize": args.font_size - 1,
            "ytick.labelsize": args.font_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    for label, runs in grouped.items():
        if len(runs) == 1:
            df = runs[0]
            x = df["step"].to_numpy()
            y = smooth(df["value"].to_numpy(), args.smooth)
            ax.plot(x, y, label=label, linewidth=args.linewidth)
        else:
            max_common_step = min(df["step"].max() for df in runs)
            min_common_step = max(df["step"].min() for df in runs)
            grid = np.linspace(min_common_step, max_common_step, args.grid_points)

            ys = []
            for df in runs:
                x = df["step"].to_numpy()
                y = smooth(df["value"].to_numpy(), args.smooth)
                ys.append(np.interp(grid, x, y))

            ys = np.vstack(ys)
            mean = ys.mean(axis=0)
            std = ys.std(axis=0)

            ax.plot(grid, mean, label=f"{label} mean", linewidth=args.linewidth)
            ax.fill_between(grid, mean - std, mean + std, alpha=args.alpha)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel if args.ylabel else args.metric)
    if args.title:
        ax.set_title(args.title)

    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = out_dir / f"{args.name}.png"
    pdf_path = out_dir / f"{args.name}.pdf"

    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    print(f"Saved:\n  {png_path}\n  {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--list-tags", type=str, default=None)

    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--metric", type=str, default="Episode")
    parser.add_argument("--name", type=str, default="teacher_compare")
    parser.add_argument("--output-root", type=str, default="outputs/figures")

    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--smooth", type=int, default=11)
    parser.add_argument("--grid-points", type=int, default=800)

    parser.add_argument("--xlabel", type=str, default="Training iteration / step")
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)

    parser.add_argument("--width", type=float, default=5.2)
    parser.add_argument("--height", type=float, default=3.4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=int, default=10)
    parser.add_argument("--linewidth", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.18)

    args = parser.parse_args()

    if args.list_tags:
        tags = list_scalar_tags(args.list_tags)
        print(f"Scalar tags in {args.list_tags}:")
        for t in tags:
            print(f"  {t}")
        return

    if not args.runs:
        parser.error("Please provide --runs, e.g. PPO=/path TeacherKL=/path PureIL=/path")

    plot_grouped_runs(args)


if __name__ == "__main__":
    main()
