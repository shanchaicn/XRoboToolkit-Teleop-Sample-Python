#!/usr/bin/env python3
"""Plot TB6R5 pickle log state/action curves.

Usage:
    PYTHONPATH=. python scripts/misc/plot_tb6r5_pkl_curves.py logs/tb6r5/teleop_log_xxx.pkl
"""

import argparse
import pickle
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _install_pyrealsense2_pickle_stub():
    class DummyFormat:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __setstate__(self, state):
            self.state = state

    if "pyrealsense2.pyrealsense2" in sys.modules:
        return

    pkg = types.ModuleType("pyrealsense2")
    pkg.__path__ = []
    sub = types.ModuleType("pyrealsense2.pyrealsense2")
    sub.format = DummyFormat
    pkg.pyrealsense2 = sub
    sys.modules.setdefault("pyrealsense2", pkg)
    sys.modules.setdefault("pyrealsense2.pyrealsense2", sub)


def load_log(path: Path) -> list[dict]:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if "pyrealsense2" not in str(exc):
            raise
        _install_pyrealsense2_pickle_stub()
        with path.open("rb") as f:
            return pickle.load(f)


def stack_field(data: list[dict], field: str) -> np.ndarray:
    values = []
    for entry in data:
        if field not in entry:
            raise KeyError(f"Field '{field}' not found. Available keys: {sorted(entry.keys())}")
        values.append(np.asarray(entry[field], dtype=float).ravel())
    return np.vstack(values)


def plot_curves(data: list[dict], output: Path, state_key: str, action_key: str, dims: int | None):
    timestamps = np.asarray([entry.get("timestamp", i) for i, entry in enumerate(data)], dtype=float)
    timestamps = timestamps - timestamps[0]

    states = stack_field(data, state_key)
    actions = stack_field(data, action_key)
    if dims is not None:
        states = states[:, :dims]
        actions = actions[:, :dims]

    dim_count = states.shape[1]
    fig, axes = plt.subplots(dim_count, 1, figsize=(14, max(2.0 * dim_count, 6.0)), sharex=True)
    if dim_count == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(timestamps, states[:, i], label=f"{state_key}[{i}]", linewidth=1.5)
        if i < actions.shape[1]:
            ax.plot(timestamps, actions[:, i], label=f"{action_key}[{i}]", linewidth=1.0, linestyle="--")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        ylabel = "gripper" if i == 7 else f"joint {i + 1}"
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{output.stem}: {state_key} vs {action_key}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"Saved curve plot: {output}")
    print(f"Frames: {len(data)}, duration: {timestamps[-1]:.3f}s, state shape: {states.shape}, action shape: {actions.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_file", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--state-key", default="observation")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--dims", type=int, default=None, help="Plot only first N dimensions, e.g. 7 for joints only")
    args = parser.parse_args()

    data = load_log(args.pkl_file)
    if not data:
        raise ValueError(f"No data in {args.pkl_file}")

    output = args.output
    if output is None:
        output = args.pkl_file.with_suffix(".curves.png")
    plot_curves(data, output, args.state_key, args.action_key, args.dims)


if __name__ == "__main__":
    main()
