#!/usr/bin/env python3
"""
Offline evaluation for a trained ACT policy on an existing LeRobot TB6-R5 dataset.

Metric:
  - action MAE (overall + per-dimension) between policy prediction and dataset action.

This helps validate deployment readiness before hardware rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.control_utils import predict_action


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Path (or HF repo) of ACT pretrained checkpoint")
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo_id")
    parser.add_argument("--task", default="tb6r5 teleoperation", help="Task string passed to policy")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda/cpu")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum number of samples to evaluate")
    parser.add_argument("--stride", type=int, default=5, help="Evaluate every N-th sample")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save comparison plots (default: outputs/eval_act/<dataset_name>)",
    )
    return parser


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_hwc_uint8(x: Any) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _plot_comparison_curves(
    indices: list[int],
    preds: np.ndarray,
    targets: np.ndarray,
    abs_err: np.ndarray,
    mae_per_dim: np.ndarray,
    output_dir: Path,
    stride: int,
    fps: float,
) -> None:
    """Plot predicted vs ground-truth action curves and per-dimension MAE bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_dims = preds.shape[1]
    x = np.arange(len(indices)) * stride / fps

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, max(2.0 * n_dims, 6.0)), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x, targets[:, i], label="ground truth", linewidth=1.5, color="C0")
        ax.plot(x, preds[:, i], label="prediction", linewidth=1.0, linestyle="--", color="C1")
        ax.fill_between(x, targets[:, i], preds[:, i], alpha=0.15, color="C3")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(f"joint {i + 1}\n(MAE={mae_per_dim[i]:.4f})")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"ACT offline eval: prediction vs ground truth (n={len(indices)}, stride={stride})")
    fig.tight_layout()
    curve_path = output_dir / "action_comparison.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    dims = np.arange(n_dims)
    ax2.bar(dims, mae_per_dim, color="C2", alpha=0.8)
    ax2.set_xlabel("action dimension")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"Per-dimension MAE (overall={mae_per_dim.mean():.6f})")
    ax2.set_xticks(dims)
    ax2.set_xticklabels([f"joint {i + 1}" for i in dims], rotation=30, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    mae_path = output_dir / "mae_per_dim.png"
    fig2.savefig(mae_path, dpi=150)
    plt.close(fig2)

    err_mean = abs_err.mean(axis=1)
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(x, err_mean, linewidth=1.0, color="C3")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("mean abs error")
    ax3.set_title("Mean absolute error over action dimensions")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    err_path = output_dir / "error_over_time.png"
    fig3.savefig(err_path, dpi=150)
    plt.close(fig3)

    print(f"Saved comparison curves: {curve_path}")
    print(f"Saved MAE bar chart:     {mae_path}")
    print(f"Saved error curve:       {err_path}")


def main() -> int:
    args = _build_parser().parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device

    ds_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_path,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    image_keys = sorted([k for k in ds_meta.features.keys() if k.startswith("observation.images.") and ".depth" not in k])
    state_key = "observation.state"
    action_key = "action"

    n_total = len(dataset)
    indices = list(range(0, n_total, args.stride))[: args.max_samples]
    if not indices:
        raise ValueError("No samples selected. Check --stride and dataset length.")

    abs_err_acc = []
    pred_acc = []
    tgt_acc = []
    for idx in indices:
        sample = dataset[idx]

        obs = {state_key: _to_numpy(sample[state_key]).astype(np.float32)}
        for k in image_keys:
            obs[k] = _to_hwc_uint8(sample[k])

        # Each strided sample is independent: clear the policy's internal action
        # queue so the prediction corresponds to THIS observation only.
        policy.reset()

        pred = predict_action(
            observation=obs,
            policy=policy,
            device=torch.device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=args.task,
            robot_type="tb6r5",
        )
        pred_np = _to_numpy(pred).squeeze(0).astype(np.float32)
        tgt_np = _to_numpy(sample[action_key]).astype(np.float32).reshape(-1)
        if pred_np.shape[0] != tgt_np.shape[0]:
            raise ValueError(f"Action dim mismatch: pred={pred_np.shape}, target={tgt_np.shape} at idx={idx}")

        abs_err_acc.append(np.abs(pred_np - tgt_np))
        pred_acc.append(pred_np)
        tgt_acc.append(tgt_np)

    abs_err = np.stack(abs_err_acc, axis=0)
    preds = np.stack(pred_acc, axis=0)
    targets = np.stack(tgt_acc, axis=0)
    mae_per_dim = abs_err.mean(axis=0)
    mae_all = float(mae_per_dim.mean())

    print("==== ACT Offline Evaluation (TB6R5 / LeRobot) ====")
    print(f"dataset_root: {args.dataset_root}")
    print(f"repo_id:      {args.repo_id}")
    print(f"policy_path:  {args.policy_path}")
    print(f"device:       {policy.config.device}")
    print(f"samples:      {len(indices)} / {len(dataset)} (stride={args.stride})")
    print(f"MAE(all):     {mae_all:.6f}")
    for i, v in enumerate(mae_per_dim):
        print(f"MAE(action[{i}]): {float(v):.6f}")
    print("===============================================")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/eval_act") / Path(args.dataset_root).name
    fps = float(ds_meta.fps) if ds_meta.fps else 50.0
    _plot_comparison_curves(indices, preds, targets, abs_err, mae_per_dim, output_dir, args.stride, fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
