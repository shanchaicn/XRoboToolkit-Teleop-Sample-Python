"""LeRobot policy loading and ACT deployment-time overrides."""

from __future__ import annotations

from .lerobot_compat import import_policy_factory, load_pretrained_config, resolve_inference_device


def load_policy_components(policy_path: str, dataset_root: str | None, repo_id: str | None, device: str):
    get_policy_class, make_policy, make_pre_post_processors = import_policy_factory()
    device = resolve_inference_device(device)
    cfg = load_pretrained_config(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device

    dataset_stats = None
    if dataset_root and repo_id:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
        dataset_stats = ds_meta.stats
        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    else:
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_path, config=cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, preprocessor, postprocessor


def apply_act_inference_overrides(
    policy,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    refresh_policy_every_step: bool,
) -> None:
    """Apply deployment-time ACT inference settings (no retraining required)."""
    chunk_size = int(policy.config.chunk_size)
    ckpt_n_action = int(policy.config.n_action_steps)

    if temporal_ensemble_coeff is not None:
        if refresh_policy_every_step:
            raise ValueError(
                "--temporal-ensemble-coeff and --refresh-policy-every-step are incompatible "
                "(reset every step destroys the temporal ensemble buffer)."
            )
        if temporal_ensemble_coeff < 0:
            raise ValueError("--temporal-ensemble-coeff must be >= 0")
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(float(temporal_ensemble_coeff), chunk_size)
        policy.reset()
        print(
            f"[ACT] Temporal Ensemble ON: coeff={temporal_ensemble_coeff:g}, "
            f"chunk_size={chunk_size}, every-step inference"
        )
        return

    new_n_action = ckpt_n_action if n_action_steps is None else int(n_action_steps)
    if not 1 <= new_n_action <= chunk_size:
        raise ValueError(f"--n-action-steps must be in [1, {chunk_size}], got {new_n_action}")

    if new_n_action != ckpt_n_action:
        policy.config.n_action_steps = new_n_action
        policy.reset()
        print(
            f"[ACT] n_action_steps override: {ckpt_n_action} -> {new_n_action} "
            f"(chunk_size={chunk_size}, re-infer every {new_n_action} control steps)"
        )
    elif refresh_policy_every_step:
        print(f"[ACT] Action queue chunk_size={chunk_size}, n_action_steps={new_n_action}, refresh every step")


def act_chunk_info(policy) -> tuple[int | None, int | None]:
    """Return (step_index_in_queue, queue_len) for ACT action queue, if available."""
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        return None, None
    queue_len = getattr(policy.config, "n_action_steps", None)
    queue = getattr(policy, "_action_queue", None)
    if queue_len is None or queue is None:
        return None, queue_len
    remaining = len(queue)
    step_index = max(queue_len - remaining - 1, 0)
    return step_index, queue_len
