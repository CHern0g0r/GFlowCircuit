from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.models import encoder_factory, head_factory, prepare_encoder_config


def build_tb_policy(
    cfg: DictConfig,
    *,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
) -> TBGFlowNetPolicy:
    """Build the contract TB policy from a resolved project configuration."""
    encoder_cfg = OmegaConf.to_container(cfg["encoder"], resolve=True)
    if not isinstance(encoder_cfg, dict):
        raise TypeError("encoder config must resolve to a mapping")
    encoder_cfg = prepare_encoder_config(
        encoder_cfg,
        obs_dim=obs_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_actions=num_actions,
        available_actions=available_actions,
    )

    encoder = encoder_factory(encoder_cfg=encoder_cfg)
    head = head_factory(
        obs_dim=encoder_cfg["out_dim"],
        num_actions=num_actions,
        head_cfg=cfg["head"],
    )
    return TBGFlowNetPolicy(encoder=encoder, head=head, num_actions=num_actions)


__all__ = ["build_tb_policy"]
