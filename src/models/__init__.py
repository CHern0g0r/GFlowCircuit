import torch.nn as nn

from .rewards import (
    SizeReward,
    DepthReward,
    ProductOfDiffReward,
    LinearReward,
    ZhuSizeReward,
)
from .Linear import IdEncoder, LinearHead, MLPHead, ValueMLP
from .GCN import GCNEncoder
from .policy import ReinforcePolicy, Policy


REWARD_TYPES = {
    "size": SizeReward,
    "depth": DepthReward,
    "product_of_diff": ProductOfDiffReward,
    "linear": LinearReward,
    "zhu_size": ZhuSizeReward,
}

ENCODERS = {
    "id": IdEncoder,
    "gcn": GCNEncoder,
}

def encoder_factory(encoder_cfg) -> nn.Module:
    if encoder_cfg['type'] in ENCODERS:
        return ENCODERS[encoder_cfg['type']](**encoder_cfg)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_cfg['type']}")

HEADS = {
    "linear": LinearHead,
    "mlp": MLPHead,
}

def head_factory(obs_dim: int, num_actions: int, head_cfg: dict) -> nn.Module:
    if head_cfg['type'] in HEADS:
        return HEADS[head_cfg['type']](obs_dim=obs_dim, num_actions=num_actions, **head_cfg)
    else:
        raise ValueError(f"Unknown head type: {head_cfg['type']}")


VALUE_NETWORKS = {
    "mlp": ValueMLP,
}


def value_factory(obs_dim: int, value_cfg: dict) -> nn.Module:
    value_type = value_cfg.get("type", "mlp")
    if value_type in VALUE_NETWORKS:
        return VALUE_NETWORKS[value_type](in_dim=obs_dim, **value_cfg)
    raise ValueError(f"Unknown value network type: {value_type}")

POLICY_TYPES = {
    "reinforce": ReinforcePolicy,
}

def policy_factory(encoder: nn.Module, head: nn.Module, num_actions: int, policy_cfg: dict) -> Policy:
    if policy_cfg['type'] in POLICY_TYPES:
        return POLICY_TYPES[policy_cfg['type']](
            encoder=encoder,
            head=head,
            num_actions=num_actions,
            **policy_cfg
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_cfg['type']}")

