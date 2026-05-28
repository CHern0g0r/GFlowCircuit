import torch.nn as nn

from functools import partial

from src.utils import zhu_vector_dim

from .rewards import (
    SizeReward,
    DepthReward,
    ProductOfDiffReward,
    LinearReward,
    ZhuSizeReward,
    DiffOfProductReward,
)
from .Linear import HybridEncoder, IdEncoder, LinearHead, MLPHead, ValueMLP, VectorMLPEncoder
from .GCN import GCNEncoder
from .zhuGCN import ZhuGCNEncoder
from .policy import ReinforcePolicy, Policy


REWARD_TYPES = {
    "size": SizeReward,
    "depth": DepthReward,
    "product_of_diff": ProductOfDiffReward,
    "diff_of_product": DiffOfProductReward,
    "linear": LinearReward,
    "zhu_size": ZhuSizeReward,
}

def reward_class_factory(reward_cfg: dict) -> type:
    reward_type = reward_cfg.get("type")
    if reward_type in REWARD_TYPES:
        reward_kwargs = {k: v for k, v in reward_cfg.items() if k != "type"}
        if not reward_kwargs:
            return REWARD_TYPES[reward_type]
        return partial(REWARD_TYPES[reward_type], **reward_kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

ENCODERS = {
    "id": IdEncoder,
    "gcn": GCNEncoder,
    "zhu_gcn": ZhuGCNEncoder,
    "vector_mlp": VectorMLPEncoder,
}


def prepare_encoder_config(
    encoder_cfg: dict,
    *,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None = None,
) -> dict:
    cfg = dict(encoder_cfg)
    encoder_type = cfg.get("type")
    if encoder_type == "hybrid":
        if "graph" not in cfg or "vector" not in cfg:
            raise ValueError("hybrid encoder requires 'graph' and 'vector' configs")
        graph_cfg = prepare_encoder_config(
            cfg["graph"],
            obs_dim=obs_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_actions=num_actions,
            available_actions=available_actions,
        )
        vector_cfg = prepare_encoder_config(
            cfg["vector"],
            obs_dim=obs_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_actions=num_actions,
            available_actions=available_actions,
        )
        out_dim = int(graph_cfg["out_dim"]) + int(vector_cfg["out_dim"])
        if "out_dim" in cfg and int(cfg["out_dim"]) != out_dim:
            raise ValueError(f"hybrid out_dim={cfg['out_dim']} does not match branch sum {out_dim}")
        cfg["graph"] = graph_cfg
        cfg["vector"] = vector_cfg
        cfg["out_dim"] = out_dim
        return cfg

    if encoder_type == "vector_mlp":
        source = str(cfg.get("source", "zhu10"))
        if source != "zhu10":
            raise ValueError(f"Unsupported vector encoder source: {source}")
        cfg["source"] = source
        cfg.setdefault("in_dim", zhu_vector_dim(num_actions, available_actions))
        cfg.setdefault("out_dim", cfg["in_dim"])
        return cfg

    if "in_dim" not in cfg:
        if cfg.get("input_graph", False):
            cfg["in_dim"] = node_dim
        else:
            cfg["in_dim"] = obs_dim
    if cfg.get("input_graph", False):
        cfg.setdefault("edge_dim", edge_dim)
    if "out_dim" not in cfg:
        cfg["out_dim"] = obs_dim
    return cfg


def encoder_factory(encoder_cfg) -> nn.Module:
    if encoder_cfg["type"] == "hybrid":
        graph_encoder = encoder_factory(encoder_cfg["graph"])
        vector_encoder = encoder_factory(encoder_cfg["vector"])
        return HybridEncoder(
            graph_encoder=graph_encoder,
            vector_encoder=vector_encoder,
            merge=encoder_cfg.get("merge", "concat"),
            out_dim=encoder_cfg.get("out_dim"),
        )
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


def value_input_dim(
    *,
    obs_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
    value_cfg: dict,
) -> int:
    input_name = str(value_cfg.get("input", "obs_tensor"))
    if input_name in ("zhu10", "vector", "vector_tensor"):
        return zhu_vector_dim(num_actions, available_actions)
    if input_name == "obs_tensor":
        return obs_dim
    raise ValueError(f"Unsupported value input: {input_name}")

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
