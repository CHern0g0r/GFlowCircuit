from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "cfg"
GFLOWNET_CONFIGS = {
    "gflownet_tb_mcnc",
    "tb_zhu",
    "tb_zhuDOP",
    "tb_zhuLinear",
}


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def test_every_gflownet_experiment_includes_selected_model() -> None:
    discovered = {
        path.stem
        for path in CONFIG_DIR.glob("*.yaml")
        if "algorithm: gflownet_tb" in path.read_text()
    }
    assert discovered == GFLOWNET_CONFIGS

    for config_name in sorted(discovered):
        defaults = (CONFIG_DIR / f"{config_name}.yaml").read_text()
        assert "- model: backbone_selected" in defaults


def test_selected_model_resolves_identically_in_every_gflownet_experiment() -> None:
    expected_encoder = {
        "type": "hybrid",
        "merge": "concat",
        "graph": {
            "type": "zhu_gcn",
            "input_graph": True,
            "hidden_dim": 12,
            "out_dim": 4,
            "num_layers": 4,
            "dropout": 0.0,
            "pooling": "mean",
            "node_feature_mode": "auto",
        },
        "vector": {
            "type": "vector_mlp",
            "source": "zhu10",
            "out_dim": 28,
            "hidden_dims": [],
        },
    }
    expected_head = {
        "type": "mlp",
        "hidden_dims": [128, 128],
        "activation": "gelu",
        "layer_norm": True,
        "dropout": 0.0,
    }

    for config_name in sorted(GFLOWNET_CONFIGS):
        cfg = _compose(config_name)
        assert OmegaConf.to_container(cfg.encoder, resolve=True) == expected_encoder
        assert OmegaConf.to_container(cfg.head, resolve=True) == expected_head
