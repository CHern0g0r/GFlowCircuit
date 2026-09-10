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
BACKBONE_MATCHED_CONFIG = "tb_zhuDOP_baseline_backbone"


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def test_every_gflownet_experiment_includes_selected_model() -> None:
    discovered = {
        path.stem
        for path in CONFIG_DIR.glob("*.yaml")
        if "algorithm: gflownet_tb" in path.read_text()
    }
    assert discovered == GFLOWNET_CONFIGS | {BACKBONE_MATCHED_CONFIG}

    for config_name in sorted(GFLOWNET_CONFIGS):
        defaults = (CONFIG_DIR / f"{config_name}.yaml").read_text()
        assert "- model: backbone_selected" in defaults

    matched_defaults = (CONFIG_DIR / f"{BACKBONE_MATCHED_CONFIG}.yaml").read_text()
    assert "- override encoder: hybrid_zhu" in matched_defaults
    assert "- model: backbone_selected" not in matched_defaults


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


def test_backbone_matched_src_run_config_freezes_health_settings_and_logging() -> None:
    cfg = _compose(BACKBONE_MATCHED_CONFIG)
    assert OmegaConf.to_container(cfg.head, resolve=True) == {
        "type": "mlp",
        "hidden_dims": [32],
        "activation": "relu",
        "layer_norm": False,
        "dropout": 0.0,
    }
    assert cfg.episodes == 200
    assert cfg.tb.trajectories_per_episode == 4
    assert cfg.tb.log_z_learning_rate == 0.01
    assert cfg.tb.log_z_initialization == "calibrated"
    assert cfg.tb.calibration_trajectories == 64
    assert cfg.tb.calibration_epsilon == 0.5
    assert cfg.tb.reward_alpha == 4.0
    assert cfg.tb.exploration_epsilon_start == 0.5
    assert cfg.tb.exploration_epsilon_end == 0.01
    assert cfg.tb.exploration_warmup_episodes == 20
    assert cfg.logging.tensorboard is True
    assert cfg.discovery_metrics.enabled is True
    assert cfg.discovery_metrics.emit_every_trajectories == 50
    assert cfg.paper_mode.num_runs == 10
