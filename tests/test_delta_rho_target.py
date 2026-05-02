import torch

from dyfo.config import DyFOConfig
from dyfo.core.link_prediction import (
    CorrelationRegressor,
    build_delta_regression_labels,
    compute_regression_metrics,
)


def test_build_delta_regression_labels_uses_only_pairs_present_today_and_tomorrow():
    corr_today = {(0, 1): 0.10, (2, 1): -0.20}
    corr_tomorrow = {(1, 0): 0.15, (1, 2): -0.05, (0, 2): 0.80}

    src, dst, delta = build_delta_regression_labels(corr_tomorrow, corr_today, num_nodes=3)

    assert src.tolist() == [0, 1]
    assert dst.tolist() == [1, 2]
    assert torch.allclose(delta, torch.tensor([0.05, 0.15]), atol=1e-6)


def test_build_delta_regression_labels_returns_zero_for_unchanged_pairs():
    corr_today = {(0, 1): 0.25, (0, 2): -0.35}
    corr_tomorrow = {(0, 1): 0.25, (0, 2): -0.35}

    _, _, delta = build_delta_regression_labels(corr_tomorrow, corr_today, num_nodes=3)

    assert torch.allclose(delta, torch.zeros_like(delta))


def test_correlation_regressor_delta_mode_is_linear_output():
    z_i = torch.ones(2, 4)
    z_j = torch.ones(2, 4)
    absolute = CorrelationRegressor(embedding_dim=4, hidden_dim=8, dropout=0.0, output_mode="absolute")
    delta = CorrelationRegressor(embedding_dim=4, hidden_dim=8, dropout=0.0, output_mode="delta")

    for model in (absolute, delta):
        for param in model.parameters():
            param.data.zero_()
        model.net[-1].bias.data.fill_(2.0)

    assert torch.all(absolute(z_i, z_j) < 1.0)
    assert torch.allclose(delta(z_i, z_j), torch.full((2,), 2.0))


def test_compute_regression_metrics_adds_reconstructed_rho_metrics():
    rho_today = torch.tensor([0.2, -0.1, 0.4])
    targets = torch.tensor([0.1, -0.2, 0.0])
    preds = torch.zeros_like(targets)

    metrics = compute_regression_metrics(preds, targets, rho_today=rho_today)

    assert "mae_reconstructed" in metrics
    assert "r_squared_reconstructed" in metrics
    assert abs(metrics["mae_reconstructed"] - targets.abs().mean().item()) < 1e-6


def test_config_accepts_delta_target_and_delta_baselines():
    assert DyFOConfig(use_delta_target=True).use_delta_target is True
    assert DyFOConfig(model_variant="zero").model_variant == "zero"
    assert DyFOConfig(model_variant="delta_ewma").model_variant == "delta_ewma"
