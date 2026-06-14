from __future__ import annotations

from scripts.run_dyfo_drl_walkforward import (
    EvalRecord,
    ExperimentConfig,
    RiskRegularizer,
    RiskRegularizerConfig,
    _bootstrap_daily_diff_ci,
    _build_walk_forward_windows,
    _raw_policy_log_prob,
    _paired_summary,
    _rows_to_frame,
    train_raw_policy,
)
import pandas as pd
import torch


def test_build_walk_forward_windows_uses_disjoint_splits():
    dates = list(range(200))
    windows = _build_walk_forward_windows(
        all_dates=dates,
        train_days=50,
        val_days=20,
        test_days=10,
        step_days=10,
        max_windows=2,
    )

    assert len(windows) == 2
    first = windows[0]
    assert first.train_dates == list(range(0, 50))
    assert first.val_dates == list(range(50, 70))
    assert first.test_dates == list(range(70, 80))
    assert max(first.train_dates) < min(first.val_dates)
    assert max(first.val_dates) < min(first.test_dates)

    second = windows[1]
    assert second.train_dates[0] == 10
    assert second.test_dates[0] == 80


def test_paired_summary_compares_matched_window_seed_episode_rows():
    records = [
        _rec("DyFO-DRL", 0, 0.03),
        _rec("EWMA-GMVP", 0, 0.01),
        _rec("DyFO-DRL", 1, -0.02),
        _rec("EWMA-GMVP", 1, -0.01),
    ]
    df = _rows_to_frame(records)

    stats = _paired_summary(df, "cumulative_log_return", "DyFO-DRL", "EWMA-GMVP")

    assert stats["n"] == 2
    assert stats["wins"] == 1
    assert stats["win_rate"] == 0.5
    assert abs(stats["mean_diff"] - 0.005) < 1e-12
    assert 0.0 <= stats["sign_test_p"] <= 1.0


def test_daily_block_bootstrap_uses_daily_return_differences():
    records = [
        _rec("DyFO-DRL", 0, 0.03, daily=[0.01, 0.02]),
        _rec("EWMA-GMVP", 0, 0.01, daily=[0.00, 0.01]),
    ]
    df = _rows_to_frame(records)

    stats = _bootstrap_daily_diff_ci(
        df,
        lhs="DyFO-DRL",
        rhs="EWMA-GMVP",
        n_bootstrap=20,
        block_len=1,
        seed=42,
    )

    assert stats["n_bootstrap"] == 20
    assert stats["n_daily_diffs"] == 2
    assert abs(stats["mean_daily_diff"] - 0.01) < 1e-12
    assert len(stats["sum_diff_ci95"]) == 2


def test_train_raw_policy_supports_ppo_value_baseline():
    universe = ["A", "B", "C", "D"]
    dates = pd.date_range("2000-01-01", periods=25, freq="D")
    prices = pd.DataFrame(
        {
            "A": [100.0 + i * 0.10 for i in range(25)],
            "B": [90.0 + i * 0.05 for i in range(25)],
            "C": [110.0 - i * 0.02 for i in range(25)],
            "D": [95.0 + ((-1) ** i) * 0.03 for i in range(25)],
        },
        index=dates,
    )
    cfg = ExperimentConfig(
        raw_optimizer="ppo",
        raw_ppo_epochs=2,
        raw_ppo_clip=0.2,
        drl_episodes=2,
        risk=RiskRegularizerConfig(reward_objective="log_return"),
    )

    policy = train_raw_policy(
        ckpt={"universe": universe},
        data={"prices": prices},
        cfg=cfg,
        episodes=[list(range(5, 15)), list(range(10, 20))],
        device=torch.device("cpu"),
        seed=7,
        regularizer=RiskRegularizer(cfg.risk),
        improved=False,
        label="Raw-DRL",
    )

    z = torch.zeros(len(universe), 3)
    weights, _log_prob = _raw_policy_log_prob(policy, z, RiskRegularizer(cfg.risk))

    assert weights.shape == (len(universe),)
    assert torch.all(weights >= 0)
    assert abs(float(weights.sum().detach()) - 1.0) < 1e-6


def _rec(condition: str, episode_idx: int, cum: float, daily=None) -> EvalRecord:
    daily_vals = daily if daily is not None else [cum]
    return EvalRecord(
        window_idx=0,
        grid_id="turnover_0_risk_0_drawdown_0",
        seed=42,
        episode_idx=episode_idx,
        condition=condition,
        test_start="2020-01-01",
        test_end="2020-01-31",
        cumulative_log_return=cum,
        sharpe=1.0,
        sortino=1.0,
        calmar=1.0,
        cvar_5=0.0,
        realized_volatility=0.1,
        max_drawdown=-0.01,
        cdar=0.01,
        mean_entropy=2.0,
        mean_turnover=0.1,
        weights_always_valid=True,
        daily_log_returns=daily_vals,
    )
