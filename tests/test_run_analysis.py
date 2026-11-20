"""Unit tests for run convergence analysis and log-loss decay fitting."""

import numpy as np
import pytest

from source.analysis.run_analysis import (
    analyze_history,
    fit_log_loss_decay,
    format_convergence_summary,
)


def test_fit_log_loss_decay_exponential():
    """Verify exponential loss decay recovers known decay rate and R^2 ≈ 1."""
    generations = np.arange(1, 101, dtype=float)
    # Loss = 0.5 * exp(-0.02 * generation)
    # Infidelity = loss -> best_rewards = 1 - loss
    k_true = 0.02
    loss = 0.5 * np.exp(-k_true * generations)
    best_rewards = 1.0 - loss

    fit = fit_log_loss_decay(generations, best_rewards)
    assert fit.valid
    assert pytest.approx(k_true, rel=1e-3) == fit.decay_rate
    assert fit.half_life is not None
    assert pytest.approx(np.log(2.0) / k_true, rel=1e-3) == fit.half_life
    assert fit.r_squared > 0.999


def test_fit_log_loss_decay_flat():
    """Verify that flat/constant rewards return decay rate 0 and no half-life."""
    generations = np.arange(1, 50, dtype=float)
    best_rewards = np.full_like(generations, 0.6)

    fit = fit_log_loss_decay(generations, best_rewards)
    assert not fit.valid or fit.decay_rate == 0.0
    assert fit.half_life is None


def test_analyze_history_milestones_and_stagnation():
    """Verify milestone detection and stagnation flag."""
    generations = np.arange(1, 101)
    # Rapid improvement up to gen 40, then flat
    rewards = []
    for g in generations:
        if g < 20:
            rewards.append(0.5 + 0.02 * g)  # reaches 0.88
        elif g < 40:
            rewards.append(0.88 + 0.0055 * (g - 20))  # crosses 0.90 and 0.99
        else:
            rewards.append(0.991)  # stops at 0.991

    history = {
        "generation": generations,
        "best_reward_ever": np.asarray(rewards),
        "evaluation_count": generations * 16,
        "elapsed_seconds": generations * 2.5,
    }

    summary = analyze_history(history)
    assert summary is not None
    assert summary.total_generations == 100
    assert summary.last_improvement_generation < 45
    assert summary.generations_since_improvement >= 55
    assert summary.is_stagnated

    # Milestones should include 0.90 and 0.99, but not 0.995 or 0.999
    target_pops = [m.target_population for m in summary.milestones]
    assert 0.90 in target_pops
    assert 0.99 in target_pops
    assert 0.995 not in target_pops

    formatted = format_convergence_summary(summary)
    assert "Optimization Dynamics" in formatted
    assert "stagnated" in formatted
    assert "≥0.9" in formatted


def test_analyze_history_empty():
    """Verify graceful handling of empty or missing history data."""
    assert analyze_history({}) is None
    assert analyze_history({"other": np.array([1, 2])}) is None
