"""Unit checks for generation-boundary checkpointing behavior."""

from __future__ import annotations

import numpy as np

from source.optimization.cmaes import run_cmaes, run_generations


class TinyStrategy:
    """Enough of pycma's interface to test the shared loop without physics."""

    sigma = 0.1

    def ask(self) -> list[np.ndarray]:
        return [np.array([1.0]), np.array([2.0])]

    def tell(self, _candidates, _losses) -> None:
        pass


def test_stop_request_preserves_the_last_completed_generation() -> None:
    checkpoints = []

    best_reward, best_weights, history, completed = run_generations(
        TinyStrategy(),
        lambda _candidates: np.array([0.25, 0.75]),
        generations=3,
        on_generation=checkpoints.append,
        should_stop=lambda: len(checkpoints) >= 1,
    )

    assert not completed
    assert best_reward == 0.75
    np.testing.assert_allclose(best_weights, [2.0])
    assert len(history) == 1
    assert checkpoints[0].evaluation_count == 2
    assert checkpoints[0].history == history
    assert checkpoints[0].sigma == 0.1


def test_evaluated_initial_center_remains_best_when_samples_are_worse() -> None:
    best_reward, best_weights, history, completed = run_generations(
        TinyStrategy(),
        lambda _candidates: np.array([0.25, 0.75]),
        generations=1,
        initial_best_reward=0.9,
        initial_best_weights=np.array([99.0]),
        initial_evaluation_count=1,
    )

    assert completed
    assert best_reward == 0.9
    np.testing.assert_allclose(best_weights, [99.0])
    assert history[0]["best_reward_ever"] == 0.9
    assert history[0]["best_reward_generation"] == 0.75
    assert history[0]["evaluation_count"] == 3


class WorseThanCenterSimulator:
    n_controls = 1

    def final_ground_population(self, weights, times) -> float:
        return 0.25


def test_run_cmaes_preserves_an_explicitly_evaluated_initial_center() -> None:
    initial_weights = np.array([0.0])

    result = run_cmaes(
        WorseThanCenterSimulator(),
        physics_config={},
        times=np.array([0.0, 1.0]),
        generations=1,
        population_size=2,
        sigma0=0.1,
        seed=1234,
        initial_weights=initial_weights,
        initial_reward=0.9,
    )

    assert result.best_reward == 0.9
    np.testing.assert_allclose(result.best_weights, initial_weights)
    assert result.history[0]["best_reward_ever"] == 0.9
    assert result.evaluation_count == 3
