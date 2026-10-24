"""Unit checks for generation-boundary checkpointing behavior."""

from __future__ import annotations

import numpy as np

from source.optimization.cmaes import run_generations


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
