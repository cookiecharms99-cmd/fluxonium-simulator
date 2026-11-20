"""Analysis utilities for optimization histories and convergence metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Milestone:
    """A milestone achievement during optimization."""

    target_population: float
    generation: int
    evaluations: int
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ConvergenceFit:
    """Exponential loss decay fit: loss = (1 - P) ~ exp(-k * t)."""

    decay_rate: float
    half_life: float | None
    r_squared: float
    valid: bool


@dataclass(frozen=True)
class ConvergenceSummary:
    """Summary of optimization dynamics extracted from generation history."""

    total_generations: int
    total_evaluations: int
    last_improvement_generation: int
    generations_since_improvement: int
    overall_fit: ConvergenceFit
    recent_fit: ConvergenceFit | None
    milestones: list[Milestone] = field(default_factory=list)

    @property
    def is_stagnated(self) -> bool:
        """Heuristic for whether the optimizer has plateaued."""
        return self.generations_since_improvement >= 30 and self.total_generations >= 60


def fit_log_loss_decay(
    generations: np.ndarray,
    best_rewards: np.ndarray,
    min_points: int = 4,
) -> ConvergenceFit:
    """Fit ln(1 - P(g,0)) = -k * generation + c using ordinary least squares.

    Loss is infidelity L = 1 - P(g,0). Under linear convergence on a log scale,
    L(t) ~ L_0 * exp(-k * t), so the slope of ln(L) vs t is -k.
    The half-life (generations needed to halve infidelity) is ln(2) / k.
    """
    if len(generations) < min_points or len(best_rewards) < min_points:
        return ConvergenceFit(decay_rate=0.0, half_life=None, r_squared=0.0, valid=False)

    loss = np.clip(1.0 - best_rewards, 1e-15, 1.0)
    log_loss = np.log(loss)

    # Check for near-constant data
    if np.allclose(log_loss, log_loss[0]) or np.allclose(generations, generations[0]):
        return ConvergenceFit(decay_rate=0.0, half_life=None, r_squared=0.0, valid=False)

    try:
        slope, intercept = np.polyfit(generations, log_loss, 1)
        predictions = slope * generations + intercept
        ss_res = np.sum((log_loss - predictions) ** 2)
        ss_tot = np.sum((log_loss - np.mean(log_loss)) ** 2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-15 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        decay_rate = float(-slope)
        half_life = float(np.log(2.0) / decay_rate) if decay_rate > 1e-9 else None

        return ConvergenceFit(
            decay_rate=decay_rate,
            half_life=half_life,
            r_squared=r_squared,
            valid=True,
        )
    except Exception:
        return ConvergenceFit(decay_rate=0.0, half_life=None, r_squared=0.0, valid=False)


def analyze_history(
    history: dict[str, np.ndarray],
    milestone_targets: tuple[float, ...] = (0.90, 0.99, 0.995, 0.999),
    recent_window_generations: int = 50,
) -> ConvergenceSummary | None:
    """Analyze generation history to compute convergence rate, milestones, and stagnation."""
    if not history or "generation" not in history or "best_reward_ever" not in history:
        return None

    generations = np.asarray(history["generation"], dtype=float)
    best_rewards = np.asarray(history["best_reward_ever"], dtype=float)

    if len(generations) == 0:
        return None

    evaluations = (
        np.asarray(history["evaluation_count"], dtype=int)
        if "evaluation_count" in history
        else (generations * 16).astype(int)
    )
    elapsed = (
        np.asarray(history["elapsed_seconds"], dtype=float)
        if "elapsed_seconds" in history
        else None
    )

    # Track milestones
    milestones: list[Milestone] = []
    for target in milestone_targets:
        indices = np.where(best_rewards >= target)[0]
        if len(indices) > 0:
            first_idx = indices[0]
            milestones.append(
                Milestone(
                    target_population=target,
                    generation=int(generations[first_idx]),
                    evaluations=int(evaluations[first_idx]),
                    elapsed_seconds=float(elapsed[first_idx]) if elapsed is not None else None,
                )
            )

    # Improvement tracking
    diffs = np.diff(best_rewards)
    improvement_indices = np.where(diffs > 1e-8)[0]
    if len(improvement_indices) > 0:
        last_improvement_idx = improvement_indices[-1] + 1
        last_improvement_gen = int(generations[last_improvement_idx])
    else:
        last_improvement_gen = int(generations[0])

    total_gens = int(generations[-1])
    gens_since_improvement = total_gens - last_improvement_gen

    # Overall exponential loss fit
    overall_fit = fit_log_loss_decay(generations, best_rewards)

    # Recent window fit if run is long enough
    recent_fit: ConvergenceFit | None = None
    if len(generations) >= recent_window_generations:
        recent_fit = fit_log_loss_decay(
            generations[-recent_window_generations:],
            best_rewards[-recent_window_generations:],
        )

    return ConvergenceSummary(
        total_generations=total_gens,
        total_evaluations=int(evaluations[-1]),
        last_improvement_generation=last_improvement_gen,
        generations_since_improvement=gens_since_improvement,
        overall_fit=overall_fit,
        recent_fit=recent_fit,
        milestones=milestones,
    )


def format_convergence_summary(summary: ConvergenceSummary) -> str:
    """Format convergence metrics into a readable terminal summary block."""
    lines: list[str] = ["--- Optimization Dynamics ---"]

    # Overall fit
    if summary.overall_fit.valid and summary.overall_fit.half_life is not None:
        lines.append(
            f"Infidelity decay: t_1/2 ≈ {summary.overall_fit.half_life:.1f} gen "
            f"(k = {summary.overall_fit.decay_rate:.2e}/gen, R² = {summary.overall_fit.r_squared:.2f})"
        )
    elif summary.overall_fit.valid and summary.overall_fit.decay_rate <= 1e-9:
        lines.append("Infidelity decay: negligible progress (flat/stagnated)")
    else:
        lines.append("Infidelity decay: insufficient data for fit")

    # Recent momentum
    if summary.recent_fit is not None and summary.recent_fit.valid:
        if summary.recent_fit.half_life is not None:
            lines.append(
                f"Recent momentum (last 50 gen): t_1/2 ≈ {summary.recent_fit.half_life:.1f} gen "
                f"(R² = {summary.recent_fit.r_squared:.2f})"
            )
        else:
            lines.append("Recent momentum (last 50 gen): stalled / flat")

    # Stagnation & last improvement
    if summary.generations_since_improvement == 0:
        lines.append("Latest gain: achieved in the final recorded generation!")
    else:
        status_note = " [stagnated]" if summary.is_stagnated else ""
        lines.append(
            f"Last improvement: generation {summary.last_improvement_generation} "
            f"({summary.generations_since_improvement} gens ago){status_note}"
        )

    # Milestones
    if summary.milestones:
        ms_parts = []
        for m in summary.milestones:
            if m.elapsed_seconds is not None:
                mins = m.elapsed_seconds / 60.0
                time_str = f"{mins:.1f}m" if mins < 60 else f"{mins / 60.0:.1f}h"
                ms_parts.append(f"≥{m.target_population:g} @ gen {m.generation} ({time_str})")
            else:
                ms_parts.append(f"≥{m.target_population:g} @ gen {m.generation}")
        lines.append("Milestones reached: " + " | ".join(ms_parts))
    else:
        lines.append("Milestones reached: none (did not reach P(g,0) >= 0.90)")

    return "\n".join(lines)
