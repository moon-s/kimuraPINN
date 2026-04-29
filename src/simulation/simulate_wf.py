"""Single-population Wright-Fisher simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class GammaSchedule:
    """Callable time-varying scaled selection schedule on t in [0, 1]."""

    mode: str
    params: Dict[str, Any]

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if self.mode == "constant":
            values = np.full_like(t_arr, float(self.params.get("gamma0", 0.0)), dtype=float)
        elif self.mode == "step":
            change = float(self.params.get("t_change", 0.5))
            before = float(self.params.get("gamma_before", 0.0))
            after = float(self.params.get("gamma_after", 0.0))
            values = np.where(t_arr < change, before, after)
        elif self.mode in {"sinusoidal", "smooth"}:
            gamma0 = float(self.params.get("gamma0", 0.0))
            amplitude = float(self.params.get("amplitude", self.params.get("A", 0.0)))
            period = float(self.params.get("period", 1.0))
            phase = float(self.params.get("phase", 0.0))
            values = gamma0 + amplitude * np.sin(2.0 * np.pi * (t_arr - phase) / period)
        elif self.mode == "piecewise_linear":
            breakpoints = np.asarray(self.params["breakpoints"], dtype=float)
            values_cfg = np.asarray(self.params["values"], dtype=float)
            values = np.interp(t_arr, breakpoints, values_cfg)
        else:
            raise ValueError(f"Unsupported gamma mode: {self.mode}")
        if np.isscalar(t):
            return float(values)
        return values


@dataclass(frozen=True)
class NuSchedule:
    """Callable relative effective population size schedule nu(t)."""

    mode: str
    params: Dict[str, Any]

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if self.mode == "constant":
            values = np.full_like(t_arr, float(self.params.get("nu", 1.0)), dtype=float)
        elif self.mode == "epoch":
            breakpoints = np.asarray(self.params["breakpoints"], dtype=float)
            epoch_values = np.asarray(self.params["values"], dtype=float)
            indices = np.searchsorted(breakpoints, t_arr, side="right") - 1
            indices = np.clip(indices, 0, len(epoch_values) - 1)
            values = epoch_values[indices]
        else:
            raise ValueError(f"Unsupported nu mode: {self.mode}")
        if np.any(values <= 0):
            raise ValueError("nu(t) must be positive")
        if np.isscalar(t):
            return float(values)
        return values


def make_gamma_schedule(config: Dict[str, Any]) -> GammaSchedule:
    """Build a gamma schedule from a config mapping."""
    return GammaSchedule(mode=str(config.get("mode", "constant")), params=dict(config))


def make_nu_schedule(config: Dict[str, Any]) -> NuSchedule:
    """Build a nu schedule from a config mapping."""
    return NuSchedule(mode=str(config.get("mode", "constant")), params=dict(config))


def draw_initial_frequencies(
    n_variants: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw starting allele frequencies for independent variants."""
    mode = str(config.get("mode", "beta"))
    eps = float(config.get("eps", 1e-4))
    if mode == "fixed":
        frequencies = np.full(n_variants, float(config.get("value", 0.01)), dtype=float)
    elif mode == "uniform":
        low = float(config.get("low", eps))
        high = float(config.get("high", 0.5))
        frequencies = rng.uniform(low, high, size=n_variants)
    elif mode == "beta":
        alpha = float(config.get("alpha", 0.4))
        beta = float(config.get("beta", 8.0))
        frequencies = rng.beta(alpha, beta, size=n_variants)
    else:
        raise ValueError(f"Unsupported initial frequency mode: {mode}")
    return np.clip(frequencies, eps, 1.0 - eps)


def simulate_wright_fisher(
    n_variants: int,
    n_pop: int,
    n_generations: int,
    gamma_schedule: GammaSchedule,
    nu_schedule: Optional[NuSchedule] = None,
    initial_frequencies: Optional[np.ndarray] = None,
    initial_frequency_config: Optional[Dict[str, Any]] = None,
    seed: int = 123,
) -> Dict[str, np.ndarray]:
    """Simulate independent SNV allele frequencies with binomial WF sampling.

    Selection is applied as a small deterministic frequency shift before each
    binomial sampling step, using gamma(t) / (n_pop * nu(t)) as the per-generation
    scaled coefficient. This is intended as a stable validation generator, not a
    full demographic inference simulator.
    """
    if n_variants <= 0:
        raise ValueError("n_variants must be positive")
    if n_pop <= 1:
        raise ValueError("n_pop must be greater than 1")
    if n_generations <= 0:
        raise ValueError("n_generations must be positive")
    rng = np.random.default_rng(seed)
    nu_schedule = nu_schedule or NuSchedule("constant", {"nu": 1.0})
    if initial_frequencies is None:
        initial_frequencies = draw_initial_frequencies(
            n_variants,
            initial_frequency_config or {},
            rng,
        )
    frequencies = np.asarray(initial_frequencies, dtype=float).copy()
    if frequencies.shape != (n_variants,):
        raise ValueError("initial_frequencies must have shape [n_variants]")

    for generation in range(n_generations):
        t = generation / max(n_generations - 1, 1)
        gamma = float(gamma_schedule(t))
        nu = float(nu_schedule(t))
        s = gamma / max(float(n_pop) * nu, 1.0)
        selected = frequencies + s * frequencies * (1.0 - frequencies)
        selected = np.clip(selected, 0.0, 1.0)
        ac = rng.binomial(n_pop, selected)
        frequencies = ac / float(n_pop)

    final_ac = rng.binomial(n_pop, np.clip(frequencies, 0.0, 1.0))
    final_frequency = final_ac / float(n_pop)
    return {
        "ac": final_ac.astype(int),
        "an": np.full(n_variants, n_pop, dtype=int),
        "af": final_frequency,
        "final_frequency": final_frequency,
    }


def make_gamma_trajectory(
    gamma_schedule: GammaSchedule,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a gamma schedule on an evenly spaced time grid."""
    times = np.linspace(0.0, 1.0, n_points)
    gamma = np.asarray(gamma_schedule(times), dtype=float)
    return times, gamma

