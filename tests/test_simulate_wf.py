import numpy as np

from src.simulation.simulate_wf import GammaSchedule, NuSchedule, simulate_wright_fisher


def test_wright_fisher_returns_valid_ac_values() -> None:
    n_pop = 20
    result = simulate_wright_fisher(
        n_variants=25,
        n_pop=n_pop,
        n_generations=5,
        gamma_schedule=GammaSchedule("constant", {"gamma0": 0.0}),
        nu_schedule=NuSchedule("constant", {"nu": 1.0}),
        initial_frequencies=np.full(25, 0.2),
        seed=4,
    )

    assert result["ac"].shape == (25,)
    assert np.all(result["ac"] >= 0)
    assert np.all(result["ac"] <= n_pop)
    assert np.all(result["an"] == n_pop)


def test_gamma_schedule_modes_are_finite() -> None:
    times = np.linspace(0.0, 1.0, 5)
    schedules = [
        GammaSchedule("constant", {"gamma0": 1.0}),
        GammaSchedule("step", {"gamma_before": 0.0, "gamma_after": 2.0, "t_change": 0.5}),
        GammaSchedule("sinusoidal", {"gamma0": 0.0, "amplitude": 1.0, "period": 1.0}),
        GammaSchedule("piecewise_linear", {"breakpoints": [0.0, 1.0], "values": [0.0, 1.0]}),
    ]

    for schedule in schedules:
        values = schedule(times)
        assert np.all(np.isfinite(values))

