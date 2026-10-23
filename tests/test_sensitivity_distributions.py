"""best_distribution must work with the installed SciPy (kstest with a name fails in 1.18)."""
import numpy as np

from pyhydra.modeling.hydraulic.sensitivity import best_distribution


def test_best_distribution_fits_normal_lognormal_and_gamma_samples():
    rng = np.random.default_rng(0)
    samples = {
        "normal": rng.normal(0.05, 0.005, 200),
        "lognormal": rng.lognormal(mean=-3.0, sigma=0.4, size=200),
        "gamma": rng.gamma(shape=9.0, scale=0.005, size=200),
    }
    for data in samples.values():
        name, params = best_distribution(data)
        assert name in {"norm", "lognorm", "gamma"}
        assert np.all(np.isfinite(params))
