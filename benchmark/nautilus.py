from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from nautilus import Prior, Sampler

from .adapters import LensCodeAdapter
from .data import MockImage
from .params import BenchmarkParams, EPLShearParams, InversionConfig, SourceGridConfig
from .solver import evaluate_model


NAUTILUS_NAMES = ("theta_E", "gamma", "e1", "e2", "gamma1", "gamma2")
NAUTILUS_BOUNDS = {
    "theta_E": (0.2, 2.0),
    "gamma": (1.5, 2.5),
    "e1": (-0.35, 0.35),
    "e2": (-0.35, 0.35),
    "gamma1": (-0.15, 0.15),
    "gamma2": (-0.15, 0.15),
}


@dataclass(frozen=True)
class NautilusResult:
    params: BenchmarkParams
    samples: dict[str, list[float]]
    log_likelihood: list[float]
    log_weight: list[float]
    log_z: float | None
    n_eff: float | None
    n_like: int | None
    posterior_mean: dict[str, float]
    posterior_median: dict[str, float]


def run_nautilus(
    adapter: LensCodeAdapter,
    mock: MockImage,
    initial: BenchmarkParams,
    source_grid: SourceGridConfig,
    inversion: InversionConfig,
    n_live: int = 100,
    n_eff: int = 200,
    n_like_max: int | None = None,
    seed: int | None = 1,
    filepath: str | Path | None = None,
    resume: bool = False,
    verbose: bool = False,
) -> NautilusResult:
    prior = Prior()
    for name in NAUTILUS_NAMES:
        prior.add_parameter(name, NAUTILUS_BOUNDS[name])

    def loglike(theta: dict[str, Any]) -> float:
        params = replace(initial, lens=lens_params_from_dict(theta, initial.lens))
        result = evaluate_model(adapter, mock, params, source_grid, inversion)
        if not np.isfinite(result.log_likelihood):
            return -np.inf
        return float(result.log_likelihood)

    sampler = Sampler(
        prior,
        loglike,
        n_dim=len(NAUTILUS_NAMES),
        n_live=n_live,
        seed=seed,
        filepath=str(filepath) if filepath is not None else None,
        resume=resume,
    )
    run_kwargs: dict[str, Any] = {"n_eff": n_eff, "verbose": verbose}
    if n_like_max is not None:
        run_kwargs["n_like_max"] = n_like_max
    sampler.run(**run_kwargs)

    samples, log_w, log_l = sampler.posterior(return_as_dict=True)
    samples = {key: np.asarray(value, dtype=float) for key, value in samples.items()}
    log_w = np.asarray(log_w, dtype=float)
    log_l = np.asarray(log_l, dtype=float)
    weights = normalized_weights(log_w)
    posterior_mean = {
        name: float(np.average(samples[name], weights=weights)) for name in NAUTILUS_NAMES
    }
    posterior_median = {name: float(np.median(samples[name])) for name in NAUTILUS_NAMES}
    params = replace(initial, lens=lens_params_from_dict(posterior_mean, initial.lens))

    return NautilusResult(
        params=params,
        samples={key: value.tolist() for key, value in samples.items()},
        log_likelihood=log_l.tolist(),
        log_weight=log_w.tolist(),
        log_z=float(sampler.log_z) if hasattr(sampler, "log_z") else None,
        n_eff=float(sampler.n_eff) if hasattr(sampler, "n_eff") else None,
        n_like=int(sampler.n_like) if hasattr(sampler, "n_like") else None,
        posterior_mean=posterior_mean,
        posterior_median=posterior_median,
    )


def lens_params_from_dict(values: dict[str, Any], template: EPLShearParams) -> EPLShearParams:
    updates = {name: float(np.asarray(values[name])) for name in NAUTILUS_NAMES}
    return replace(template, **updates)


def normalized_weights(log_w: np.ndarray) -> np.ndarray:
    if log_w.size == 0:
        return log_w
    weights = np.exp(log_w - np.max(log_w))
    return weights / np.sum(weights)
