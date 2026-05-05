from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

if __package__ in (None, ""):
    import sys

    script_dir = str(Path(__file__).resolve().parent)
    sys.path = [path for path in sys.path if path != script_dir]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.adapters import get_adapter
    from benchmark.data import load_mock
    from benchmark.nautilus import run_nautilus
    from benchmark.params import BenchmarkParams, EPLShearParams, InversionConfig, SourceGridConfig
    from benchmark.solver import evaluate_model
else:
    from .adapters import get_adapter
    from .data import load_mock
    from .nautilus import run_nautilus
    from .params import BenchmarkParams, EPLShearParams, InversionConfig, SourceGridConfig
    from .solver import evaluate_model


FIT_NAMES = ("theta_E", "gamma", "e1", "e2", "gamma1", "gamma2")
FIT_BOUNDS = (
    (0.2, 2.0),
    (1.5, 2.5),
    (-0.35, 0.35),
    (-0.35, 0.35),
    (-0.15, 0.15),
    (-0.15, 0.15),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one shared EPL+shear + rectangular pixel-source benchmark mock."
    )
    parser.add_argument("--dataset", default="test_simple")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--adapters",
        default="lenstronomy,jaxtronomy,herculens,tinylensgpu,autolens",
        help="Comma-separated adapter names.",
    )
    parser.add_argument("--mode", choices=["evaluate", "fit", "nautilus"], default="evaluate")
    parser.add_argument("--nx-src", type=int, default=25)
    parser.add_argument("--ny-src", type=int, default=25)
    parser.add_argument("--source-half-size", type=float, default=1.2)
    parser.add_argument("--lambda-reg", type=float, default=1.0e-2)
    parser.add_argument("--regularization", choices=["identity", "gradient"], default="gradient")
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--nlive", type=int, default=100)
    parser.add_argument("--n-eff", type=int, default=200)
    parser.add_argument("--n-like-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    mock = load_mock(args.dataset, args.index)
    source_grid = SourceGridConfig(
        nx=args.nx_src,
        ny=args.ny_src,
        x_min=-args.source_half_size,
        x_max=args.source_half_size,
        y_min=-args.source_half_size,
        y_max=args.source_half_size,
    )
    inversion = InversionConfig(
        lambda_reg=args.lambda_reg,
        regularization=args.regularization,
    )

    adapter_names = [name.strip() for name in args.adapters.split(",") if name.strip()]
    output = {
        "dataset": args.dataset,
        "index": args.index,
        "mode": args.mode,
        "source_grid": asdict(source_grid),
        "inversion": asdict(inversion),
        "truth": asdict(mock.truth_params),
        "results": {},
    }

    for adapter_name in adapter_names:
        adapter = get_adapter(adapter_name)
        started = time.perf_counter()
        sampler_output = None
        if args.mode == "nautilus":
            sampler_result = run_nautilus(
                adapter,
                mock,
                mock.truth_params,
                source_grid,
                inversion,
                n_live=args.nlive,
                n_eff=args.n_eff,
                n_like_max=args.n_like_max,
                seed=args.seed,
                verbose=True,
            )
            params = sampler_result.params
            fit_info = {
                "success": True,
                "message": "nautilus completed",
                "nfev": sampler_result.n_like,
                "log_z": sampler_result.log_z,
                "n_eff": sampler_result.n_eff,
                "posterior_mean": sampler_result.posterior_mean,
                "posterior_median": sampler_result.posterior_median,
            }
            sampler_output = {
                "samples": sampler_result.samples,
                "log_likelihood": sampler_result.log_likelihood,
                "log_weight": sampler_result.log_weight,
            }
        elif args.mode == "fit":
            params, fit_info = fit_lens_params(adapter, mock, mock.truth_params, source_grid, inversion, args.maxiter)
        else:
            params = mock.truth_params
            fit_info = {"success": None, "message": "evaluated at catalogue truth", "nfev": 1}
        result = evaluate_model(adapter, mock, params, source_grid, inversion)
        elapsed = time.perf_counter() - started
        output["results"][adapter.name] = {
            "params": asdict(params),
            "fit": fit_info,
            "log_likelihood": result.log_likelihood,
            "chi2": result.chi2,
            "reduced_chi2": result.reduced_chi2,
            "reg_penalty": result.reg_penalty,
            "lens_light_amp": result.lens_light_amp,
            "wall_time_s": elapsed,
        }
        if sampler_output is not None:
            output["results"][adapter.name]["sampler"] = sampler_output
        print(
            f"{adapter.name:12s} logL={result.log_likelihood: .3e} "
            f"red_chi2={result.reduced_chi2: .3f} "
            f"lens_amp={result.lens_light_amp: .3e} "
            f"time={elapsed: .2f}s"
        )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def fit_lens_params(adapter, mock, initial, source_grid, inversion, maxiter: int):
    x0 = vector_from_params(initial.lens)

    def objective(vector: np.ndarray) -> float:
        params = replace(initial, lens=params_from_vector(vector, initial.lens))
        result = evaluate_model(adapter, mock, params, source_grid, inversion)
        if not np.isfinite(result.log_likelihood):
            return 1.0e100
        return -result.log_likelihood

    opt = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=FIT_BOUNDS,
        options={"maxiter": maxiter, "ftol": 1.0e-4},
    )
    params = replace(initial, lens=params_from_vector(opt.x, initial.lens))
    return params, {
        "success": bool(opt.success),
        "message": str(opt.message),
        "nfev": int(opt.nfev),
        "objective": float(opt.fun),
    }


def vector_from_params(params: EPLShearParams) -> np.ndarray:
    return np.asarray([getattr(params, name) for name in FIT_NAMES], dtype=float)


def params_from_vector(vector: np.ndarray, template: EPLShearParams) -> EPLShearParams:
    updates = dict(zip(FIT_NAMES, map(float, vector), strict=True))
    return replace(template, **updates)


if __name__ == "__main__":
    main()
