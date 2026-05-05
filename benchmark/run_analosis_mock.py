from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

if __package__ in (None, ""):
    import sys as _sys

    _script_dir = str(Path(__file__).resolve().parent)
    _sys.path = [p for p in _sys.path if p != _script_dir]
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.adapters import get_adapter
    from benchmark.data import MockImage, make_image_grid, make_noise_map, make_psf_kernel
    from benchmark.nautilus import run_nautilus
    from benchmark.params import BenchmarkParams, EPLShearParams, InversionConfig, SersicLightParams, SourceGridConfig, params_from_truth
    from benchmark.plotting import write_diagnostics
    from benchmark.solver import evaluate_model
else:
    from .adapters import get_adapter
    from .data import MockImage, make_image_grid, make_noise_map, make_psf_kernel
    from .nautilus import run_nautilus
    from .params import BenchmarkParams, EPLShearParams, InversionConfig, SersicLightParams, SourceGridConfig, params_from_truth
    from .plotting import write_diagnostics
    from .solver import evaluate_model


ANALOSIS_ROOT = Path(__file__).resolve().parents[2] / "analosis"

FIT_NAMES = ("theta_E", "gamma", "e1", "e2", "gamma1", "gamma2")
FIT_BOUNDS = (
    (0.2, 2.0),
    (1.5, 2.5),
    (-0.35, 0.35),
    (-0.35, 0.35),
    (-0.15, 0.15),
    (-0.15, 0.15),
)


def build_analosis_mock(
    seed: int | None = None,
    numpix: int = 100,
    einstein_radius_min: float = 0.5,
    gamma_min: float = 1.8,
    gamma_max: float = 2.2,
    min_aspect_ratio: float = 0.7,
    max_shear: float = 0.03,
    max_source_offset_factor: float = 1.0,
    min_aspect_ratio_source: float = 0.9,
    min_aspect_ratio_ll: float = 0.7,
    max_source_perturbations: int = 3,
    source_perturbation_model: str = "gaussian_clumps",
) -> MockImage:
    if str(ANALOSIS_ROOT) not in sys.path:
        sys.path.insert(0, str(ANALOSIS_ROOT))

    from astropy.cosmology import FlatLambdaCDM
    from colossus.cosmology import cosmology as colcos
    from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    from analosis.image.image_generator import Image as AnalosiImage
    from analosis.image.los import LOS
    from analosis.image.simple import Simple
    from analosis.image.source import Source
    from analosis.utilities.useful_functions import Utilities

    if seed is not None:
        np.random.seed(seed)

    colcos.setCosmology("planck18")
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
    util = Utilities(cosmo, Path("/tmp"))

    redshifts = {
        "lens": np.random.uniform(0.4, 0.6),
        "source": np.random.uniform(1.5, 2.5),
    }
    distances = {
        "os": util.dA(0, redshifts["source"]),
        "od": util.dA(0, redshifts["lens"]),
        "ds": util.dA(redshifts["lens"], redshifts["source"]),
    }

    simple = Simple(
        Einstein_radius_min=einstein_radius_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        min_aspect_ratio=min_aspect_ratio,
        redshifts=redshifts,
        distances=distances,
        util=util,
        min_aspect_ratio_ll=min_aspect_ratio_ll,
    )
    los = LOS(util=util, gamma_max=max_shear, model="SHEAR")
    source = Source(
        redshifts=redshifts,
        distances=distances,
        util=util,
        maximum_source_offset_factor=max_source_offset_factor,
        min_aspect_ratio_source=min_aspect_ratio_source,
        telescope="HST",
        band="WFC3_F160W",
        index=0,
        Einstein_radius=simple.theta_E,
        lens_mass_centre={"x": 0.0, "y": 0.0},
    )

    band = HST(band="WFC3_F160W", psf_type="GAUSSIAN")
    kwargs_band = band.kwargs_single_band()
    pixel_size = band.camera["pixel_scale"]

    kwargs_psf = {
        "psf_type": "GAUSSIAN",
        "fwhm": kwargs_band["seeing"],
        "pixel_size": pixel_size,
        "truncation": 3,
    }
    kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}

    kwargs_shear = {"gamma1": los.kwargs["gamma1"], "gamma2": los.kwargs["gamma2"]}
    kwargs_epl = {
        "theta_E": simple.kwargs["theta_E_epl"],
        "gamma": simple.kwargs["gamma_epl"],
        "e1": simple.kwargs["e1_epl"],
        "e2": simple.kwargs["e2_epl"],
        "center_x": simple.kwargs["center_x_epl"],
        "center_y": simple.kwargs["center_y_epl"],
    }

    main_source_kwargs = {
        "magnitude": source.kwargs["magnitude_sl"],
        "R_sersic": source.kwargs["R_sersic_sl"],
        "n_sersic": source.kwargs["n_sersic_sl"],
        "center_x": source.kwargs["x_sl"],
        "center_y": source.kwargs["y_sl"],
        "e1": source.kwargs["e1_sl"],
        "e2": source.kwargs["e2_sl"],
    }

    image_settings = {
        "max_source_perturbations": max_source_perturbations,
        "source_perturbation_model": source_perturbation_model,
    }
    img_gen = AnalosiImage()
    source_model_list, kwargs_source_mag = img_gen._build_source_components(
        image_settings, main_source_kwargs
    )

    kwargs_ll_mag = [{
        "magnitude": simple.lens_light_kwargs["magnitude_ll"],
        "R_sersic": simple.lens_light_kwargs["R_sersic_ll"],
        "n_sersic": simple.lens_light_kwargs["n_sersic_ll"],
        "center_x": simple.lens_light_kwargs["x_ll"],
        "center_y": simple.lens_light_kwargs["y_ll"],
        "e1": simple.lens_light_kwargs["e1_ll"],
        "e2": simple.lens_light_kwargs["e2_ll"],
    }]

    kwargs_model = {
        "lens_model_list": ["SHEAR", "EPL"],
        "lens_light_model_list": ["SERSIC_ELLIPSE"],
        "source_light_model_list": source_model_list,
    }
    sim = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model)
    kwargs_data = sim.kwargs_data

    kwargs_ll_amp, kwargs_source_amp, _ = sim.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_ll_mag,
        kwargs_source_mag=kwargs_source_mag,
    )

    imSim = sim.image_model_class(kwargs_numerics)
    image = imSim.image(
        kwargs_lens=[kwargs_shear, kwargs_epl],
        kwargs_source=kwargs_source_amp,
        kwargs_lens_light=kwargs_ll_amp,
    )
    image_noisy = image + sim.noise_for_model(model=image)
    kwargs_data["image_data"] = image_noisy

    truth_row = {
        "theta_E_epl": simple.kwargs["theta_E_epl"],
        "gamma_epl": simple.kwargs["gamma_epl"],
        "e1_epl": simple.kwargs["e1_epl"],
        "e2_epl": simple.kwargs["e2_epl"],
        "center_x_epl": simple.kwargs["center_x_epl"],
        "center_y_epl": simple.kwargs["center_y_epl"],
        "gamma1": los.kwargs["gamma1"],
        "gamma2": los.kwargs["gamma2"],
        "R_sersic_ll": simple.lens_light_kwargs["R_sersic_ll"],
        "n_sersic_ll": simple.lens_light_kwargs["n_sersic_ll"],
        "e1_ll": simple.lens_light_kwargs["e1_ll"],
        "e2_ll": simple.lens_light_kwargs["e2_ll"],
        "x_ll": simple.lens_light_kwargs["x_ll"],
        "y_ll": simple.lens_light_kwargs["y_ll"],
    }

    image_arr = np.asarray(kwargs_data["image_data"], dtype=float)
    x_grid, y_grid = make_image_grid(kwargs_data, image_arr.shape)
    psf_kernel = make_psf_kernel(kwargs_psf)
    noise_map = make_noise_map(image_arr, kwargs_data)

    truth_params = params_from_truth(truth_row)

    source_truth = {
        "source_light_model_list": source_model_list,
        "kwargs_source_mag": kwargs_source_mag,
        "kwargs_source_amp": kwargs_source_amp,
    }

    return MockImage(
        dataset="analosis_simple",
        index=seed if seed is not None else -1,
        image=image_arr,
        noise_map=noise_map,
        x_grid=x_grid,
        y_grid=y_grid,
        psf_kernel=psf_kernel,
        truth_row=truth_row,
        truth_params=truth_params,
        source_truth=source_truth,
        kwargs_data=kwargs_data,
        kwargs_psf=kwargs_psf,
        kwargs_numerics=kwargs_numerics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one EPL+shear benchmark mock generated on the fly by analosis."
    )
    parser.add_argument(
        "--adapters",
        default="lenstronomy,jaxtronomy,herculens,tinylensgpu,autolens",
        help="Comma-separated adapter names.",
    )
    parser.add_argument("--mode", choices=["evaluate", "fit", "nautilus"], default="evaluate")
    parser.add_argument("--numpix", type=int, default=100, help="Image size in pixels.")
    parser.add_argument("--einstein-radius-min", type=float, default=0.5)
    parser.add_argument("--gamma-min", type=float, default=1.8)
    parser.add_argument("--gamma-max", type=float, default=2.2)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.7)
    parser.add_argument("--max-shear", type=float, default=0.03)
    parser.add_argument("--max-source-perturbations", type=int, default=3)
    parser.add_argument("--source-perturbation-model", default="gaussian_clumps",
                        choices=["gaussian_clumps", "sersic"])
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
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()

    mock = build_analosis_mock(
        seed=args.seed,
        numpix=args.numpix,
        einstein_radius_min=args.einstein_radius_min,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        min_aspect_ratio=args.min_aspect_ratio,
        max_shear=args.max_shear,
        max_source_perturbations=args.max_source_perturbations,
        source_perturbation_model=args.source_perturbation_model,
    )

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

    print(f"Mock: numpix={args.numpix}  seed={args.seed}")
    print(f"Truth: theta_E={mock.truth_row['theta_E_epl']:.3f}  "
          f"gamma={mock.truth_row['gamma_epl']:.3f}  "
          f"e1={mock.truth_row['e1_epl']:.3f}  "
          f"e2={mock.truth_row['e2_epl']:.3f}  "
          f"gamma1={mock.truth_row['gamma1']:.4f}  "
          f"gamma2={mock.truth_row['gamma2']:.4f}")

    adapter_names = [name.strip() for name in args.adapters.split(",") if name.strip()]
    output = {
        "dataset": "analosis_simple",
        "seed": args.seed,
        "numpix": args.numpix,
        "mode": args.mode,
        "source_grid": asdict(source_grid),
        "inversion": asdict(inversion),
        "truth": mock.truth_row,
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
            params, fit_info = fit_lens_params(
                adapter, mock, mock.truth_params, source_grid, inversion, args.maxiter
            )
        else:
            params = mock.truth_params
            fit_info = {"success": None, "message": "evaluated at catalogue truth", "nfev": 1}

        result = evaluate_model(adapter, mock, params, source_grid, inversion)
        elapsed = time.perf_counter() - started

        if args.plot_dir:
            write_diagnostics(
                args.plot_dir,
                adapter.name,
                mock,
                result,
                (
                    source_grid.x_min,
                    source_grid.x_max,
                    source_grid.y_min,
                    source_grid.y_max,
                ),
            )

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
