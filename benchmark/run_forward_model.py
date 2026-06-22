"""Time the parametric forward model (ray-trace + PSF convolution) for each code.

Usage
-----
    python -m benchmark.run_forward_model
    python -m benchmark.run_forward_model --numpix 100 --n-repeat 50 --output results/fm.json
    python -m benchmark.run_forward_model --adapters lenstronomy,jaxtronomy --numpix 200

The script runs each requested adapter, prints a timing table, and optionally
saves results as JSON.  Pass --save-images to also write per-adapter model
images as NPZ files (useful for a quick sanity check).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.forward_model import (
        ForwardModelConfig, ImageConfig, LensConfig, SersicConfig,
        ADAPTERS, get_adapter, time_adapter,
    )
else:
    from .forward_model import (
        ForwardModelConfig, ImageConfig, LensConfig, SersicConfig,
        ADAPTERS, get_adapter, time_adapter,
    )

_DEFAULT_ADAPTERS = "lenstronomy,jaxtronomy,herculens,tinylensgpu,autolens"


def _jax_device_info() -> str:
    try:
        import jax
        devices = jax.devices()
        return ", ".join(str(d) for d in devices)
    except Exception:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time the parametric forward model for each lensing code."
    )
    parser.add_argument(
        "--adapters", default=_DEFAULT_ADAPTERS,
        help="Comma-separated adapter names.",
    )
    parser.add_argument("--numpix", type=int, default=100)
    parser.add_argument("--dpix", type=float, default=0.05,
                        help="Pixel scale in arcsec.")
    parser.add_argument("--psf-fwhm", type=float, default=0.1,
                        help="PSF FWHM in arcsec.")
    parser.add_argument("--n-warmup", type=int, default=3,
                        help="JIT warmup calls before timing (JAX adapters).")
    parser.add_argument("--n-repeat", type=int, default=20,
                        help="Timed calls per adapter.")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results.")
    parser.add_argument("--save-images", action="store_true",
                        help="Save per-adapter model images as NPZ files alongside output.")
    args = parser.parse_args()

    cfg = ForwardModelConfig(
        image=ImageConfig(
            numpix=args.numpix,
            dpix=args.dpix,
            psf_fwhm=args.psf_fwhm,
        ),
    )

    print(f"Forward model benchmark")
    print(f"  numpix={args.numpix}  dpix={args.dpix}\"  psf_fwhm={args.psf_fwhm}\"")
    print(f"  warmup={args.n_warmup}  repeat={args.n_repeat}")
    print(f"  JAX devices: {_jax_device_info()}")
    print()

    adapter_names = [n.strip() for n in args.adapters.split(",") if n.strip()]

    output = {
        "config": {
            "numpix": args.numpix,
            "dpix": args.dpix,
            "psf_fwhm": args.psf_fwhm,
            "n_warmup": args.n_warmup,
            "n_repeat": args.n_repeat,
        },
        "jax_devices": _jax_device_info(),
        "results": {},
    }

    images: dict[str, np.ndarray] = {}

    header = f"{'adapter':>14}  {'median (ms)':>11}  {'min (ms)':>8}  {'mean (ms)':>9}  {'std (ms)':>8}"
    print(header)
    print("-" * len(header))

    for name in adapter_names:
        try:
            adapter = get_adapter(name, cfg)
        except KeyError as exc:
            print(f"  {name}: {exc}")
            continue

        try:
            timing = time_adapter(adapter, n_warmup=args.n_warmup, n_repeat=args.n_repeat)
        except Exception as exc:
            print(f"  {name}: ERROR — {exc}")
            output["results"][name] = {"error": str(exc)}
            continue

        median_ms = timing["median_s"] * 1e3
        min_ms = timing["min_s"] * 1e3
        mean_ms = timing["mean_s"] * 1e3
        std_ms = timing["std_s"] * 1e3

        print(
            f"{name:>14}  {median_ms:>11.3f}  {min_ms:>8.3f}  {mean_ms:>9.3f}  {std_ms:>8.3f}"
        )
        output["results"][name] = timing

        if args.save_images:
            result = adapter()
            import jax
            if adapter.is_jax:
                jax.block_until_ready(result)
            images[name] = np.asarray(result)

    print()

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"Results saved to {path}")

        if args.save_images and images:
            img_path = path.with_suffix(".npz")
            np.savez_compressed(img_path, **images)
            print(f"Model images saved to {img_path}")


if __name__ == "__main__":
    main()
