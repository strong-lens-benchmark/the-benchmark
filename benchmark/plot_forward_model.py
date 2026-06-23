"""Make plots from a forward-model benchmark run.

Reads the JSON written by ``run_forward_model.py`` (timings) and, if present,
the sibling ``.npz`` (per-adapter model images), and writes PNGs.

Usage
-----
    python -m benchmark.plot_forward_model results/forward_model_5343049.json
    python benchmark/plot_forward_model.py results/forward_model_5343049.json --outdir plots

Produces (in --outdir, default alongside the JSON):
  * <stem>_timings.png    bar chart of median time per adapter (log scale, std error bars)
  * <stem>_images.png     grid of model images        (only if the .npz exists)
  * <stem>_residuals.png  per-adapter (image - reference) maps (only if the .npz exists)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe (works on the cluster with no display)
import matplotlib.pyplot as plt

plt.style.use("sanglier")


def _load(json_path: Path):
    data = json.loads(json_path.read_text())
    results = data.get("results", {})
    # keep only adapters that produced timings (skip {"error": ...})
    timed = {k: v for k, v in results.items() if "median_s" in v}
    return data, timed


def plot_timings(timed: dict, cfg: dict, out: Path) -> None:
    names = list(timed)
    median_ms = np.array([timed[n]["median_s"] for n in names]) * 1e3
    std_ms = np.array([timed[n]["std_s"] for n in names]) * 1e3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, median_ms, yerr=std_ms, capsize=4,
                  color="steelblue", edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")  # times span 0.9 -> 150 ms
    ax.set_ylabel("median forward-model time (ms)")
    ax.set_title(
        f"Forward model: numpix={cfg.get('numpix')}, "
        f"dpix={cfg.get('dpix')}\", repeat={cfg.get('n_repeat')}"
    )
    ax.grid(axis="y", which="both", ls=":", alpha=0.5)
    for bar, val in zip(bars, median_ms):
        ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def _image_grid(images: dict, out: Path, title: str, residual_ref: str | None = None,
                shared: bool = False) -> None:
    """Plot a grid of images (or residuals vs ``residual_ref``).

    ``shared=True`` puts every panel on one common colour scale with a single
    figure-wide colourbar (good for comparing magnitudes across codes).
    ``shared=False`` autoscales each panel and gives it its own colourbar
    (good for inspecting the structure of small residuals).
    """
    names = list(images)
    n = len(names)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False, constrained_layout=True)

    ref = np.asarray(images[residual_ref]) if residual_ref else None
    cmap = "RdBu_r" if ref is not None else "magma"

    # build the data each panel will show
    panels = {}
    for name in names:
        img = np.asarray(images[name])
        if ref is not None and img.shape != ref.shape:
            panels[name] = None  # shape mismatch, handled below
        else:
            panels[name] = (img - ref) if ref is not None else img

    # shared colour limits across all valid panels
    svmin = svmax = None
    if shared:
        valid = [d for d in panels.values() if d is not None]
        if ref is not None:
            svmax = max((np.abs(d).max() for d in valid), default=1.0)
            svmin = -svmax
        else:
            svmax = max((d.max() for d in valid), default=1.0)
            svmin = min((d.min() for d in valid), default=0.0)

    im = None
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        data = panels[name]
        if data is None:
            ax.set_title(f"{name}\n(shape != ref)")
            ax.axis("off")
            continue
        if shared:
            vmin, vmax = svmin, svmax
        elif ref is not None:
            vmax = np.abs(data).max(); vmin = -vmax
        else:
            vmin = vmax = None
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} - {residual_ref}" if ref is not None else name)
        ax.set_xticks([]); ax.set_yticks([])
        if not shared:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide any unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # one shared colourbar for the whole figure
    if shared and im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot a forward-model benchmark run.")
    p.add_argument("json", type=Path, help="Path to forward_model_*.json")
    p.add_argument("--npz", type=Path, default=None,
                   help="Model images .npz (default: same stem as JSON).")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output directory (default: alongside the JSON).")
    p.add_argument("--ref", default="lenstronomy",
                   help="Reference adapter for residual maps (default: lenstronomy).")
    args = p.parse_args()

    data, timed = _load(args.json)
    if not timed:
        raise SystemExit("No timed adapters found in JSON.")

    outdir = args.outdir or args.json.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.json.stem

    plot_timings(timed, data.get("config", {}), outdir / f"{stem}_timings.png")

    npz_path = args.npz or args.json.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path) as npz:
            images = {k: npz[k] for k in npz.files}
        _image_grid(images, outdir / f"{stem}_images.png",
                    "Model images", shared=False)
        _image_grid(images, outdir / f"{stem}_images_shared.png",
                    "Model images (shared colourbar)", shared=True)
        if args.ref in images:
            _image_grid(images, outdir / f"{stem}_residuals.png",
                        f"Residuals (adapter - {args.ref})",
                        residual_ref=args.ref, shared=False)
            _image_grid(images, outdir / f"{stem}_residuals_shared.png",
                        f"Residuals (adapter - {args.ref}, shared colourbar)",
                        residual_ref=args.ref, shared=True)
        else:
            print(f"note: reference '{args.ref}' not in images; skipping residuals")
    else:
        print(f"note: {npz_path} not found; skipping image plots "
              "(re-run the benchmark with --save-images)")


if __name__ == "__main__":
    main()
