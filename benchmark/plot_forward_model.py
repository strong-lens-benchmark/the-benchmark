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


def _image_grid(images: dict, out: Path, title: str, residual_ref: str | None = None) -> None:
    names = list(images)
    n = len(names)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    ref = np.asarray(images[residual_ref]) if residual_ref else None

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        img = np.asarray(images[name])
        if ref is not None:
            if img.shape != ref.shape:
                ax.set_title(f"{name}\n(shape {img.shape} != ref)")
                ax.axis("off")
                continue
            data = img - ref
            im = ax.imshow(data, origin="lower", cmap="RdBu_r",
                           vmin=-np.abs(data).max(), vmax=np.abs(data).max())
            ax.set_title(f"{name} - {residual_ref}")
        else:
            im = ax.imshow(img, origin="lower", cmap="magma")
            ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

    # hide any unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
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
        _image_grid(images, outdir / f"{stem}_images.png", "Model images")
        if args.ref in images:
            _image_grid(images, outdir / f"{stem}_residuals.png",
                        f"Residuals (adapter - {args.ref})", residual_ref=args.ref)
        else:
            print(f"note: reference '{args.ref}' not in images; skipping residuals")
    else:
        print(f"note: {npz_path} not found; skipping image plots "
              "(re-run the benchmark with --save-images)")


if __name__ == "__main__":
    main()
