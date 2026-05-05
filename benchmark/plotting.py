from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import MockImage
from .solver import InversionResult


def write_diagnostics(
    output_dir: str | Path,
    adapter_name: str,
    mock: MockImage,
    result: InversionResult,
    source_extent: tuple[float, float, float, float],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_arrays(output / f"{adapter_name}_diagnostics.npz", mock, result, source_extent)
    save_figure(output / f"{adapter_name}_diagnostics.png", adapter_name, mock, result, source_extent)


_INPUT_SOURCE_RESOLUTION = 200


def save_arrays(
    path: Path,
    mock: MockImage,
    result: InversionResult,
    source_extent: tuple[float, float, float, float],
) -> None:
    input_source = render_input_source(
        mock, (_INPUT_SOURCE_RESOLUTION, _INPUT_SOURCE_RESOLUTION), source_extent
    )
    np.savez_compressed(
        path,
        image=mock.image,
        noise_map=mock.noise_map,
        model=result.model_image,
        residual=mock.image - result.model_image,
        normalized_residual=(mock.image - result.model_image) / mock.noise_map,
        source_model_image=result.source_model_image,
        lens_light_image=result.lens_light_image,
        input_source=input_source,
        source=result.source,
    )


def save_figure(
    path: Path,
    adapter_name: str,
    mock: MockImage,
    result: InversionResult,
    source_extent: tuple[float, float, float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = mock.image - result.model_image
    normalized_residual = residual / mock.noise_map
    image_vmin, image_vmax = robust_limits(mock.image)
    input_source = render_input_source(
        mock, (_INPUT_SOURCE_RESOLUTION, _INPUT_SOURCE_RESOLUTION), source_extent
    )
    input_source_vmin, input_source_vmax = robust_limits(input_source.ravel())
    recovered_vmin, recovered_vmax = robust_limits(result.source.ravel())

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), constrained_layout=True)
    panels = [
        ("data", mock.image, image_vmin, image_vmax, "image"),
        ("model", result.model_image, image_vmin, image_vmax, "image"),
        ("residual / noise", normalized_residual, -5.0, 5.0, "image"),
        ("input source", input_source, input_source_vmin, input_source_vmax, "source"),
        ("recovered source", result.source, recovered_vmin, recovered_vmax, "source"),
    ]

    for ax, (title, image, vmin, vmax, kind) in zip(axes, panels, strict=True):
        extent = image_extent(mock) if kind == "image" else source_extent
        im = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="magma" if title not in {"residual", "residual / noise"} else "coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest" if kind == "source" else "bicubic",
        )
        ax.set_title(title)
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")
        if kind == "image":
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{adapter_name}: reduced chi2={result.reduced_chi2:.3f}, "
        f"lens light amp={result.lens_light_amp:.3g}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def image_extent(mock: MockImage) -> tuple[float, float, float, float]:
    return (
        float(np.min(mock.x_grid)),
        float(np.max(mock.x_grid)),
        float(np.min(mock.y_grid)),
        float(np.max(mock.y_grid)),
    )


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [1.0, 99.5])
    if np.isclose(vmin, vmax):
        delta = max(abs(float(vmin)), 1.0) * 1.0e-3
        return float(vmin - delta), float(vmax + delta)
    return float(vmin), float(vmax)


def symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(finite), 99.0))
    vmax = vmax if vmax > 0.0 else 1.0
    return -vmax, vmax


def render_input_source(
    mock: MockImage,
    shape: tuple[int, int],
    source_extent: tuple[float, float, float, float],
) -> np.ndarray:
    if mock.source_truth is None:
        return np.full(shape, np.nan)

    try:
        from lenstronomy.LightModel.light_model import LightModel
    except ImportError:
        return np.full(shape, np.nan)

    x_min, x_max, y_min, y_max = source_extent
    ny, nx = shape
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    model_list = mock.source_truth["source_light_model_list"]
    kwargs_source = mock.source_truth["kwargs_source_amp"]
    return np.asarray(LightModel(model_list).surface_brightness(xx, yy, kwargs_source), dtype=float)

