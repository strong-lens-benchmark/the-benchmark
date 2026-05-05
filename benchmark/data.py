from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .params import BenchmarkParams, params_from_truth


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "analosis" / "analosis" / "results" / "datasets"


@dataclass(frozen=True)
class MockImage:
    dataset: str
    index: int
    image: np.ndarray
    noise_map: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    psf_kernel: np.ndarray
    truth_row: dict[str, float]
    truth_params: BenchmarkParams
    source_truth: dict | None
    kwargs_data: dict
    kwargs_psf: dict
    kwargs_numerics: dict


def load_mock(dataset: str = "test_simple", index: int = 0) -> MockImage:
    with (DATASET_ROOT / f"{dataset}_hyperdata.pickle").open("rb") as handle:
        hyperdata = pickle.load(handle)
    truth = pd.read_csv(DATASET_ROOT / f"{dataset}_input_kwargs.csv")

    kwargs_data, kwargs_psf, kwargs_numerics = hyperdata[index]
    image = np.asarray(kwargs_data["image_data"], dtype=float)
    x_grid, y_grid = make_image_grid(kwargs_data, image.shape)
    psf_kernel = make_psf_kernel(kwargs_psf)
    noise_map = make_noise_map(image, kwargs_data)
    truth_row = truth.iloc[index].to_dict()
    source_truth = load_source_truth(dataset, index)

    return MockImage(
        dataset=dataset,
        index=index,
        image=image,
        noise_map=noise_map,
        x_grid=x_grid,
        y_grid=y_grid,
        psf_kernel=psf_kernel,
        truth_row=truth_row,
        truth_params=params_from_truth(truth_row),
        source_truth=source_truth,
        kwargs_data=kwargs_data,
        kwargs_psf=kwargs_psf,
        kwargs_numerics=kwargs_numerics,
    )


def make_image_grid(kwargs_data: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype=float)
    transform = np.asarray(kwargs_data["transform_pix2angle"], dtype=float)
    x = (
        float(kwargs_data["ra_at_xy_0"])
        + transform[0, 0] * cols
        + transform[0, 1] * rows
    )
    y = (
        float(kwargs_data["dec_at_xy_0"])
        + transform[1, 0] * cols
        + transform[1, 1] * rows
    )
    return x, y


def make_noise_map(image: np.ndarray, kwargs_data: dict) -> np.ndarray:
    background_rms = float(kwargs_data["background_rms"])
    exposure_time = float(kwargs_data.get("exposure_time", 0.0))
    if exposure_time <= 0.0:
        return np.full_like(image, background_rms, dtype=float)
    poisson = np.clip(image, 0.0, None) / exposure_time
    return np.sqrt(background_rms**2 + poisson)


def make_psf_kernel(kwargs_psf: dict) -> np.ndarray:
    psf_type = kwargs_psf.get("psf_type", "").upper()
    if psf_type != "GAUSSIAN":
        raise ValueError(f"Only GAUSSIAN PSFs are supported by the shared driver, got {psf_type!r}")

    fwhm = float(kwargs_psf["fwhm"])
    pixel_size = float(kwargs_psf["pixel_size"])
    truncation = float(kwargs_psf.get("truncation", 3.0))
    sigma_pix = fwhm / 2.354820045 / pixel_size
    radius = max(1, int(np.ceil(truncation * sigma_pix)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma_pix**2)
    return kernel / np.sum(kernel)


def load_source_truth(dataset: str, index: int) -> dict | None:
    path = DATASET_ROOT / f"{dataset}_source_truth.pickle"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        source_truth = pickle.load(handle)
    matches = [entry for entry in source_truth if int(entry.get("image_index", -1)) == index]
    return matches[0] if matches else None
