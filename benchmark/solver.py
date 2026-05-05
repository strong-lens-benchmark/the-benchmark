from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.signal import fftconvolve

from .adapters import LensCodeAdapter
from .data import MockImage
from .params import BenchmarkParams, InversionConfig, SourceGridConfig


@dataclass(frozen=True)
class InversionResult:
    log_likelihood: float
    chi2: float
    reg_penalty: float
    reduced_chi2: float
    lens_light_amp: float
    source: np.ndarray
    source_model_image: np.ndarray
    lens_light_image: np.ndarray
    model_image: np.ndarray


def evaluate_model(
    adapter: LensCodeAdapter,
    mock: MockImage,
    params: BenchmarkParams,
    source_grid: SourceGridConfig,
    inversion: InversionConfig,
) -> InversionResult:
    beta_x, beta_y = adapter.ray_trace(mock.x_grid, mock.y_grid, params.lens)
    mapping = bilinear_source_mapping(beta_x, beta_y, source_grid)
    source_matrix = convolved_mapping_matrix(mapping, mock.image.shape, mock.psf_kernel)

    lens_light_unit = adapter.lens_light(mock.x_grid, mock.y_grid, params.lens_light)
    lens_light_unit = convolve_image(lens_light_unit, mock.psf_kernel).reshape(-1, 1)
    design_matrix = np.hstack([source_matrix, lens_light_unit])

    data_vec = mock.image.ravel()
    noise_vec = mock.noise_map.ravel()
    coeffs, reg_matrix = solve_regularized_linear_model(
        design_matrix=design_matrix,
        data_vec=data_vec,
        noise_vec=noise_vec,
        n_source=source_grid.nx * source_grid.ny,
        source_grid=source_grid,
        inversion=inversion,
    )

    model_vec = design_matrix @ coeffs
    residual = (data_vec - model_vec) / noise_vec
    chi2 = float(np.dot(residual, residual))
    source = coeffs[: source_grid.nx * source_grid.ny]
    lens_light_amp = float(coeffs[-1])
    reg_penalty = float(inversion.lambda_reg * np.dot(reg_matrix @ source, reg_matrix @ source))
    noise_norm = float(np.sum(np.log(2.0 * np.pi * noise_vec**2))) if inversion.include_noise_norm else 0.0
    log_likelihood = -0.5 * (chi2 + reg_penalty + noise_norm)
    dof = max(1, data_vec.size - coeffs.size)

    source_model = source_matrix @ source
    lens_light_model = (lens_light_unit[:, 0] * lens_light_amp).reshape(mock.image.shape)
    return InversionResult(
        log_likelihood=float(log_likelihood),
        chi2=chi2,
        reg_penalty=reg_penalty,
        reduced_chi2=float(chi2 / dof),
        lens_light_amp=lens_light_amp,
        source=source.reshape(source_grid.ny, source_grid.nx),
        source_model_image=source_model.reshape(mock.image.shape),
        lens_light_image=lens_light_model,
        model_image=model_vec.reshape(mock.image.shape),
    )


def bilinear_source_mapping(
    beta_x: np.ndarray, beta_y: np.ndarray, config: SourceGridConfig
) -> sparse.csr_matrix:
    n_image = beta_x.size
    n_source = config.nx * config.ny
    tx = (beta_x.ravel() - config.x_min) / (config.x_max - config.x_min) * (config.nx - 1)
    ty = (beta_y.ravel() - config.y_min) / (config.y_max - config.y_min) * (config.ny - 1)
    ix0 = np.floor(tx).astype(int)
    iy0 = np.floor(ty).astype(int)
    dx = tx - ix0
    dy = ty - iy0

    rows = []
    cols = []
    vals = []
    for ox, wx in ((0, 1.0 - dx), (1, dx)):
        for oy, wy in ((0, 1.0 - dy), (1, dy)):
            ix = ix0 + ox
            iy = iy0 + oy
            valid = (ix >= 0) & (ix < config.nx) & (iy >= 0) & (iy < config.ny)
            rows.append(np.nonzero(valid)[0])
            cols.append((iy[valid] * config.nx + ix[valid]).astype(int))
            vals.append((wx[valid] * wy[valid]).astype(float))

    if not rows:
        return sparse.csr_matrix((n_image, n_source), dtype=float)
    return sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_image, n_source),
    ).tocsr()


def convolved_mapping_matrix(
    mapping: sparse.csr_matrix, image_shape: tuple[int, int], psf_kernel: np.ndarray
) -> np.ndarray:
    dense = np.asarray(mapping.toarray(), dtype=float)
    for col in range(dense.shape[1]):
        dense[:, col] = convolve_image(dense[:, col].reshape(image_shape), psf_kernel).ravel()
    return dense


def convolve_image(image: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
    return fftconvolve(image, psf_kernel, mode="same")


def solve_regularized_linear_model(
    design_matrix: np.ndarray,
    data_vec: np.ndarray,
    noise_vec: np.ndarray,
    n_source: int,
    source_grid: SourceGridConfig,
    inversion: InversionConfig,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    weighted_matrix = design_matrix / noise_vec[:, None]
    weighted_data = data_vec / noise_vec
    lhs = weighted_matrix.T @ weighted_matrix
    rhs = weighted_matrix.T @ weighted_data

    reg_matrix = source_regularization_matrix(source_grid, inversion.regularization)
    reg_block = reg_matrix.T @ reg_matrix
    lhs[:n_source, :n_source] += inversion.lambda_reg * reg_block.toarray()

    try:
        coeffs = linalg.solve(lhs, rhs, assume_a="pos")
    except linalg.LinAlgError:
        coeffs = linalg.lstsq(lhs, rhs)[0]
    return np.asarray(coeffs, dtype=float), reg_matrix


def source_regularization_matrix(
    config: SourceGridConfig, regularization: str
) -> sparse.csr_matrix:
    n_source = config.nx * config.ny
    if regularization == "identity":
        return sparse.identity(n_source, format="csr", dtype=float)
    if regularization != "gradient":
        raise ValueError("regularization must be 'identity' or 'gradient'")

    rows = []
    cols = []
    vals = []
    row = 0
    for iy in range(config.ny):
        for ix in range(config.nx - 1):
            left = iy * config.nx + ix
            right = left + 1
            rows.extend([row, row])
            cols.extend([left, right])
            vals.extend([-1.0, 1.0])
            row += 1
    for iy in range(config.ny - 1):
        for ix in range(config.nx):
            lower = iy * config.nx + ix
            upper = lower + config.nx
            rows.extend([row, row])
            cols.extend([lower, upper])
            vals.extend([-1.0, 1.0])
            row += 1
    return sparse.coo_matrix((vals, (rows, cols)), shape=(row, n_source)).tocsr()
