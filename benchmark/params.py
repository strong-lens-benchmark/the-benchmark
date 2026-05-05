from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EPLShearParams:
    theta_E: float
    gamma: float
    e1: float
    e2: float
    gamma1: float
    gamma2: float
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass(frozen=True)
class SersicLightParams:
    R_sersic: float
    n_sersic: float
    e1: float
    e2: float
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass(frozen=True)
class SourceGridConfig:
    nx: int = 25
    ny: int = 25
    x_min: float = -1.2
    x_max: float = 1.2
    y_min: float = -1.2
    y_max: float = 1.2


@dataclass(frozen=True)
class InversionConfig:
    lambda_reg: float = 1.0e-2
    regularization: str = "gradient"
    include_noise_norm: bool = True


@dataclass(frozen=True)
class BenchmarkParams:
    lens: EPLShearParams
    lens_light: SersicLightParams


def params_from_truth(row: dict[str, float]) -> BenchmarkParams:
    return BenchmarkParams(
        lens=EPLShearParams(
            theta_E=float(row["theta_E_epl"]),
            gamma=float(row["gamma_epl"]),
            e1=float(row["e1_epl"]),
            e2=float(row["e2_epl"]),
            gamma1=float(row["gamma1"]),
            gamma2=float(row["gamma2"]),
            center_x=float(row.get("center_x_epl", 0.0)),
            center_y=float(row.get("center_y_epl", 0.0)),
        ),
        lens_light=SersicLightParams(
            R_sersic=float(row["R_sersic_ll"]),
            n_sersic=float(row["n_sersic_ll"]),
            e1=float(row["e1_ll"]),
            e2=float(row["e2_ll"]),
            center_x=float(row.get("x_ll", 0.0)),
            center_y=float(row.get("y_ll", 0.0)),
        ),
    )
