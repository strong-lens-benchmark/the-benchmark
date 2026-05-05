from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .params import EPLShearParams, SersicLightParams


class LensCodeAdapter(Protocol):
    name: str

    def ray_trace(
        self, x: np.ndarray, y: np.ndarray, params: EPLShearParams
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def lens_light(
        self, x: np.ndarray, y: np.ndarray, params: SersicLightParams
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class LenstronomyAdapter:
    name: str = "lenstronomy"

    def ray_trace(self, x: np.ndarray, y: np.ndarray, params: EPLShearParams):
        from lenstronomy.LensModel.lens_model import LensModel

        lens_model = LensModel(["EPL", "SHEAR"])
        kwargs_lens = [
            {
                "theta_E": params.theta_E,
                "gamma": params.gamma,
                "e1": params.e1,
                "e2": params.e2,
                "center_x": params.center_x,
                "center_y": params.center_y,
            },
            {"gamma1": params.gamma1, "gamma2": params.gamma2},
        ]
        bx, by = lens_model.ray_shooting(x, y, kwargs_lens)
        return np.asarray(bx), np.asarray(by)

    def lens_light(self, x: np.ndarray, y: np.ndarray, params: SersicLightParams):
        from lenstronomy.LightModel.light_model import LightModel

        light_model = LightModel(["SERSIC_ELLIPSE"])
        image = light_model.surface_brightness(
            x,
            y,
            [
                {
                    "amp": 1.0,
                    "R_sersic": params.R_sersic,
                    "n_sersic": params.n_sersic,
                    "e1": params.e1,
                    "e2": params.e2,
                    "center_x": params.center_x,
                    "center_y": params.center_y,
                }
            ],
        )
        return np.asarray(image)


@dataclass(frozen=True)
class JAXtronomyAdapter:
    name: str = "jaxtronomy"

    def ray_trace(self, x: np.ndarray, y: np.ndarray, params: EPLShearParams):
        from jaxtronomy.LensModel.lens_model import LensModel

        lens_model = LensModel(["EPL", "SHEAR"])
        kwargs_lens = [
            {
                "theta_E": params.theta_E,
                "gamma": params.gamma,
                "e1": params.e1,
                "e2": params.e2,
                "center_x": params.center_x,
                "center_y": params.center_y,
            },
            {"gamma1": params.gamma1, "gamma2": params.gamma2},
        ]
        bx, by = lens_model.ray_shooting(x, y, kwargs_lens)
        return np.asarray(bx), np.asarray(by)

    def lens_light(self, x: np.ndarray, y: np.ndarray, params: SersicLightParams):
        from jaxtronomy.LightModel.light_model import LightModel

        light_model = LightModel(["SERSIC_ELLIPSE"])
        image = light_model.surface_brightness(
            x,
            y,
            [
                {
                    "amp": 1.0,
                    "R_sersic": params.R_sersic,
                    "n_sersic": params.n_sersic,
                    "e1": params.e1,
                    "e2": params.e2,
                    "center_x": params.center_x,
                    "center_y": params.center_y,
                }
            ],
        )
        return np.asarray(image)


@dataclass(frozen=True)
class HerculensAdapter:
    name: str = "herculens"

    def ray_trace(self, x: np.ndarray, y: np.ndarray, params: EPLShearParams):
        from herculens.MassModel.mass_model import MassModel

        lens_model = MassModel(["EPL", "SHEAR"])
        kwargs_lens = [
            {
                "theta_E": params.theta_E,
                "gamma": params.gamma,
                "e1": params.e1,
                "e2": params.e2,
                "center_x": params.center_x,
                "center_y": params.center_y,
            },
            {"gamma1": params.gamma1, "gamma2": params.gamma2, "ra_0": 0.0, "dec_0": 0.0},
        ]
        bx, by = lens_model.ray_shooting(x, y, kwargs_lens)
        return np.asarray(bx), np.asarray(by)

    def lens_light(self, x: np.ndarray, y: np.ndarray, params: SersicLightParams):
        from herculens.LightModel.light_model import LightModel

        light_model = LightModel(["SERSIC_ELLIPSE"])
        image = light_model.surface_brightness(
            x,
            y,
            [
                {
                    "amp": 1.0,
                    "R_sersic": params.R_sersic,
                    "n_sersic": params.n_sersic,
                    "e1": params.e1,
                    "e2": params.e2,
                    "center_x": params.center_x,
                    "center_y": params.center_y,
                }
            ],
        )
        return np.asarray(image)


@dataclass(frozen=True)
class TinyLensGPUAdapter:
    name: str = "tinylensgpu"

    def ray_trace(self, x: np.ndarray, y: np.ndarray, params: EPLShearParams):
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.epl import EPL
        from TinyLensGpu.PhysicalModel import Shear

        epl = EPL(
            theta_E=params.theta_E,
            gamma=params.gamma,
            e1=params.e1,
            e2=params.e2,
            center_x=params.center_x,
            center_y=params.center_y,
        )
        shear = Shear(gamma1=params.gamma1, gamma2=params.gamma2)
        ax_epl, ay_epl = epl.deriv(x, y)
        ax_shear, ay_shear = shear.deriv(x, y)
        return np.asarray(x - ax_epl - ax_shear), np.asarray(y - ay_epl - ay_shear)

    def lens_light(self, x: np.ndarray, y: np.ndarray, params: SersicLightParams):
        from TinyLensGpu.PhysicalModel import SersicEllipse

        light = SersicEllipse(
            R_sersic=params.R_sersic,
            n_sersic=params.n_sersic,
            e1=params.e1,
            e2=params.e2,
            center_x=params.center_x,
            center_y=params.center_y,
            Ie=1.0,
        )
        return np.asarray(light.light(x, y))


@dataclass(frozen=True)
class PyAutoLensAdapter:
    name: str = "autolens"

    def ray_trace(self, x: np.ndarray, y: np.ndarray, params: EPLShearParams):
        import autolens as al

        grid = _autolens_grid(x, y)
        power_law = al.mp.PowerLaw(
            centre=(params.center_y, params.center_x),
            ell_comps=(params.e2, params.e1),
            einstein_radius=params.theta_E,
            slope=params.gamma,
        )
        shear = al.mp.ExternalShear(gamma_1=params.gamma1, gamma_2=params.gamma2)
        defl_pl = np.asarray(power_law.deflections_yx_2d_from(grid=grid))
        defl_sh = np.asarray(shear.deflections_yx_2d_from(grid=grid))
        alpha_y = (defl_pl[:, 0] + defl_sh[:, 0]).reshape(x.shape)
        alpha_x = (defl_pl[:, 1] + defl_sh[:, 1]).reshape(x.shape)
        return x - alpha_x, y - alpha_y

    def lens_light(self, x: np.ndarray, y: np.ndarray, params: SersicLightParams):
        import autolens as al
        from autoarray.structures.grids.irregular_2d import Grid2DIrregular

        # Grid2DIrregular skips autolens's @aa.over_sample decorator, giving
        # point-evaluation at pixel centres to match lenstronomy's convention.
        # Grid2D.no_mask() would use pixel-averaged sub-grids computed from the
        # mask (with y-flipped coordinates), producing a systematically wrong image.
        values = np.stack([y.ravel(), x.ravel()], axis=-1)
        grid = Grid2DIrregular(values=values)
        light = al.lp.Sersic(
            centre=(params.center_y, params.center_x),
            ell_comps=(params.e2, params.e1),
            intensity=1.0,
            effective_radius=params.R_sersic,
            sersic_index=params.n_sersic,
        )
        return np.asarray(light.image_2d_from(grid=grid)).reshape(x.shape)


def _autolens_grid(x: np.ndarray, y: np.ndarray):
    import autolens as al

    values = np.stack([y.ravel(), x.ravel()], axis=-1)
    return al.Grid2D.no_mask(
        values=values,
        pixel_scales=float(abs(x[0, 1] - x[0, 0])),
        shape_native=x.shape,
    )


ADAPTERS = {
    "lenstronomy": LenstronomyAdapter,
    "jaxtronomy": JAXtronomyAdapter,
    "herculens": HerculensAdapter,
    "tinylensgpu": TinyLensGPUAdapter,
    "tiny": TinyLensGPUAdapter,
    "autolens": PyAutoLensAdapter,
    "pyautolens": PyAutoLensAdapter,
}


def get_adapter(name: str) -> LensCodeAdapter:
    key = name.lower()
    if key not in ADAPTERS:
        valid = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown adapter {name!r}; valid adapters are {valid}")
    return ADAPTERS[key]()
