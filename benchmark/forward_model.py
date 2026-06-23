"""Forward model benchmark adapters and timing utilities.

Each adapter pre-builds all persistent objects at construction time so that
__call__() contains only the computation to be timed: ray-tracing, light
evaluation, and PSF convolution.

Each code renders the image with its OWN native image-simulation API and its
OWN PSF convolution, built from a Gaussian of the configured FWHM -- the
out-of-the-box path a user of that code would take:
  - lenstronomy / jaxtronomy : ImSim.image_model.ImageModel.image()
  - herculens                : LensImage.model()
  - TinyLensGpu              : LensSimulator.simulate()
  - autolens                 : SimulatorImaging.via_tracer_from()
(jaxtronomy's GAUSSIAN psf_type is broken out of the box, so it is given the
Gaussian kernel as a PIXEL PSF and still convolves it internally.)

JAX-backed adapters (jaxtronomy, herculens, TinyLensGpu, autolens):
  - Call the adapter once before timing to trigger JIT compilation.
  - Use jax.block_until_ready(result) before stopping the clock so that
    asynchronous GPU dispatch does not undercount execution time.

Parameter / output conventions
------------------------------
All adapters use the same physical parameters. Two convention differences are
reconciled in the adapters:

1. herculens's SERSIC_ELLIPSE parameterises R_sersic as the semi-major axis
   (R_sersic_ma) rather than the product-average radius used by the others:
       R_sersic_ma = R_sersic_pa / sqrt(q),  q = (1 - e) / (1 + e)

2. Image normalisation. lenstronomy/jaxtronomy (ImageModel) and herculens
   (LensImage.model) all return flux-per-pixel (surface brightness * pixel
   area).  TinyLensGpu.simulate() returns surface brightness, so its adapter
   multiplies by the pixel area to match the shared flux-per-pixel convention.

Both are applied at construction/output time and require no modification to the
underlying codes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageConfig:
    numpix: int = 100
    dpix: float = 0.05       # arcsec / pixel
    psf_fwhm: float = 0.1    # arcsec


@dataclass(frozen=True)
class LensConfig:
    theta_E: float = 1.0
    gamma: float = 2.1
    e1: float = 0.1
    e2: float = 0.05
    gamma1: float = 0.02
    gamma2: float = 0.01
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass(frozen=True)
class SersicConfig:
    amp: float
    R_sersic: float
    n_sersic: float
    e1: float
    e2: float
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass(frozen=True)
class ForwardModelConfig:
    image: ImageConfig = field(default_factory=ImageConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    lens_light: SersicConfig = field(default_factory=lambda: SersicConfig(
        amp=1.0, R_sersic=1.0, n_sersic=4.0, e1=0.1, e2=0.05,
    ))
    source: SersicConfig = field(default_factory=lambda: SersicConfig(
        amp=1.0, R_sersic=0.2, n_sersic=1.0, e1=0.05, e2=0.02,
        center_x=0.05, center_y=0.05,
    ))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _herculens_R_sersic_ma(s: SersicConfig) -> float:
    """Convert product-average R_sersic to herculens semi-major axis convention."""
    e = float(np.hypot(s.e1, s.e2))
    q = (1.0 - e) / (1.0 + e)
    return s.R_sersic / float(np.sqrt(q))


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

class ForwardModelAdapter(Protocol):
    name: str
    is_jax: bool

    def __call__(self) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# lenstronomy  (CPU reference)
# ---------------------------------------------------------------------------

class LenstronomyAdapter:
    name = "lenstronomy"
    is_jax = False

    def __init__(self, cfg: ForwardModelConfig) -> None:
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.Util import simulation_util as sim_util

        numpix = cfg.image.numpix
        dpix = cfg.image.dpix

        # Native image rendering with lenstronomy's own Gaussian PSF convolution.
        kwargs_data = sim_util.data_configure_simple(numpix, dpix)
        data = ImageData(**kwargs_data)
        psf = PSF(psf_type="GAUSSIAN", fwhm=cfg.image.psf_fwhm, pixel_size=dpix)

        self._image_model = ImageModel(
            data_class=data,
            psf_class=psf,
            lens_model_class=LensModel(["EPL", "SHEAR"]),
            source_model_class=LightModel(["SERSIC_ELLIPSE"]),
            lens_light_model_class=LightModel(["SERSIC_ELLIPSE"]),
            kwargs_numerics={"supersampling_factor": 1},
        )

        lens = cfg.lens
        self._kwargs_lens = [
            {
                "theta_E": lens.theta_E, "gamma": lens.gamma,
                "e1": lens.e1, "e2": lens.e2,
                "center_x": lens.center_x, "center_y": lens.center_y,
            },
            {"gamma1": lens.gamma1, "gamma2": lens.gamma2},
        ]
        src = cfg.source
        self._kwargs_source = [{
            "amp": src.amp, "R_sersic": src.R_sersic, "n_sersic": src.n_sersic,
            "e1": src.e1, "e2": src.e2,
            "center_x": src.center_x, "center_y": src.center_y,
        }]
        ll = cfg.lens_light
        self._kwargs_ll = [{
            "amp": ll.amp, "R_sersic": ll.R_sersic, "n_sersic": ll.n_sersic,
            "e1": ll.e1, "e2": ll.e2,
            "center_x": ll.center_x, "center_y": ll.center_y,
        }]

    def __call__(self) -> np.ndarray:
        return np.asarray(self._image_model.image(
            self._kwargs_lens, self._kwargs_source, self._kwargs_ll
        ))


# ---------------------------------------------------------------------------
# JAXtronomy  (JAX / GPU)
# ---------------------------------------------------------------------------

class JAXtronomyAdapter:
    name = "jaxtronomy"
    is_jax = True

    def __init__(self, cfg: ForwardModelConfig) -> None:
        import jax
        jax.config.update("jax_enable_x64", True)
        from jaxtronomy.ImSim.image_model import ImageModel
        from jaxtronomy.Data.imaging_data import ImageData
        from jaxtronomy.LensModel.lens_model import LensModel
        from jaxtronomy.LightModel.light_model import LightModel
        # jaxtronomy reuses lenstronomy's PSF and data-configuration utilities.
        from lenstronomy.Data.psf import PSF
        from lenstronomy.Util import simulation_util as sim_util

        numpix = cfg.image.numpix
        dpix = cfg.image.dpix

        # Native image rendering with jaxtronomy's own Gaussian PSF convolution.
        # jaxtronomy's ImageData requires noise/exposure parameters even though
        # only .image() (no noise) is used here; the dummy values do not affect
        # the rendered image.
        kwargs_data = sim_util.data_configure_simple(
            numpix, dpix, exposure_time=1.0, background_rms=1.0,
        )
        data = ImageData(**kwargs_data)
        psf = PSF(psf_type="GAUSSIAN", fwhm=cfg.image.psf_fwhm, pixel_size=dpix)

        self._image_model = ImageModel(
            data_class=data,
            psf_class=psf,
            lens_model_class=LensModel(["EPL", "SHEAR"]),
            source_model_class=LightModel(["SERSIC_ELLIPSE"]),
            lens_light_model_class=LightModel(["SERSIC_ELLIPSE"]),
            kwargs_numerics={"supersampling_factor": 1},
        )

        lens = cfg.lens
        self._kwargs_lens = [
            {
                "theta_E": lens.theta_E, "gamma": lens.gamma,
                "e1": lens.e1, "e2": lens.e2,
                "center_x": lens.center_x, "center_y": lens.center_y,
            },
            {"gamma1": lens.gamma1, "gamma2": lens.gamma2},
        ]
        src = cfg.source
        self._kwargs_source = [{
            "amp": src.amp, "R_sersic": src.R_sersic, "n_sersic": src.n_sersic,
            "e1": src.e1, "e2": src.e2,
            "center_x": src.center_x, "center_y": src.center_y,
        }]
        ll = cfg.lens_light
        self._kwargs_ll = [{
            "amp": ll.amp, "R_sersic": ll.R_sersic, "n_sersic": ll.n_sersic,
            "e1": ll.e1, "e2": ll.e2,
            "center_x": ll.center_x, "center_y": ll.center_y,
        }]

    def __call__(self):
        # point_source_add=False: there are no point sources, and jaxtronomy's
        # image() otherwise enters its point-source rendering path unconditionally.
        return self._image_model.image(
            self._kwargs_lens, self._kwargs_source, self._kwargs_ll,
            point_source_add=False,
        )


# ---------------------------------------------------------------------------
# herculens  (JAX / GPU)
# ---------------------------------------------------------------------------

class HerculensAdapter:
    name = "herculens"
    is_jax = True

    def __init__(self, cfg: ForwardModelConfig) -> None:
        import sys, types
        import jax
        jax.config.update("jax_enable_x64", True)

        # herculens/__init__.py unconditionally imports mass_model_multiplane,
        # which requires jax_cosmo (only needed for time-delay / multiplane models).
        # Stub it out so the import succeeds without that optional dependency.
        if "jax_cosmo" not in sys.modules:
            sys.modules["jax_cosmo"] = types.ModuleType("jax_cosmo")

        from herculens.Coordinates.pixel_grid import PixelGrid
        from herculens.Instrument.psf import PSF
        from herculens.LensImage.lens_image import LensImage
        from herculens.LightModel.light_model import LightModel
        from herculens.MassModel.mass_model import MassModel

        numpix = cfg.image.numpix
        dpix = cfg.image.dpix

        ra_at_xy_0 = -(numpix / 2 - 0.5) * dpix
        dec_at_xy_0 = -(numpix / 2 - 0.5) * dpix
        grid = PixelGrid(
            nx=numpix, ny=numpix,
            transform_pix2angle=np.array([[dpix, 0.0], [0.0, dpix]]),
            ra_at_xy_0=ra_at_xy_0,
            dec_at_xy_0=dec_at_xy_0,
        )
        # Native Gaussian PSF (herculens builds and convolves it internally).
        psf = PSF(psf_type="GAUSSIAN", fwhm=cfg.image.psf_fwhm, pixel_size=dpix)

        self._lens_image = LensImage(
            grid_class=grid,
            psf_class=psf,
            lens_mass_model_class=MassModel(["EPL", "SHEAR"]),
            source_model_class=LightModel(["SERSIC_ELLIPSE"]),
            lens_light_model_class=LightModel(["SERSIC_ELLIPSE"]),
        )

        lens = cfg.lens
        self._kwargs_lens = [
            {
                "theta_E": lens.theta_E, "gamma": lens.gamma,
                "e1": lens.e1, "e2": lens.e2,
                "center_x": lens.center_x, "center_y": lens.center_y,
            },
            {"gamma1": lens.gamma1, "gamma2": lens.gamma2, "ra_0": 0.0, "dec_0": 0.0},
        ]
        src = cfg.source
        self._kwargs_source = [{
            "amp": src.amp,
            "R_sersic": _herculens_R_sersic_ma(src),
            "n_sersic": src.n_sersic,
            "e1": src.e1, "e2": src.e2,
            "center_x": src.center_x, "center_y": src.center_y,
        }]
        ll = cfg.lens_light
        self._kwargs_ll = [{
            "amp": ll.amp,
            "R_sersic": _herculens_R_sersic_ma(ll),
            "n_sersic": ll.n_sersic,
            "e1": ll.e1, "e2": ll.e2,
            "center_x": ll.center_x, "center_y": ll.center_y,
        }]

    def __call__(self):
        # herculens's LensImage.model() returns flux-per-pixel (re_size_convolve
        # ends with image_conv * pixel_width**2).  lenstronomy's and jaxtronomy's
        # ImageModel.image() use the same flux-per-pixel convention, so herculens
        # needs no rescaling to be directly comparable.
        return self._lens_image.model(
            kwargs_lens=self._kwargs_lens,
            kwargs_source=self._kwargs_source,
            kwargs_lens_light=self._kwargs_ll,
        )


# ---------------------------------------------------------------------------
# TinyLensGpu  (JAX / GPU)
# ---------------------------------------------------------------------------

class TinyLensGpuAdapter:
    name = "tinylensgpu"
    is_jax = True

    def __init__(self, cfg: ForwardModelConfig) -> None:
        import jax
        jax.config.update("jax_enable_x64", True)

        from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig, make_grid_2d
        from TinyLensGpu.ForwardSimulation.LensImage.parametric import LensSimulator
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light.gaussian import GaussianEllipse
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.epl import EPL
        from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass.shear import Shear
        from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel

        lens = cfg.lens
        epl = EPL(
            theta_E=lens.theta_E, gamma=lens.gamma,
            e1=lens.e1, e2=lens.e2,
            center_x=lens.center_x, center_y=lens.center_y,
        )
        for p in [epl.theta_E, epl.gamma, epl.e1, epl.e2, epl.center_x, epl.center_y]:
            p.to_static()

        shear = Shear(gamma1=lens.gamma1, gamma2=lens.gamma2)
        for p in [shear.gamma1, shear.gamma2]:
            p.to_static()

        src = cfg.source
        source = SersicEllipse(
            R_sersic=src.R_sersic, n_sersic=src.n_sersic,
            e1=src.e1, e2=src.e2,
            center_x=src.center_x, center_y=src.center_y,
            Ie=src.amp,
        )
        src_params = [source.R_sersic, source.n_sersic, source.e1, source.e2,
                      source.center_x, source.center_y]
        if hasattr(source, "Ie") and hasattr(source.Ie, "to_static"):
            src_params.append(source.Ie)
        for p in src_params:
            p.to_static()

        ll = cfg.lens_light
        lens_light = SersicEllipse(
            R_sersic=ll.R_sersic, n_sersic=ll.n_sersic,
            e1=ll.e1, e2=ll.e2,
            center_x=ll.center_x, center_y=ll.center_y,
            Ie=ll.amp,
        )
        ll_params = [lens_light.R_sersic, lens_light.n_sersic, lens_light.e1, lens_light.e2,
                     lens_light.center_x, lens_light.center_y]
        if hasattr(lens_light, "Ie") and hasattr(lens_light.Ie, "to_static"):
            ll_params.append(lens_light.Ie)
        for p in ll_params:
            p.to_static()

        phys_model = PhysicalModel(
            lens_mass=[epl, shear],
            source_light=[source],
            lens_light=[lens_light],
        )
        # Native Gaussian PSF kernel built with TinyLensGpu's own GaussianEllipse
        # profile (matching its demo sim_data.py), convolved internally by simulate().
        sigma = cfg.image.psf_fwhm / 2.3548200450309493  # FWHM -> sigma
        psf_radius = max(1, int(np.ceil(3.0 * sigma / cfg.image.dpix)))
        psf_npix = 2 * psf_radius + 1
        x_psf, y_psf = make_grid_2d(psf_npix, cfg.image.dpix)
        psf_kernel = GaussianEllipse(
            flux=1.0, sigma=sigma, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
        ).light(x=x_psf, y=y_psf)
        psf_kernel = np.asarray(psf_kernel)
        psf_kernel = psf_kernel / psf_kernel.sum()

        sim_config = SimulatorConfig(
            dpix=cfg.image.dpix,
            npix=cfg.image.numpix,
            psf_kernel=psf_kernel,
            nsub=1,
        )
        self._simulator = LensSimulator(phys_model=phys_model, sim_config=sim_config)
        self._pixel_area = cfg.image.dpix ** 2

    def __call__(self):
        # TinyLensGpu.simulate() returns surface brightness (no pixel-area
        # weighting), whereas lenstronomy/jaxtronomy/herculens return
        # flux-per-pixel.  Multiply by the pixel area to match that convention.
        return self._simulator.simulate(use_linear=False) * self._pixel_area


# ---------------------------------------------------------------------------
# autolens  (JAX / GPU)
# ---------------------------------------------------------------------------

class AutolensAdapter:
    name = "autolens"
    is_jax = True

    def __init__(self, cfg: ForwardModelConfig) -> None:
        import jax
        jax.config.update("jax_enable_x64", True)
        import autoarray as aa
        import autolens as al

        numpix = cfg.image.numpix
        dpix = cfg.image.dpix
        self._numpix = numpix

        self._grid = aa.Grid2D.uniform(shape_native=(numpix, numpix), pixel_scales=dpix)

        # Native Gaussian PSF built with autolens's own Kernel2D.from_gaussian.
        sigma = cfg.image.psf_fwhm / 2.3548200450309493  # FWHM -> sigma
        psf_radius = max(1, int(np.ceil(3.0 * sigma / dpix)))
        psf_npix = 2 * psf_radius + 1
        psf = aa.Kernel2D.from_gaussian(
            shape_native=(psf_npix, psf_npix), pixel_scales=dpix,
            sigma=sigma, normalize=True,
        )
        self._simulator = al.SimulatorImaging(psf=psf, add_poisson_noise=False)

        lens = cfg.lens
        ll = cfg.lens_light
        src = cfg.source

        # autolens ell_comps convention: (ell_comps_0, ell_comps_1) = (e2, e1)
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.PowerLaw(
                centre=(lens.center_y, lens.center_x),
                ell_comps=(lens.e2, lens.e1),
                einstein_radius=lens.theta_E,
                slope=lens.gamma,
            ),
            shear=al.mp.ExternalShear(
                gamma_1=lens.gamma1,
                gamma_2=lens.gamma2,
            ),
            light=al.lp.Sersic(
                centre=(ll.center_y, ll.center_x),
                ell_comps=(ll.e2, ll.e1),
                effective_radius=ll.R_sersic,
                sersic_index=ll.n_sersic,
                intensity=ll.amp,
            ),
        )
        source_galaxy = al.Galaxy(
            redshift=2.0,
            light=al.lp.Sersic(
                centre=(src.center_y, src.center_x),
                ell_comps=(src.e2, src.e1),
                effective_radius=src.R_sersic,
                sersic_index=src.n_sersic,
                intensity=src.amp,
            ),
        )
        self._tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    def __call__(self):
        import jax.numpy as jnp
        # SimulatorImaging.via_tracer_from renders via autolens's own imaging
        # pipeline (its Convolver PSF + over-sampling) and returns the image in
        # the data array's orientation, so no manual reshape/flip is needed.
        dataset = self._simulator.via_tracer_from(
            tracer=self._tracer, grid=self._grid, xp=jnp,
        )
        return jnp.asarray(dataset.data.native)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, type] = {
    "lenstronomy": LenstronomyAdapter,
    "jaxtronomy": JAXtronomyAdapter,
    "herculens": HerculensAdapter,
    "tinylensgpu": TinyLensGpuAdapter,
    "autolens": AutolensAdapter,
}


def get_adapter(name: str, cfg: ForwardModelConfig) -> ForwardModelAdapter:
    key = name.lower()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown adapter {name!r}; valid: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[key](cfg)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_adapter(
    adapter: ForwardModelAdapter,
    n_warmup: int = 3,
    n_repeat: int = 20,
) -> dict:
    """Time one forward model adapter.

    Warmup calls trigger JIT compilation for JAX-backed adapters.
    jax.block_until_ready() ensures GPU dispatch completes before the clock stops.
    """
    if adapter.is_jax:
        import jax

    for _ in range(n_warmup):
        result = adapter()
        if adapter.is_jax:
            jax.block_until_ready(result)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = adapter()
        if adapter.is_jax:
            jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "std_s": float(arr.std()),
        "n_warmup": n_warmup,
        "n_repeat": n_repeat,
        "times_s": arr.tolist(),
    }
