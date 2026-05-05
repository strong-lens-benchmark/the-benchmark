# the-benchmark

## Single-mock shared inversion smoke test

Run the current same-model smoke test from this directory with the project
environment:

```bash
.venv/bin/python -m benchmark.run_single_mock \
  --mode evaluate \
  --nx-src 25 \
  --ny-src 25 \
  --adapters lenstronomy,jaxtronomy,herculens,tinylensgpu,autolens \
  --output results/single_mock_evaluate.json
```

This loads `analosis` dataset `test_simple`, fixes the EPL and lens-light
centres to the catalogue values, ray-traces with each code's EPL+SHEAR
implementation, and solves the same rectangular pixelised-source inversion for
all adapters. The lens-light amplitude and source pixels are linear parameters.

For a quick optimizer smoke test, use:

```bash
.venv/bin/python -m benchmark.run_single_mock \
  --mode fit \
  --maxiter 25 \
  --nx-src 25 \
  --ny-src 25 \
  --adapters lenstronomy \
  --output results/single_mock_fit_lenstronomy.json
```

For a capped Nautilus smoke test, use:

```bash
.venv/bin/python -m benchmark.run_single_mock \
  --mode nautilus \
  --nlive 100 \
  --n-eff 200 \
  --n-like-max 1000 \
  --nx-src 25 \
  --ny-src 25 \
  --adapters lenstronomy,jaxtronomy,herculens,tinylensgpu,autolens \
  --output results/single_mock_nautilus_smoke.json
```

## Run manifest

Before any cluster run, write a manifest next to the results:

```bash
.venv/bin/python -m benchmark.manifest results/run_manifest.json
```

The manifest records each local repo's branch, commit, remote, dirty status,
short status, and the `uv.lock` SHA256. Benchmark production runs should be
made from committed benchmark branches or tags, not floating `main` checkouts.
