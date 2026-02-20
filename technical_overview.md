# Technical Overview — gccestimating

## 1. What is this project?

**gccestimating** is a Python library that implements **Generalized Cross Correlation (GCC)** estimators for time-delay estimation between two signals. It is based on the foundational paper by Knapp and Carter (1976).

The library computes cross-correlations entirely in the frequency domain using the Wiener-Kinchin relations, and provides six different spectral weighting strategies that trade off between noise robustness, peak sharpness, and resolution.

**Dependencies:** numpy >= 1.19, scipy >= 1.5, Python > 3.8

## 2. Core Concept

The fundamental problem: two sensors observe the same source signal at different times. The time difference (delay) between the two observations encodes spatial information — e.g., the direction or distance to the source.

**Standard cross-correlation** finds this delay by sliding one signal past the other and measuring similarity at each lag. The peak of the cross-correlation corresponds to the time delay.

**Generalized cross-correlation** improves on this by working in the frequency domain. Instead of directly correlating the time-domain signals, GCC:

1. Transforms both signals into the frequency domain (FFT)
2. Computes the cross-power spectrum
3. Applies a **weighting function** to the spectrum (this is the "generalized" part)
4. Transforms back to the time domain (IFFT)

Different weighting functions emphasize different spectral properties — some suppress noise, some sharpen the correlation peak, some maximize the likelihood of correct detection. Choosing the right one depends on the signal-to-noise ratio and noise characteristics.

## 3. Mathematical Foundation

### General GCC formula

$$\hat{R}_{xy}^{(\text{g})}(\tau) = \int_{-\infty}^{\infty}{\psi_\text{g}(f) \, G_{xy}(f) \, e^{i 2\pi f \tau} \, df}$$

where:
- $\hat{R}_{xy}^{(\text{g})}(\tau)$ is the generalized cross-correlation at lag $\tau$
- $\psi_\text{g}(f)$ is the weighting function (varies per estimator)
- $G_{xy}(f)$ is the cross-power spectrum

### Spectral quantities (Wiener-Kinchin relations)

Given signals $x(t)$ and $y(t)$ with Fourier transforms $X(f)$ and $Y(f)$:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| $G_{xx}(f)$ | $X(f) \cdot X^*(f) = \lvert X(f) \rvert^2$ | Auto-power spectrum of $x$ |
| $G_{yy}(f)$ | $Y(f) \cdot Y^*(f) = \lvert Y(f) \rvert^2$ | Auto-power spectrum of $y$ |
| $G_{xy}(f)$ | $X(f) \cdot Y^*(f)$ | Cross-power spectrum |
| $\gamma_{xy}(f)$ | $\frac{G_{xy}(f)}{\sqrt{G_{xx}(f) G_{yy}(f)}}$ | Coherence (normalized cross-spectrum) |

The coherence $\gamma_{xy}(f)$ measures how linearly related $x$ and $y$ are at each frequency. A value of 1 means perfectly correlated; 0 means uncorrelated (noise-dominated).

## 4. The Six Estimators

Each estimator differs only in its weighting function $\psi(f)$. They all follow the same pipeline: weight the cross-power spectrum, then inverse-FFT.

### CC — Cross Correlation

$$\psi_{\text{CC}} = 1$$

The baseline. No weighting applied. Equivalent to standard time-domain cross-correlation. Good when SNR is high.

**Method:** `gcc.cc()`

### Roth

$$\psi_{\text{Roth}} = \frac{1}{G_{xx}(f)}$$

Divides by the auto-power of the first signal. This is the H1 Wiener-Hopf estimator. It whitens the first signal, reducing bias from its spectral shape. The peak value at the true delay equals 1 for autocorrelation.

**Method:** `gcc.roth()`

### SCOT — Smoothed Coherence Transform

$$\psi_{\text{SCOT}} = \frac{1}{(G_{xx}(f) \, G_{yy}(f))^\alpha}$$

Normalizes by both auto-power spectra raised to $\alpha$. This is a symmetric version of Roth that whitens both signals.

- **Original SCOT:** $\alpha = 0.5$ — full normalization, produces the coherence function
- **Improved SCOT:** $\alpha = 0.75$ — partial normalization, often more robust in practice

**Method:** `gcc.scot(alpha=0.5)`

### PHAT — Phase Transform

$$\psi_{\text{PHAT}} = \frac{1}{|G_{xy}(f)|^\alpha}$$

Normalizes the cross-spectrum by its own magnitude. At $\alpha = 1$, only phase information remains — all frequency bins contribute equally regardless of amplitude. Produces the sharpest peaks but is sensitive to noise.

- **Original PHAT:** $\alpha = 1$ — pure phase, sharpest peak
- **Improved PHAT:** $\alpha \in [0.5, 0.7]$ — retains some amplitude weighting for noise robustness

**Method:** `gcc.phat(alpha=1.0)`

### Eckart

$$\psi_{\text{Eckart}} = \frac{G_{uu}(f)}{G_{nn}(f) \, G_{mm}(f)}$$

Requires separate estimates of the clean signal ($u$) and the noise in each channel ($n$, $m$). Weights frequencies based on signal-to-noise ratio. Optimal when noise statistics are known.

**Method:** `gcc.eckart(sig0, noise1, noise2)`

**Note:** Implemented but not fully tested. The method uses `self._fftlen` internally which appears to reference an older attribute name.

### HT — Hanan Thomson / Maximum Likelihood

$$\psi_{\text{HT}} = \frac{|\gamma_{xy}(f)|^2}{|G_{xy}(f)| \, (1 - |\gamma_{xy}(f)|^2)}$$

Uses the coherence to weight frequencies. Frequencies where the signals are highly coherent get amplified; frequencies dominated by noise (low coherence) get suppressed. Theoretically optimal for Gaussian noise.

**Method:** `gcc.ht()`

### Estimator Selection Guide

| Scenario | Recommended | Why |
|----------|-------------|-----|
| High SNR, broadband signal | CC | Simple, no artifacts |
| Unknown noise, general purpose | SCOT (0.5) | Balanced normalization |
| Reverberant environment | PHAT (1.0) | Sharp peak cuts through reflections |
| Moderate noise | PHAT (0.5–0.7) | Good peak sharpness with some noise rejection |
| Known noise characteristics | Eckart | Optimal weighting with noise estimates |
| Gaussian noise, multiple observations | HT | Maximum likelihood, benefits from averaging |

## 5. Architecture and Data Flow

### Single-file library

The entire implementation lives in `gccestimating.py` (~328 lines). There are no submodules.

### Processing pipeline

```
Input signals ──► FFT (zero-padded to corr_len) ──► Cross-power & auto-power spectra
                                                          │
                                              ┌───────────┼───────────┐
                                              ▼           ▼           ▼
                                           S_xx(f)     S_yy(f)     S_xy(f)
                                              │           │           │
                                              └───────────┼───────────┘
                                                          │
                                              Apply weighting ψ(f)
                                                          │
                                              Zero-pad at Nyquist (upsample)
                                                          │
                                                   Inverse FFT
                                                          │
                                                      fftshift
                                                          │
                                                  GCC.Estimate object
                                                  (sig=time domain, spec=frequency domain)
```

### The GCC class

**Constructor:** `GCC(sig_len, upsample=1, dtype='complex64', beta=0.9)`

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `sig_len` | int | — | Length of input signals (both must be the same length) |
| `upsample` | int | 1 | Whittaker-Shannon interpolation factor. 2 = double the lag resolution |
| `dtype` | str | `'complex64'` | Controls FFT function selection (complex → fft/ifft, real → rfft/irfft) |
| `beta` | float | 0.9 | EMA smoothing factor for spectral averaging across multiple `fit()` calls |

**Key internal attributes set by the constructor:**

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `_sig_len` | `sig_len` | Stored input length |
| `_corr_len` | `2 * sig_len - 1` | Length of the linear cross-correlation |
| `_out_len` | `2 * sig_len * upsample - 1` | Length of the upsampled output |
| `_pad` | `_out_len - _corr_len` | Number of zeros to insert for upsampling |
| `_beta` | `beta` | EMA factor |
| `_spec1`, `_spec2` | None | Cached FFT spectra |
| `_spec11`, `_spec22`, `_spec12` | None | Cached power spectra |
| `_cc`, `_roth`, etc. | None | Cached estimate results |

### The Estimate dataclass

Every estimator method returns a `GCC.Estimate` object:

```python
@dataclass
class Estimate:
    name: str          # Estimator name (e.g., 'CC', 'PHAT')
    sig: np.ndarray    # Time-domain correlation signal
    spec: np.ndarray   # Frequency-domain weighted spectrum
```

It supports:
- `len(estimate)` — returns length of `sig`
- `np.asarray(estimate)` — returns `sig` as a numpy array
- `np.asarray(estimate, dtype=np.float16)` — returns `sig` cast to the given dtype

### Caching

All estimates are lazily computed and cached. Calling `gcc.phat()` twice returns the same object without recomputation. Calling `fit()` or `fit_from_spectra()` clears all cached estimates (but not the EMA-averaged spectra).

### EMA spectral averaging

When `fit()` or `fit_from_spectra()` is called multiple times, the spectra are smoothed via exponential moving average:

$$S_{\text{new}} = \beta \cdot S_{\text{old}} + (1 - \beta) \cdot S_{\text{current}}$$

- `beta = 0.0` — no averaging, each call uses only the current spectra
- `beta = 0.9` — strong smoothing, old spectra dominate (good for streaming/real-time)
- `beta = 1.0` — frozen, ignores new data entirely

The first call always uses the raw spectra (no prior to average with). This averaging reduces noise in the spectral estimates when processing sequential blocks of data from a continuous stream.

## 6. Key Implementation Details

### FFT function selection (`_get_fftfuncs`)

Based on the `dtype` parameter:
- If `dtype` contains `"complex"` → uses `scipy.fft.fft` / `scipy.fft.ifft` (two-sided spectrum)
- Otherwise → uses `scipy.fft.rfft` / `scipy.fft.irfft` (one-sided spectrum, ~2x faster for real signals)

### Zero-padding for correlation length

Before computing the FFT, signals are zero-padded to `corr_len = 2 * sig_len - 1`. This ensures the FFT-based multiplication computes a **linear** cross-correlation (not circular). This is the standard technique equivalent to `numpy.correlate(sig1, sig2, 'full')`.

### Whittaker-Shannon upsampling (`_backtransform`)

To achieve sub-sample lag resolution without changing the input signals, the library uses spectral zero-insertion:

1. Split the spectrum at the Nyquist frequency
2. Insert `_pad` zeros between the two halves
3. Multiply by the upsample factor
4. Inverse FFT produces a sinc-interpolated time-domain signal

This is mathematically exact (not an approximation) — it is the Whittaker-Shannon interpolation formula applied in the frequency domain.

### Zero-division protection (`_prevent_zerodivision`)

Several estimators divide by spectral magnitudes that can be zero. The function:

- Replaces values in the range $[0, \text{reg})$ with $\text{rep}$ (default: $10^{-12}$)
- Replaces values in the range $(-\text{reg}, 0]$ with $-\text{rep}$
- Modifies the array **in-place**

### `fftshift` for centering

After the inverse FFT, `scipy.fft.fftshift` rearranges the output so that lag=0 is at the center of the array. Without this, lag=0 would be at index 0 and negative lags would wrap around to the end.

## 7. API Reference

### `GCC` class

#### `GCC(sig_len, upsample=1, dtype='complex64', beta=0.9)`

Create a GCC processor for signals of length `sig_len`.

#### `gcc.fit(sig1, sig2) → self`

Process two time-domain signals. Computes FFTs internally and updates spectral estimates (with EMA averaging if not the first call). Returns `self` for method chaining.

#### `gcc.fit_from_spectra(spec1, spec2) → self`

Same as `fit()` but accepts pre-computed frequency-domain spectra (e.g., from `scipy.fft.fft`). Useful when you've already computed FFTs or are working with streaming FFT data.

#### `gcc.fft(sig) → ndarray`

Compute the FFT of a signal, zero-padded to `_corr_len`. Convenience wrapper used internally by `fit()`.

#### `gcc.clear()`

Reset all cached spectra and estimates to `None`. Use this to start fresh without creating a new GCC instance.

#### `gcc.cc() → GCC.Estimate`

Cross Correlation estimate. No spectral weighting.

#### `gcc.roth() → GCC.Estimate`

Roth estimate. Divides cross-spectrum by auto-power of signal 1.

#### `gcc.scot(alpha=0.5) → GCC.Estimate`

SCOT estimate. Divides by both auto-powers raised to `alpha`. Use `alpha=0.75` for the improved variant.

#### `gcc.phat(alpha=1.0) → GCC.Estimate`

PHAT estimate. Divides cross-spectrum by its own magnitude raised to `alpha`. Use `alpha` in [0.5, 0.7] for the improved variant.

#### `gcc.eckart(sig0, noise1, noise2) → GCC.Estimate`

Eckart estimate. Requires separate estimates of the clean signal and noise in each channel.

#### `gcc.ht() → GCC.Estimate`

Hanan Thomson / Maximum Likelihood estimate. Uses coherence-weighted cross-spectrum.

#### `gcc.gamma12() → ndarray`

Returns the coherence spectrum $\gamma_{xy}(f)$ (complex-valued, normalized cross-spectrum).

#### `gcc.coherence() → ndarray`

Returns the squared coherence $|\gamma_{xy}(f)|^2$ (real-valued, range [0, 1]).

### Utility functions

#### `corrlags(n, samplerate=1, upsample=1) → ndarray`

Generate an array of lag values (in seconds if `samplerate` is in Hz) corresponding to the output of a GCC estimate.

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `n` | int | — | Length of the input signal |
| `samplerate` | float | 1 | Sample rate in Hz |
| `upsample` | int | 1 | Must match the upsample factor used in GCC |

Returns an array of length `2 * n * upsample - 1`, centered at 0.

## 8. Usage Examples

### Basic time-delay estimation

```python
import numpy as np
from gccestimating import GCC, corrlags

# Two signals: same noise, offset by 500 samples
nsamp = 1024
noise = np.random.randn(256)

sig1 = np.random.randn(nsamp) * 0.5
sig2 = np.random.randn(nsamp) * 0.5
sig1[:256] = noise
sig2[500:756] = noise

# Create GCC (beta=0.0 disables averaging for single-shot use)
gcc = GCC(nsamp, beta=0.0)
gcc.fit(sig1, sig2)

# Compute estimates
cc = gcc.cc()
phat = gcc.phat()

# Find the delay: peak of the correlation
lags = corrlags(nsamp, samplerate=1)
delay = lags[np.argmax(np.abs(cc.sig))]
print(f"Estimated delay: {delay} samples")
```

### Sub-sample resolution with upsampling

```python
# 4x interpolation for finer lag resolution
gcc = GCC(nsamp, upsample=4, beta=0.0)
gcc.fit(sig1, sig2)

phat = gcc.phat()
lags = corrlags(nsamp, samplerate=44100, upsample=4)
delay = lags[np.argmax(np.abs(phat.sig))]
print(f"Estimated delay: {delay:.6f} seconds")
```

### Streaming with EMA averaging

```python
# Process consecutive blocks with spectral smoothing
gcc = GCC(block_size, beta=0.9)

for block1, block2 in stream_blocks():
    gcc.fit(block1, block2)
    # Each fit() call smooths with previous spectra
    # Estimates improve over time as noise averages out

phat = gcc.phat()
```

### Working with pre-computed spectra

```python
spec1 = np.fft.fft(sig1, 2 * len(sig1) - 1)
spec2 = np.fft.fft(sig2, 2 * len(sig2) - 1)

gcc = GCC(len(sig1), dtype='complex64', beta=0.0)
gcc.fit_from_spectra(spec1, spec2)
cc = gcc.cc()
```

## 9. Project Structure

```
gccestimating/
├── gccestimating.py        # All source code (~328 lines)
├── test_gccestimating.py   # pytest test suite
├── pyproject.toml           # Build config (hatchling), dependencies
├── setup.py                 # Legacy setup (imports from pyproject.toml)
├── README.md                # Usage examples, math formulas
├── LICENSE                  # MPL-2.0
├── .travis.yml              # CI configuration
├── .readthedocs.yml         # Documentation hosting config
└── docs/                    # Sphinx documentation source
    ├── conf.py
    ├── index.rst
    ├── Makefile
    ├── make.bat
    └── requirements.txt
```

### Install

```bash
pip install .
```

### Run tests

```bash
pytest test_gccestimating.py
```

## 10. References

- Knapp, C. H. and Carter, G. C., "The Generalized Correlation Method for Estimation of Time Delay," IEEE Trans. Acoust., Speech, Signal Processing, Vol. ASSP-24, No. 4, August 1976.
