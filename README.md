# Generalized Cross Correlation (GCC) Estimates

This project provides estimators for the generalized cross correlation according to *Knapp and Carter 1976* [KC76].


## Implemented Estimators (compare [KC76])

The generalized Estimator can be described by


$$\hat{R}_{xy}^{(\text{g})}(\tau) = \int_{-\infty}^{\infty}{\psi_\text{g}(f) G_{xy}(f)~e^{\text{i} 2\pi f \tau} df}$$

where $G_{xy}(f)$ denotes the cross power spectrum of $x(t)$ and $y(t)$.
In this project, all estimates are computed in the spectral domain using the *Wiener-Kinchin relations* (e.g. $G_{xx}(f)=X(f)X^{*}(f)$).

Following estimators are implemented:

- **Cross Correlation** 
  $$\psi_{\text{CC}}=1$$

- **Roth**: same as the $\text{H}_1$ estimator describing the Wiener-Hopf filter
  $$\psi_{\text{Roth}} = \frac{1}{G_{xx}(f)}$$

- **Parameterized Smoothed Coherence Transform** (SCOT): 
  $$\psi_{\text{SCOT}} = \frac{1}{(G_{xx}(f)G_{yy}(f))^\alpha}$$
  - Original SCOT: $\alpha = 0.5$
  - Improved SCOT: $\alpha = 0.75$

- **Parameterized PHAse Transform** (PHAT): 
  $$\psi_{\text{PHAT}} = \frac{1}{|G_{xy}(f)|^\alpha}$$
  - Original PHAT: $\alpha = 1$
  - Improved PHAT: $\alpha\in[0.5, 0.7]$

- **Eckart**
  $$\psi_{\text{Eckart}} = \frac{G_{uu}(f)}{G_{nn}(f)G_{mm}(f)}$$

- **Hanan Thomson** (HT), also known as **Maximum Likelihood** (ML) estimator 
  $$\psi_{\text{HT}} = \psi_{\text{ML}} = \frac{\left|\gamma_{xy}(f)\right|^2}{\left|G_{xy}\right| \left(1-\gamma_{xy}(f)\right)^2}$$
  with 
  $$\gamma_{xy}(f) = \frac{G_{xy}(f)}{\sqrt{G_{xx}(f)G_{yy}(f)}}$$

## Insalling

`pip install .`

otherwise use your own favorite way to install/use the code in your environment.

## Example

```python
import numpy as np
import matplotlib.pylab as plt
from gccestimating import GCC, corrlags

 # generate some noise signals
nsamp = 1024

noise1 =  0.5*np.random.randn(nsamp)
sig1 = np.zeros(nsamp) + noise1

noise2 =  0.5*np.random.randn(nsamp)
sig2 = np.zeros_like(sig1) + noise2

noise_both = np.random.randn(256)

sig1[:256] = noise_both
sig2[500:756] = noise_both

# Create the a GCC instance, without averaging
gcc = GCC(nsamp, beta=0.0)
gcc.fit(sig1, sig2)

# create a lags array
lags = corrlags(gcc._corrlen, samplerate=1)

def mkplot(est, p):
    plt.subplot(p)
    plt.plot(lags, est.sig, label=est.name)
    plt.legend()

# calculate the standard cc estimate
cc_est = gcc.cc()

# plot it using the mkplot function
mkplot(cc_est, 611)

# plot the other estimates
mkplot(gcc.scot(), 612)
mkplot(gcc.phat(), 613)
mkplot(gcc.roth(), 614)
mkplot(gcc.ht(), 615)
mkplot(gcc.eckart(noise_both, noise1, noise2), 616)

# compare cc to the timedomain based 
# implementation from Numpy
# you will see: very close (errors < 1e-13)
plt.figure()
plt.plot(np.correlate(sig1, sig2, 'full'))
plt.plot(gcc.cc())
plt.show()

```

## References

[KC76]: Knapp and Carter, "The Generalized Correlation Method for Estimation of Time Delay", IEEE Trans. Acoust., Speech, Signal Processing, August, 1976
