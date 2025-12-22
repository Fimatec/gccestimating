"""Generalized Cross Correlation Estimators.

Istanciation Signatures:

- `gcc = GCC(fftlen)`

Estimators:

`gcc.cc()`, `gcc.roth()`, `gcc.scot()`, `gcc.phat()`, `gcc.ht()`

`gcc.gamma12()`

"""
from dataclasses import dataclass as _dataclass
import numpy as _np
import scipy as _sc


class GCC(object):
    """
    Returns a GCC instance to listen two radios.

    Provides estimation methods for Generalized Cross Correlation.

    Parameters
    ----------
    sig_len: int
        Length of the both input signals
    upsample: int
        Whittaker–Shannon interpolation density (default=1)
    dtype: str
        Data type in str, compatible with np.dtype(dtype_str)
    beta : float (default=0.9)
        EMA Smoothing factor for spectra

    Returns
    -------
    gcc : GCC

    """

    def __init__(self, sig_len, upsample=1, dtype='complex64', beta=0.9):
        self._fft, self._ifft = _get_fftfuncs(dtype)
        self._upsample = upsample
        self._sig_len = sig_len
        self._corr_len = 2 * sig_len - 1
        self._out_len = 2 * sig_len * upsample - 1
        self._pad = self._out_len - self._corr_len
        self._beta = beta
        self._spec1 = None
        self._spec2 = None   
        self._spec11 = None
        self._spec22 = None
        self._spec12 = None
        self._gamma12 = None
        self._cc = None
        self._roth = None
        self._scot = None
        self._phat = None
        self._eckart = None
        self._ht = None

    def fit_from_spectra(self, spec1, spec2):
        self._spec1 = spec1
        self._spec2 = spec2   
        if self._spec11 is None:
            self._spec11 = _np.real(self._spec1 * _np.conj(self._spec1))
            self._spec22 = _np.real(self._spec2 * _np.conj(self._spec2))
            self._spec12 = self._spec1 * _np.conj(self._spec2)
        else:
            self._spec11 = self._beta * self._spec11 + \
                (1 - self._beta) * _np.real(self._spec1 * _np.conj(self._spec1))
            self._spec22 = self._beta * self._spec22 + \
                (1 - self._beta) * _np.real(self._spec2 * _np.conj(self._spec2))
            self._spec12 = self._beta * self._spec12 + \
                (1 - self._beta) * (self._spec1 * _np.conj(self._spec2))
        self._gamma12 = self.gamma12()
        self._cc = None
        self._roth = None
        self._scot = None
        self._phat = None
        self._eckart = None
        self._ht = None
        return self

    def fit(self, sig1, sig2):
        spec1, spec2 = map(self.fft, (sig1, sig2))
        return self.fit_from_spectra(spec1, spec2)

    def fft(self, sig):
        return self._fft(sig, self._corr_len)
        
    def _backtransform(self, spec):
        spec = _np.r_[
            spec[:self._corr_len//2+1],
            _np.zeros(self._pad, dtype=complex),
            spec[self._corr_len//2+1:]
        ] * self._upsample
        sig = self._ifft(spec)
        sig = _sc.fft.fftshift(sig)
        return sig

    def clear(self):
        self._spec1 = None
        self._spec2 = None   
        self._spec11 = None
        self._spec22 = None
        self._spec12 = None
        self._gamma12 = None
        self._cc = None
        self._roth = None
        self._scot = None
        self._phat = None
        self._eckart = None
        self._ht = None

    @_dataclass(init=True, repr=True, eq=True)
    class Estimate():
        """Data of an Estimate. 
        Instances are returned by estimators in GCC.
        
        Parameters
        ----------
        name : str
            Name of the estimator.
        sig : ndarray
            Estimator signal array (Rxy(t), Cross Correlation).
        spec : ndarray
            Estimator spectrum (Rxy(f)).

        """
        name: str
        sig: _np.ndarray
        spec: _np.ndarray

        def __array__(self, dtype=None):
            if dtype is not None:
                return self.sig.astype(dtype)
            else:
                return self.sig

        def __len__(self):
            return len(self.sig)


    def cc(self):
        """Returns GCC estimate 
        
        $\\mathcal{F}^{-1} (S_{xy})$
        
        """         
        if self._cc is None:
            self._cc = GCC.Estimate(
                name='CC', 
                sig=self._backtransform(self._spec12), 
                spec=self._spec12)

        return self._cc

    def roth(self):
        """Returns GCC Roth estimate 
    
        $\\mathcal{F}^{-1} (S_{xy}/S_{xx})$

        """
        if self._roth is None:
            spec = self._spec12 / _prevent_zerodivision(self._spec11)
            self._roth = GCC.Estimate(
                name='Roth', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._roth

    def scot(self, alpha=0.5):
        """Returns parameterized GCC SCOT estimate 
        
        Smoothed gamma12 Transformed GCC.
    
        $\\mathcal{F}^{-1} (S_{xy}/{(S_{xx}S_{yy})^\\alpha})$
        
        """        
        if self._scot is None:
            spec = self._spec12 / _prevent_zerodivision(
                (self._spec11*self._spec22)**alpha)
            self._scot = GCC.Estimate(
                name='SCOT', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._scot
        
    def gamma12(self):
        """Returns gamma12 $\\gamma_{12}(f)$"""
        if self._gamma12 is None:
            self._gamma12 = self._spec12 / _prevent_zerodivision(
                _np.sqrt(self._spec11*self._spec22))
        return self._gamma12

    def coherence(self):
        """Returns the coherence."""
        return self.gamma12()**2

    def phat(self, alpha=1.0):
        """Returns parameterized GCC PHAT estimate 
        
        PHAse Transformed GCC.
        
        $\\mathcal{F}^{-1}(S_{xy}/|S_{xy}|^\\alpha)$
        
        """        
        if self._phat is None:
            spec = self._spec12 / _prevent_zerodivision(
                _np.abs(self._spec12)**alpha)
            self._phat = GCC.Estimate(
                name='PHAT', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._phat

    def eckart(self, sig0, noise1, noise2):
        """Returns an eckart estimate.
        
        Parameters
        ----------
        sig0 : ndarray
            estimate of the actual signal to be correlated.
        noise1 : ndarray
            estimated noise in sig1.
        noise2 : ndarray
            estimated noise in sig2

        Returns
        -------
        estmate : GCC.Estimate

        Note
        ----
        only implemented, not fully tested.

        """
        if self._eckart is None:
            spec_sig0 = self._fft(sig0, self._fftlen)
            spec_noise1 = self._fft(noise1, self._fftlen)
            spec_noise2 = self._fft(noise2, self._fftlen)
            spec_sig00 = _np.real(spec_sig0*spec_sig0.conj())
            spec_noise11 = _np.real(spec_noise1*spec_noise1.conj())
            spec_noise22 = _np.real(spec_noise2*spec_noise2.conj())
            weight = spec_sig00 /_prevent_zerodivision(
                spec_noise11*spec_noise22)
            spec = self._spec12*weight
            self._eckart = GCC.Estimate(
                name='Eckart', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._eckart

    def ht(self):
        """Returns GCC HT estimate"""
        if self._ht is None:
            coh = _np.abs(self.gamma12())**2
            spec = self._spec12*coh/_prevent_zerodivision(_np.abs(self._spec12)*(1-coh))
            self._ht = GCC.Estimate(
                name='HT', 
                sig=self._backtransform(spec), 
                spec=spec)
        return self._ht


def corrlags(n, samplerate=1, upsample=1):
    """Returns array of lags.
    
    Parameters
    ----------
    n: int
        Length of the signal
    samplerate : scalar
        Samplerate of the signal
    upsample: int
        Interpolation density (default=1)
    
    Returns
    -------
    lags : ndarray
    
    """
    dt = 1 / (samplerate * upsample)
    m = n * upsample
    return _np.arange(-m+1, m) * dt


def _get_fftfuncs(dtype):
    """Returns fft, ifft function depending on given dtype (str)."""
    if "complex" in dtype: return _sc.fft.fft, _sc.fft.ifft 
    else: return _sc.fft.rfft, _sc.fft.irfft


def _prevent_zerodivision(sig, reg=1e-12, rep=1e-12):
    """Replaces values smaler reg. Same for negative values and negative reg.
    
    Replaces 
    
    `sig < reg & sig >= 0` with `rep`
    
    and
    
    `sig > -reg & sig <= 0` with `-rep`

    Parameters
    ----------
    sig : ndarray
        Will be modified by this function. 
        Provide a copy of your array if original is needed.
    reg : scalar
        All values around 0 (-reg, reg) are replaced by `rep`
    rep : scalar
        Replace value.

    Returns
    -------
    sig : ndarray
        Modified sig usable for division.

    """
    reg = abs(reg)
    rep = abs(rep)
    sig[_np.logical_and(sig < reg, sig >= 0)] = rep
    sig[_np.logical_and(sig > -reg, sig <= 0)] = -rep
    return sig
