"""Generalized Cross Correlation Estimators.

Istanciation Signatures:

- `gcc = GCC(sig1, sig2, fftlen)`

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
    fftlen : int or None
        Length of fft to be computed. 
        If None, it will be calculated automatically as next power of two.
    beta : float (default=0.9)
        EMA Smoothing factor for spectra

    Returns
    -------
    gcc : GCC

    """

    def __init__(self, sig_len, fftlen=None, window='boxcar', beta=0.9):
        corrlen = 2 * sig_len - 1
        fftlen = fftlen or int(2**_np.ceil(_np.log2(corrlen)))
        self._corrlen = corrlen
        self._fftlen = fftlen
        self._window = window
        self._beta = beta
        self._spec1 = None
        self._spec2 = None   
        self._sig1 = None
        self._sig2 = None   
        self._fft = None  
        self._ifft = None
        self._spec11 = 0
        self._spec22 = 0
        self._spec12 = 0
        self._gamma12 = None
        self._cc = None
        self._roth = None
        self._scot = None
        self._phat = None
        self._eckart = None
        self._ht = None

    def fit(self, sig1, sig2) -> 'GCC':
        fft, ifft = _get_fftfuncs(sig1, sig2)
        spec1 = fft(sig1, self._fftlen)
        spec2 = fft(sig2, self._fftlen)
        self._sig1 = sig1
        self._sig2 = sig2   
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
        self._fft = fft  
        self._ifft = ifft
        self._gamma12 = None
        self._cc = None
        self._roth = None
        self._scot = None
        self._phat = None
        self._eckart = None
        self._ht = None
        return self

    def _backtransform(self, spec):
        window = _sc.signal.get_window(self._window, len(spec))
        sig = self._ifft(window * spec, self._fftlen)
        sig = _np.roll(sig, len(sig)//2)
        start = (len(sig)-self._corrlen)//2 + 1
        return sig[start:start+self._corrlen]

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

        def index_to_lag(self, index, samplerate=None):
            lag = (index - len(self.sig)//2) 
            if samplerate:
                lag /= samplerate
            return lag

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


def corrlags(corrlen, samplerate=1):
    """Returns array of lags.
    
    Parameters
    ----------
    corrlen : int
        Lenght of correlation function (usually 2N-1).
    samplerate : scalar
    
    Returns
    -------
    lags : ndarray
    
    """
    dt = 1 / samplerate
    la = corrlen // 2
    lb = la+1 if corrlen%2 else la
    return _np.arange(-la*dt, lb*dt, dt)


def _get_fftfuncs(*signals):
    """Returns fft, ifft function depending on given signals data type."""
    if _np.all([_np.all(_np.isreal(sig)) for sig in signals]):
        return _sc.fft.rfft, _sc.fft.irfft
    else:
        return _sc.fft.fft, _sc.fft.ifft


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
