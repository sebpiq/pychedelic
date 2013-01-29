from tempfile import NamedTemporaryFile
import math
import numpy as np
import pandas as pnd
import pysox

from utils.files import read_wav, write_wav


def fft(samples, sample_rate):
    """
    Performs the FFT of a real, 1-dimension signal.
    """
    f_results = np.fft.fftfreq(len(samples), 1.0 / sample_rate)
    f_results = f_results[:len(f_results)/2 + 1]
    f_results[-1] = -f_results[-1] # Because last term is -Nyquist f
    results = np.fft.rfft(samples)
    return f_results, results


def ifft(samples, sample_rate):
    """
    Performs the inverse FFT of the spectrum of a real 1-dimension signal (when in time domain).
    """
    results = np.fft.irfft(samples)
    t_results = np.arange(0, (len(samples) - 1) * 2) * 1.0 / sample_rate
    return t_results, results


def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.
 inverse_tan(b(n)/a(n))
    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage :
        
        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz
        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    if isinstance(samples, np.ndarray): samples = samples.tolist() # We need simple list, no numpy.array
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    f_results = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(complex result, amplitude, phase)`
        results.append(complex(0.5 * w_real * d1 - d2, w_imag * d1))
        f_results.append(f * sample_rate)
    return np.array(f_results), np.array(results)


def get_ft_phase(term):
    """
    """
    return math.atan2(term.imag / term.real)


def get_ft_amplitude(term):
    """
    """
    return abs(term)


def get_ft_phase_array(array):
    """
    """
    return np.arctan2(np.imag(array), np.real(array)).astype(float)


def get_ft_amplitude_array(array):
    """
    """
    return np.absolute(array).astype(float)


def maxima(dseries, take_edges=True):
    cond1 = lambda grad: grad > 0
    cond2 = lambda grad: grad < 0
    return _extrema(dseries, cond1, cond2, take_edges=take_edges)


def minima(dseries, take_edges=True):
    cond1 = lambda grad: grad < 0
    cond2 = lambda grad: grad > 0
    return _extrema(dseries, cond1, cond2, take_edges=take_edges)


def _extrema(dseries, cond1, cond2, take_edges=True):
    # Preparing data and helper functions.
    # We will collect indices of extrema in `extrema_ind`, and then
    # apply this to `dseries`.
    dseries = dseries.sort_index()
    gradients = dseries.diff()[1:].values
    extrema_ind = []
    def extremum_found(ind):
        extrema_ind.append(ind)
    def gradient0_found(ind):
        j = ind + 1
        while j < len(gradients) - 2 and gradients[j] == 0: j += 1
        if j >= len(gradients) - 1 or cond2(gradients[j]): extremum_found(ind)

    # Handling lower edge
    if (take_edges and cond2(gradients[0])): extremum_found(0)
    if gradients[0] == 0: gradient0_found(0)

    # In the middle
    for i, grad in enumerate(gradients[:-1]):
        if cond1(grad):
            # i + 1, because we need `diff[i]` is `y[i+1] - y[i]`
            if cond2(gradients[i+1]): extremum_found(i+1)
            elif gradients[i+1] == 0: gradient0_found(i+1)

    # Handling upper edge         
    if take_edges and cond1(gradients[-1]): extremum_found(-1)

    # Return
    return dseries[extrema_ind]


def optimize_windowsize(n):
    orig_n = n
    while True:
        n = orig_n
        while (n % 2) == 0: n /= 2
        while (n % 3) == 0: n /= 3
        while (n % 5) == 0: n /= 5
        if n < 2: break
        orig_n += 1
    return orig_n


def smooth(samples, window_size=11, window_func='hanning'):
    """
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        data: the input signal `data(x, V)`
        window_size: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal `data(x, V)`
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_size/2-1):-(window_size/2)] instead of just y.

    http://www.scipy.org/Cookbook/SignalSmooth
    """
    if len(samples) < window_size:
        raise ValueError('sample count of the sound needs to be bigger than window size.')
    if window_size < 3:
        return samples

    w = window(window_func, window_size)
    s = np.r_[samples[window_size-1:0:-1], samples, samples[-1:-window_size:-1]]

    smoothed = np.convolve(w/w.sum(), s, mode='valid')
    smoothed = smoothed[math.floor(window_size/2.0)-1:-math.ceil(window_size/2.0)] # To have `length(output) == length(input)`
    return pnd.Series(smoothed, index=samples.index)


def window(wfunc, size):
    if not wfunc in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    if wfunc == 'flat': # moving average
        return np.ones(size, 'd')
    else:
        return getattr(np, wfunc)(size)


def time_stretch(samples, ratio, sample_rate=44100):
    with NamedTemporaryFile(delete=True, suffix='.wav') as infile:
        with NamedTemporaryFile(delete=True, suffix='.wav') as outfile:
            write_wav(infile, samples, sample_rate=sample_rate)
            infile.flush()
            sox_in = pysox.CSoxStream(infile.name)
            sox_out = pysox.CSoxStream(outfile.name, 'w', sox_in.get_signal())
            chain = pysox.CEffectsChain(sox_in, sox_out)
            effect = pysox.CEffect('tempo', [b'%s' % ratio])
            chain.add_effect(effect)
            chain.flow_effects()
            sox_out.close()
            outfile.flush()
            data, infos = read_wav(outfile)
    return data


def interleaved(samples):
    """
    Convert an array of multi-channel data to a 1d interleaved array.
    For example :

        >>> data_3channels = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
        >>> interleaved(data_3channels)
        np.array([1, 2, 3, 11, 22, 33, 111, 222, 333])
    """
    if samples.ndim == 2 and (samples.size % samples.shape[1]) != 0:
        raise ValueError('malformed array')
    if samples.ndim > 2: raise ValueError('Data should be 1 or 2 dimensions')

    if samples.ndim == 1:
        return samples
    elif samples.ndim == 2:
        return samples.reshape((samples.size,))


def deinterleaved(samples, channel_count):
    if samples.ndim != 1:
        raise ValueError('interleaved arrays have only one dimension')
    if (samples.size % channel_count) != 0:
        raise ValueError('malformed array')
    return samples.reshape((samples.size / channel_count, channel_count))

