#coding=utf8
from tempfile import NamedTemporaryFile
import math
import numpy as np
from scipy import signal
import pandas as pnd
import pysox

from utils.files import read_wav, write_wav
from utils.stream import reblock


def fft(samples, frame_rate):
    """
    Performs the FFT of a real, 1-dimension signal.
    """
    f_results = np.fft.fftfreq(len(samples), 1.0 / frame_rate)
    f_results = f_results[:len(f_results)/2 + 1]
    f_results[-1] = -f_results[-1] # Because last term is -Nyquist f
    results = np.fft.rfft(samples)
    return f_results, results


def ifft(samples, frame_rate):
    """
    Performs the inverse FFT of the spectrum of a real 1-dimension signal (when in time domain).
    """
    results = np.fft.irfft(samples)
    t_results = np.arange(0, (len(samples) - 1) * 2) * 1.0 / frame_rate
    return t_results, results


def goertzel(samples, frame_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `frame_rate`.
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
    f_step = frame_rate / float(window_size)
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
        f_results.append(f * frame_rate)
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


def time_stretch(samples, ratio, frame_rate=44100):
    with NamedTemporaryFile(delete=True, suffix='.wav') as infile:
        with NamedTemporaryFile(delete=True, suffix='.wav') as outfile:
            write_wav(infile, samples, frame_rate=frame_rate)
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


def optimize_block_size(n):
    orig_n=n
    while True:
        n=orig_n
        while (n%2)==0:
            n/=2
        while (n%3)==0:
            n/=3
        while (n%5)==0:
            n/=5

        if n<2:
            break
        orig_n+=1
    return orig_n


def paulstretch(samples, ratio, block_size=None, frame_rate=44100, nchannels=1, nsamples=0):
    """
    Paul's Extreme Sound Stretch (Paulstretch) - Python version
    
    by Nasca Octavian PAUL, Targu Mures, Romania
    
    https://github.com/paulnasca/paulstretch_python
    http://hypermammut.sourceforge.net/paulstretch/
    """
    #nsamples, nchannels = samples.shape[0], samples.shape[1]

    # make sure that block_size is even and larger than 16
    block_size = block_size or frame_rate / 4
    block_size = max(16, block_size)
    block_size = optimize_block_size(block_size)
    block_size = int(block_size/2) * 2
    half_block_size = int(block_size/2)

    # correct the end of the smp
    end_size = int(frame_rate*0.05)
    if end_size < 16: end_size=16
    #samples[nsamples-end_size:nsamples,:] *= np.array([np.linspace(1, 0, end_size)]).transpose()

    # create Window window
    window = pow(1.0 - pow(np.linspace(-1.0, 1.0, block_size), 2.0), 1.25)
    window = np.array([window]).transpose()
    old_windowed_buf = np.zeros((block_size, nchannels))

    for block in reblock(samples, block_size, when_exhausted='pad'):
        print block.shape, block_size
        # get the windowed buffer
        buf = block * window
    
        # get the amplitudes of the frequency components and discard the phases
        freqs = abs(np.fft.rfft(buf, axis=0))

        # randomize the phases by multiplication with a random complex number with modulus=1
        ph = np.random.uniform(0, 2*np.pi, (freqs.shape[0], nchannels)) * 1j
        freqs = freqs * np.exp(ph)

        # do the inverse FFT 
        buf = np.fft.irfft(freqs, axis=0)

        # window again the output buffer
        buf *= window

        # overlap-add the output
        output = buf[0:half_block_size,:] + old_windowed_buf[half_block_size:block_size,:]
        old_windowed_buf = buf

        print 'OUT', output.shape
        # remove the resulted amplitude modulation
        # update: there is no need to the new windowing function
        yield output



'''
def paulstretch(samples, ratio, windowsize_seconds=0.25, frame_rate=44100):
    """
    Paul's Extreme Sound Stretch (Paulstretch) - Python version
    
    by Nasca Octavian PAUL, Targu Mures, Romania
    
    https://github.com/paulnasca/paulstretch_python
    http://hypermammut.sourceforge.net/paulstretch/
    """
    nsamples, nchannels = samples.shape[0], samples.shape[1]

    # make sure that windowsize is even and larger than 16
    windowsize = int(windowsize_seconds * frame_rate)
    if windowsize < 16: windowsize = 16
    windowsize = optimize_windowsize(windowsize)
    windowsize = int(windowsize/2) * 2
    half_windowsize = int(windowsize/2)

    # correct the end of the smp
    end_size = int(frame_rate*0.05)
    if end_size < 16: end_size=16
    samples[nsamples-end_size:nsamples,:] *= np.array([np.linspace(1, 0, end_size)]).transpose()
    
    # compute the displacement inside the input file
    start_pos = 0.0
    displace_pos = (windowsize*0.5)/ratio

    # create Window window
    window = pow(1.0 - pow(np.linspace(-1.0, 1.0, windowsize), 2.0), 1.25)
    window = np.array([window]).transpose()
    old_windowed_buf = np.zeros((windowsize, nchannels))

    while start_pos < nsamples:
        # get the windowed buffer
        istart_pos = int(np.floor(start_pos))
        buf = samples[istart_pos:istart_pos+windowsize,:]
        if buf.shape[0] < windowsize:
            buf = np.append(buf, np.zeros((windowsize - buf.shape[0], nchannels)), axis=0)
        buf = buf * window
    
        # get the amplitudes of the frequency components and discard the phases
        freqs = abs(np.fft.rfft(buf, axis=0))

        # randomize the phases by multiplication with a random complex number with modulus=1
        ph = np.random.uniform(0, 2*np.pi, (freqs.shape[0], nchannels)) * 1j
        freqs = freqs * np.exp(ph)

        # do the inverse FFT 
        buf = np.fft.irfft(freqs, axis=0)

        # window again the output buffer
        buf *= window

        # overlap-add the output
        output = buf[0:half_windowsize,:] + old_windowed_buf[half_windowsize:windowsize,:]
        old_windowed_buf = buf

        # remove the resulted amplitude modulation
        # update: there is no need to the new windowing function
        yield output

        start_pos += displace_pos
'''




def calculate_replaygain(samples, frame_rate=44100):
    """
    Determine the replay gain (perceived loudness) of `samples` in dB.

    METHOD:
    1) Calculate Vrms every 50ms
    2) Sort in ascending order of loudness
    3) The value which most accurately matches perceived loudness is around 95% of the max,
        so this value is used by Replay Level.
    4) Convert this value into dB

    David Robinson, 10th July 2001. http://www.David.Robinson.org/
    ref : http://replaygain.hydrogenaudio.org/calculating_rg.html
    Ported to Python by Sébastien Piquemal <sebpiq@gmail.com>
    """
    channel_count = samples.shape[1]
    frame_count = samples.shape[0]
    try:
        coeffs = RG_FILTER_COEFFS[frame_rate]
    except KeyError:
        raise ValueError('frame rate %s is not supported' % frame_rate)
    a1, b1, a2, b2 = coeffs['a1'], coeffs['b1'], coeffs['a2'], coeffs['b2']

    # Window for Vrms calculation (50ms)
    rms_window_size = np.round(50 * (frame_rate/1000.0))
    rms_per_block = 20
    block_size = np.round(rms_per_block * rms_window_size)

    # Check that the file is long enough to process in block_size blocks
    if frame_count < block_size:
        raise ValueError('file too short')

    # Loop through all the file in blocks a defined above
    i = 0
    Vrms_all = []
    while i*block_size < frame_count:
        # Grab a section of audio and filter it using the equal
        # loudness curve filter
        inaudio = samples[i*block_size:(i+1)*block_size,:]
        inaudio = signal.lfilter(b1, a1, inaudio, axis=0);
        inaudio = signal.lfilter(b2, a2, inaudio, axis=0);

        # Calculate Vrms: take a 50ms block, calculate the mean of RMS of each channel
        for j in range(rms_per_block):
            rms_block = inaudio[j*rms_window_size:(j+1)*rms_window_size,:]
            if rms_block.shape[0] < rms_window_size: break # if there's not enough data left
            Vrms_all.append(np.power(rms_block, 2).mean(axis=0).mean())
        i += 1

    # `10*log10(signal)` is the same as `20*log10(square_root(signal))`,
    # so this does both the square root and the conversion to dB.
    # We add 10**-10 simply to prevent calculation of log(0).
    Vrms_all = 10 * np.log10(np.array(Vrms_all) + 10**-10)

    # Pick the value at 95%, calculate difference to reference and returns.
    Vrms_all.sort()
    calc_Vrms = Vrms_all[round(Vrms_all.shape[0]*0.95)]
    return calc_Vrms


RG_FILTER_COEFFS = {
    44100: {
        'a1': [
            1.00000000000000, -3.47845948550071, 6.36317777566148, -8.54751527471874,
            9.47693607801280, -8.81498681370155, 6.85401540936998, -4.39470996079559,
            2.19611684890774, -0.75104302451432, 0.13149317958808,
        ],
        'b1': [
            0.05418656406430, -0.02911007808948, -0.00848709379851, -0.00851165645469,
            -0.00834990904936, 0.02245293253339, -0.02596338512915, 0.01624864962975,
            -0.00240879051584, 0.00674613682247, -0.00187763777362,
        ],
        'a2': [1.00000000000000, -1.96977855582618, 0.97022847566350],
        'b2': [0.98500175787242, -1.97000351574484, 0.98500175787242],
    }
}


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

