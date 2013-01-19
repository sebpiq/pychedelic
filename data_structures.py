from tempfile import NamedTemporaryFile
import math
import wave
import subprocess
import os
import pylab
import numpy as np
import pandas as pnd
import traceback
# TODO: mmaybe get rid of the scipy dependency ?
from scipy.io import wavfile

import algorithms as algos
from utils import guess_fileformat, convert_file

try:
    from pyechonest import track as echonest_track
    from pyechonest.util import EchoNestAPIError
except ImportError:
    pass


class PychedelicDataFrame(pnd.DataFrame):

    def maxima(self, i, take_edges=True):
        return algos.maxima(self[i], take_edges=take_edges)

    def minima(self, i, take_edges=True):
        return algos.minima(self[i], take_edges=take_edges)

    def smooth(self, i, window_size=11, window_func='hanning'):
        return algos.smooth(self[i], window_size=window_size, window_func=window_func)

    def convolve(self, i, array, mode='full'):
        return pnd.Series(np.convolve(self[i], array, mode=mode))

    def _constructor(self, *args, **kwargs):
        """
        This is used by `pandas` to create a new `DataFrame` when doing any operation,
        for example slicing, ...
        PB is, it is not implemented everywhere yet : https://github.com/pydata/pandas/issues/60
        """
        return self.__class__(*args, **kwargs)


class PychedelicSampledDataFrame(PychedelicDataFrame):

    def __init__(self, data, **kwargs):
        try:
            sample_rate = kwargs.pop('sample_rate')
        except KeyError:
            raise TypeError('sample_rate kwarg is required')
        if 'index' in kwargs:
            raise TypeError('index is generated automatically with sample_rate')

        super(PychedelicSampledDataFrame, self).__init__(data, **kwargs)
        self.sample_rate = sample_rate
        self.index = np.arange(0, self.shape[0]) * 1.0 / sample_rate

    @property
    def sample_count(self):
        return self.shape[0]

    def _constructor(self, *args, **kwargs):
        kwargs.setdefault('sample_rate', self.sample_rate)
        return self.__class__(*args, **kwargs)


class Sound(PychedelicSampledDataFrame):
    # TODO: when using echonest for example, no need to load the whole sound
    # from .mp3 to .wav back to .mp3 again

    def __init__(self, data, **kwargs):
        super(Sound, self).__init__(data, **kwargs)

    @property
    def t(self):
        return self.index

    @property
    def length(self):
        """
        Returns the sound length in seconds.
        """
        # The first sample is at `x = 0`, so we take `sample count - 1`
        return (self.sample_count - 1) / float(self.sample_rate)

    @property
    def echonest(self):
        if not hasattr(self, '_echonest'):
            with NamedTemporaryFile(mode='wb', delete=True, suffix='.mp3') as fd:
                self.to_file(fd.name)
                try:
                    self._echonest = echonest_track.track_from_filename(fd.name)#, force_upload=True)
                except EchoNestAPIError as exc:
                    print traceback.format_exc()
        return self._echonest

    @property
    def channel_count(self):
        return self.shape[1]

    def channel(i):
        """
        Returns the channel `i`. This is 1-based. 
        """
        return self.icol(ind - 1)

    def mix(self):
        return Sound(self.sum(1), sample_rate=self.sample_rate)

    def time_stretch(self, length):
        """
        Stretch sound to `length`.
        """
        ratio = length / self.length
        return self._constructor(algos.time_stretch(self.values, ratio))

    def pitch_shift_semitones(self, semitones):
        return self._constructor(algos.pitch_shift_semitones(self.values, semitones))

    def fade(self, in_dur=None, out_dur=None):
        # Calculate fade-in
        if in_dur is not None:
            window_size = in_dur * self.sample_rate
            fade = (np.exp(np.linspace(0, np.log(100), window_size)) - 1) / (100 - 1)
            fade_in = np.ones(self.sample_count)
            fade_in[:len(fade)] = fade

        # Calculate fade-out
        if out_dur is not None:
            window_size = out_dur * self.sample_rate
            fade = (np.exp(np.linspace(0, np.log(100), window_size)) - 1) / (100 - 1)
            fade_out = np.ones(self.sample_count)
            fade_out[-len(fade):] = fade[::-1]

        # Apply fades to all channels
        sound_data = self.values.copy()
        for channel_data in sound_data.T:
            if in_dur is not None: channel_data *= fade_in
            if out_dur is not None: channel_data *= fade_out
        return self._constructor(sound_data)

    @classmethod
    def from_file(cls, filename, fileformat=None, start=None, end=None):
        # TODO: Test with 8-bit wavs ?
        # TODO: the file might be very big, so this should be lazy
        # If the file is not .wav, we need to convert it
        converted_filename = convert_file(filename, 'wav')

        # Reading the data from the converted file
        raw = wave.open(converted_filename, 'rb')

        channels = raw.getnchannels()
        sample_width = raw.getsampwidth()       # Sample width in byte
        if sample_width == 1: np_type = 'uint8'
        elif sample_width == 2: np_type = 'uint16'
        else: raise ValueError('Wave format not supported')
        frame_rate = raw.getframerate()         
        frame_width = channels * sample_width

        if start is None: start = 0
        start_frame = start * frame_rate
        if end is None: end_frame = raw.getnframes()
        else: end_frame = end * frame_rate
        frame_count = end_frame - start_frame
        sample_count = frame_count * channels

        raw.setpos(int(start_frame))
        data = raw.readframes(int(frame_count))
        data = np.fromstring(data, dtype=np_type)
        data = data.reshape([frame_count, channels])

        # If a temp filename was created for the conversion, remove it.
        if converted_filename != filename: os.remove(converted_filename)
        sound = cls(data, sample_rate=frame_rate)
        return sound

    def to_file(self, filename):
        fileformat = guess_fileformat(filename)
        if fileformat != 'wav':
            with NamedTemporaryFile(mode='wb', delete=True, suffix='.wav') as origin_file:
                wavfile.write(origin_file.name, self.sample_rate, self.values.astype(np.int16))
                convert_file(origin_file.name, fileformat, to_filename=filename)
        else:
            wavfile.write(filename, self.sample_rate, self.values.astype(np.int16))

    def get_spectrum(self, window_func='flat'):
        """
        Returns the spectrum of the signal `data(x, v)`. Negative spectrum is removed.
        """
        mixed_sound = self.mix()
        window = algos.window(window_func, mixed_sound.sample_count)
        freqs, results = algos.fft(mixed_sound[0] * window, self.sample_rate)
        f_sample_rate = 1.0 / (freqs[1] - freqs[0])
        return Spectrum(results, sample_rate=f_sample_rate)

    def get_spectrogram(self, **kwargs):
        # TODO: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.stats.moments.rolling_apply.html#pandas.stats.moments.rolling_apply
        window_size = kwargs.get('window_size', 256)
        overlap = kwargs.get('overlap', 16)
        if overlap > window_size: raise ValueError('overlap must be less than window size')
        start = 0
        end = window_size
        offset = 0
        # Rows are t, columns are f
        spectrogram_data = np.array([])
        while(end < self.sample_count):
            sound_slice = self.ix[start:end]
            spectrum = sound_slice.get_spectrum()
            start = end - overlap
            end = start + window_size
            # Builds the data freq/time/amplitude for this window :
            # we basically copy the frequency data over all time samples on `window - overlap`.
            current_window = np.tile(np.array(spectrum.amplitudes), (window_size - overlap, 1))
            # Concatenate with previous data.
            if not spectrogram_data.size: vstack_data = (current_window,)
            else: vstack_data = (spectrogram_data, current_window)
            spectrogram_data = np.vstack(vstack_data)
        spectrogram = Spectrogram(spectrogram_data, sample_rate=self.sample_rate, columns=spectrum.index)
        return spectrogram


class Spectrum(PychedelicSampledDataFrame):

    def __init__(self, data, **kwargs):
        super(Spectrum, self).__init__(data, **kwargs)
        if self.shape[1] != 1:
            raise ValueError('Spectrum just needs one set of complex data.')
        self['amplitudes'] = algos.get_ft_amplitude_array(self.icol(0))
        self['phases'] = algos.get_ft_phase_array(self.icol(0))

    @property
    def f(self):
        return self.index

    def get_sound(self):
        """
        Performs inverse FFT to reconstruct a sound from this spectrum.
        """
        times, results = algos.ifft(self.icol(0), 1.0 / self.sample_rate)
        sample_rate = 2 * (self.sample_count - 1) / self.sample_rate
        return Sound(results, sample_rate=sample_rate)


class Spectrogram(PychedelicSampledDataFrame):

    @property
    def t(self):
        return self.axes[0]

    @property
    def f(self):
        return self.axes[1]
