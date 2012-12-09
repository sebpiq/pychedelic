from tempfile import NamedTemporaryFile
import math
import subprocess
import os
import pylab
import numpy as np
import pandas as pnd
from scipy.io import wavfile
import algorithms as algos


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
    def channel_count(self):
        return self.shape[1]

    def channel(i):
        """
        Returns the channel `i`. This is 1-based. 
        """
        return self.icol(ind - 1)

    def mix(self):
        return Sound(self.sum(1), sample_rate=self.sample_rate)

    @classmethod
    def from_file(cls, filename, fileformat=None):
        # TODO: the file might be very big, so this should be lazy
        # Get the format of the file
        try:
            fileformat = filename.split('.')[-1]
        except IndexError:
            raise ValueError('unknown file format')

        # If the file is not .wav, we need to convert it
        if fileformat != 'wav':

            # Copying source file to a temporary file
            origin_file = NamedTemporaryFile(mode='wb', delete=False)
            with open(filename, 'r') as fd:
                while True:
                    copy_buffer = fd.read(1024*1024)
                    if copy_buffer: origin_file.write(copy_buffer)
                    else: break
            origin_file.flush()

            # Converting the file to wav
            dest_file = NamedTemporaryFile(mode='rb', delete=False)
            ffmpeg_call = ['ffmpeg', '-y',
                            '-f', fileformat,
                            '-i', origin_file.name,  # input options (filename last)
                            '-vn',  # Drop any video streams if there are any
                            '-f', 'wav',  # output options (filename last)
                            dest_file.name
                          ]
            subprocess.check_call(ffmpeg_call, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
            filename = dest_file.name

            # Closing file descriptors, removing files
            origin_file.close()
            os.unlink(origin_file.name)

        # Finally create sound 
        sample_rate, data = wavfile.read(filename)
        if len(data.shape) == 1:
            data = np.array([data]).transpose()
        sound = cls(data, sample_rate=sample_rate)

        # Cleaning
        if fileformat != 'wav':
            dest_file.close()
            os.unlink(dest_file.name)

        return sound

    def to_file(self, filename):
        wavfile.write(filename, self.sample_rate, self.astype(np.int16))

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
