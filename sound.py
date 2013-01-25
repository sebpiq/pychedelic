from tempfile import NamedTemporaryFile
import subprocess
import os
import numpy as np
import traceback

import algorithms as algos
from base_data_frames import PychedelicSampledDataFrame
from utils.files import read_wav, write_wav, guess_fileformat, convert_file

try:
    from pyechonest import track as echonest_track
    from pyechonest.util import EchoNestAPIError
except ImportError:
    pass


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
    def from_file(cls, filename, start=None, end=None):
        # If the file is not .wav, we need to convert it
        converted_filename = convert_file(filename, 'wav')

        data, infos = read_wav(converted_filename, start=start, end=end)
        # If a temp filename was created for the conversion, remove it.
        if converted_filename != filename: os.remove(converted_filename)
        sound = cls(data, sample_rate=infos['sample_rate'])
        return sound

    def to_file(self, filename):
        fileformat = guess_fileformat(filename)
        if fileformat != 'wav':
            with NamedTemporaryFile(mode='wb', delete=True, suffix='.wav') as origin_file:
                write_wav(origin_file, self.values, sample_rate=self.sample_rate)
                convert_file(origin_file.name, fileformat, to_filename=filename)
        else:
            write_wav(filename, self.values, sample_rate=self.sample_rate)

    def get_spectrum(self, window_func='flat'):
        """
        Returns the spectrum of the signal `data(x, v)`. Negative spectrum is removed.
        """
        from spectrum import Spectrum
        mixed_sound = self.mix()
        window = algos.window(window_func, mixed_sound.sample_count)
        freqs, results = algos.fft(mixed_sound[0] * window, self.sample_rate)
        f_sample_rate = 1.0 / (freqs[1] - freqs[0])
        return Spectrum(results, sample_rate=f_sample_rate)

    def get_spectrogram(self, **kwargs):
        # TODO: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.stats.moments.rolling_apply.html#pandas.stats.moments.rolling_apply
        from spectrum import Spectrogram
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
