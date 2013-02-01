import algorithms as algos
from base_data_frames import PychedelicSampledDataFrame


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
        from sound import Sound
        times, results = algos.ifft(self.icol(0), 1.0 / self.frame_rate)
        frame_rate = 2 * (self.frame_count - 1) / self.frame_rate
        return Sound(results, frame_rate=frame_rate)


class Spectrogram(PychedelicSampledDataFrame):

    @property
    def t(self):
        return self.axes[0]

    @property
    def f(self):
        return self.axes[1]
