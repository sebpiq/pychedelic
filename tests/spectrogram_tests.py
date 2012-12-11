import copy
import numpy as np
import pylab
from pychedelic.data_structures import Sound, PychedelicSampledDataFrame
from __init__ import PychedelicTestCase, plot_opt


class Spectrogram_Test(PychedelicTestCase):

    def get_spectrogram_test(self):
        SAMPLE_RATE = 44100.0
        t1 = np.linspace(0, 0.1, SAMPLE_RATE / 10)
        sine_wave1 = np.sin(2*np.pi*440*t1)
        t2 = np.linspace(0.1, 0.2, SAMPLE_RATE / 10)
        sine_wave2 = np.sin(2*np.pi*220*t2)

        sound_data = np.array([np.hstack((sine_wave1, sine_wave2))]).transpose()
        sine_wave = Sound(data=sound_data, sample_rate=SAMPLE_RATE)
        spectrogram = sine_wave.get_spectrogram(window_size=1024, overlap=0)
        limited_spectrogram = PychedelicSampledDataFrame({220: spectrogram[215.33203125], 440: spectrogram[430.6640625]}, sample_rate=SAMPLE_RATE)

        if plot_opt:
            limited_spectrogram.plot()
            pylab.show()
