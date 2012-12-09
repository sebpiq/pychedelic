import copy
import numpy as np
import pylab
from pychedelic.data_structures import Spectrum, Sound
from __init__ import PychedelicTestCase, plot_opt


class Spectrum_Test(PychedelicTestCase):

    def get_spectrum_test(self):
        t = np.linspace(0, 1, 44100)
        sine_wave = np.sin(2*np.pi*440*t)
        sound = Sound(sine_wave, sample_rate=44100)

        spectrum = sound.get_spectrum(window_func='flat')
        maxima = spectrum.maxima('amplitudes')
        self.assertEqual(maxima.index, [440])

    def get_sound_test(self):
        SAMPLE_RATE = 44100.0
        t = np.linspace(0, 1.0 / 2, SAMPLE_RATE / 2)
        sine_wave = np.sin(2*np.pi*440*t)
        sound = Sound(sine_wave, sample_rate=SAMPLE_RATE)
        spectrum = sound.get_spectrum()

        # Reconstructing sound wave from spectrum
        reconstructed = spectrum.get_sound()
        self.assertTrue(isinstance(reconstructed, Sound))
        self.assertEqual(sound.sample_rate, SAMPLE_RATE)
        self.assertEqual(np.round(sine_wave, 4), np.round(reconstructed.icol(0), 4))

        if plot_opt:
            sound = Sound(sine_wave[:300], sample_rate=44100)
            reconstructed = reconstructed[:300].plot()
            pylab.subplot(2, 1, 1)
            pylab.title('Original signal')
            sound.plot()
            pylab.subplot(2, 1, 2)
            pylab.title('Reconstructed signal')
            reconstructed.plot()
            pylab.show()
