import copy
import numpy as np
import pylab
from pychedelic.base_data_frames import PychedelicSampledDataFrame
from pychedelic.sound import Sound
from __init__ import PychedelicTestCase, plot_opt


class Spectrum_Test(PychedelicTestCase):

    def get_spectrum_test(self):
        t = np.linspace(0, 1, 44100)
        sine_wave = np.sin(2*np.pi*440*t)
        sound = Sound(sine_wave, frame_rate=44100)

        spectrum = sound.get_spectrum(window_func='flat')
        maxima = spectrum.maxima('amplitudes')
        self.assertEqual(maxima.index, [440])

    def get_sound_test(self):
        frame_rate = 44100.0
        t = np.linspace(0, 1.0 / 2, frame_rate / 2)
        sine_wave = np.sin(2*np.pi*440*t)
        sound = Sound(sine_wave, frame_rate=frame_rate)
        spectrum = sound.get_spectrum()

        # Reconstructing sound wave from spectrum
        reconstructed = spectrum.get_sound()
        self.assertTrue(isinstance(reconstructed, Sound))
        self.assertEqual(sound.frame_rate, frame_rate)
        self.assertEqual(np.round(sine_wave, 4), np.round(reconstructed.icol(0), 4))

        if plot_opt:
            sound = Sound(sine_wave[:300], frame_rate=44100)
            reconstructed = reconstructed[:300].plot()
            pylab.subplot(2, 1, 1)
            pylab.title('Original signal')
            sound.plot()
            pylab.subplot(2, 1, 2)
            pylab.title('Reconstructed signal')
            reconstructed.plot()
            pylab.show()


class Spectrogram_Test(PychedelicTestCase):

    def get_spectrogram_test(self):
        frame_rate = 44100.0
        t1 = np.linspace(0, 0.1, frame_rate / 10)
        sine_wave1 = np.sin(2*np.pi*440*t1)
        t2 = np.linspace(0.1, 0.2, frame_rate / 10)
        sine_wave2 = np.sin(2*np.pi*220*t2)

        sound_data = np.array([np.hstack((sine_wave1, sine_wave2))]).transpose()
        sine_wave = Sound(data=sound_data, frame_rate=frame_rate)
        spectrogram = sine_wave.get_spectrogram(window_size=1024, overlap=0)
        limited_spectrogram = PychedelicSampledDataFrame({
            220: spectrogram[215.33203125], 440: spectrogram[430.6640625]},
            frame_rate=frame_rate)

        if plot_opt:
            limited_spectrogram.plot()
            pylab.show()
