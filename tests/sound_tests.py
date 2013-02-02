import copy
import scipy.io.wavfile as sp_wavfile
import numpy as np
import pylab
import wave

from pychedelic.sound import Sound
from __init__ import PychedelicTestCase, A440_MONO_16B, A440_STEREO_16B, A440_MONO_MP3, MILES_MP3

# Setup the API key for testing 
from pyechonest import config
config.ECHO_NEST_API_KEY = '3B8NIMYFZT7YALMRB'


class Sound_Test(PychedelicTestCase):

    def init_test(self):
        sound = Sound([1, 2, 3], frame_rate=44100)
        self.assertEqual(sound.channel_count, 1)
        sound = Sound([[1, 4], [2, 5], [3, 6]], frame_rate=44100)
        self.assertEqual(sound.channel_count, 2)

    def from_file_test(self):
        sound = Sound.from_file(A440_STEREO_16B)
        self.assertEqual(sound.channel_count, 2)
        self.assertEqual(sound.frame_rate, 44100)
        self.assertEqual(sound.frame_count, 441)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(A440_STEREO_16B)
        self.assertEqual(sound.values[0:10,:].round(4), (samples_test[:10] / float(2**15)).round(4))

        sound = Sound.from_file(A440_MONO_MP3)
        self.assertEqual(sound.values.shape[1], 1)
        self.assertEqual(sound.channel_count, 1)
        self.assertEqual(sound.frame_rate, 44100)
        self.assertTrue(sound.frame_count > 44100 and sound.frame_count < 50000)

        sound = Sound.from_file(A440_MONO_MP3, end=0.006)
        self.assertEqual(sound.channel_count, 1)
        self.assertEqual(sound.frame_rate, 44100)
        self.assertEqual(sound.frame_count, 264)

    def to_file_test(self):
        sound = Sound.from_file(A440_MONO_16B)
        sound.to_file('/tmp/to_file_test.mp3')
        sound = Sound.from_file('/tmp/to_file_test.mp3')

        self.assertEqual(sound.channel_count, 1)
        self.assertEqual(sound.frame_rate, 44100)
        self.assertTrue(sound.frame_count >= 441)

    def iter_raw_test(self):
        sound = Sound.from_file(A440_MONO_16B)
        raw_data = ''
        for chunk in sound.iter_raw(10):
            raw_data += chunk
        raw_test = wave.open(A440_MONO_16B, 'rb').readframes(100000)    
        self.assertEqual(raw_data, raw_test)

    def to_raw_test(self):
        sound = Sound.from_file(A440_MONO_16B)
        raw_data = sound.to_raw()
        raw_test = wave.open(A440_MONO_16B, 'rb').readframes(100000)    
        self.assertEqual(raw_data, raw_test)

    def mix_test(self):
        sound1 = Sound([[1, 1], [1, 1], [1, 1]], frame_rate=2)
        sound2 = Sound([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], frame_rate=2)
        sound3 = Sound([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]], frame_rate=2)
        mixed = Sound.mix(
            {'sound': sound1, 'gain': 2},
            {'sound': sound2, 'start': 0.5},
            {'sound': sound3, 'start': 0.75}
        )
        self.assertEqual(mixed.length, 2.0)
        self.assertEqual(mixed.frame_rate, 2)
        self.assertEqual(mixed.frame_count, 5)
        self.assertEqual(mixed.values, [[2, 2], [2.5, 2.5], [2.75, 2.75], [0.75, 0.75], [0.25, 0.25]])

        # Mix mono to stereo
        sound1 = Sound([[1], [1], [1]], frame_rate=2)
        sound2 = Sound([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], frame_rate=2)
        mixed = Sound.mix(
            {'sound': sound1},
            {'sound': sound2, 'start': 0.1},
        )
        self.assertEqual(mixed.length, 1.5)
        self.assertEqual(mixed.frame_rate, 2)
        self.assertEqual(mixed.frame_count, 4)
        self.assertEqual(mixed.values, [[1, 1], [1.5, 1.5], [1.5, 1.5], [0.5, 0.5]])

    def to_mono_test(self):
        sound = Sound([[1, 0.5, 0.5], [2, 0.4, 0.4], [3, 0.3, 0.3], [4, 0.2, 0.2], [5, 0.1, 0.1], [6, 0, 0], [7, -0.1, -0.1], [8, -0.2, -0.2]], frame_rate=2)
        mixed = sound.to_mono()
        self.assertTrue(isinstance(mixed, Sound))
        self.assertEqual(mixed.index, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        self.assertEqual(np.round(mixed.icol(0), 4), np.round([2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6], 4))

        sound = Sound([[1], [2], [3], [4], [5], [6], [7], [8]], frame_rate=2)
        mixed = sound.to_mono()
        self.assertTrue(isinstance(mixed, Sound))
        self.assertEqual(mixed.index, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        self.assertEqual(mixed.icol(0), [1, 2, 3, 4, 5, 6, 7, 8])

    def time_stretch_test(self):
        # length : 0.010
        sound = Sound.from_file(A440_MONO_16B)

        stretched = sound.time_stretch(0.005, algorithm='sox')
        self.assertEqual(np.round(stretched.length, 4), 0.005)

        stretched = sound.time_stretch(0.003, algorithm='sox')
        self.assertEqual(np.round(stretched.length, 4), 0.003)

    def pitch_shift_semitones_test(self):
        # TODO
        """
        sound = Sound.from_file(os.path.join(dirname, 'sounds/directions.mp3'))
        stretched = sound.pitch_shift_semitones(12)
        stretched.to_file(os.path.join(dirname, 'sounds/directions_pitched.wav'))"""
        #sound.plot()
        #stretched.plot()
        #pylab.show()
        
    def fade_test(self):
        sound = Sound({0: np.ones(22050), 1: np.ones(22050)}, frame_rate=44100)
        sound = sound.fade(in_dur=0.088, out_dur=0.4)
        if True or plot_opt:
            sound.plot()
            pylab.show()

    def echonest_test(self):
        sound = Sound.from_file(MILES_MP3)
        self.assertTrue(len(sound.echonest.bars) > 0)
