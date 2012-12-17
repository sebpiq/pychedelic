import copy
import os
import numpy as np
import scipy
import pylab
from pychedelic.data_structures import Sound
from __init__ import PychedelicTestCase
dirname = os.path.dirname(__file__)

# Setup the API key for testing 
from pyechonest import config
config.ECHO_NEST_API_KEY = '3B8NIMYFZT7YALMRB'


class Sound_Test(PychedelicTestCase):

    def init_test(self):
        sound = Sound([1, 2, 3], sample_rate=44100)
        self.assertEqual(sound.channel_count, 1)
        sound = Sound([[1, 4], [2, 5], [3, 6]], sample_rate=44100)
        self.assertEqual(sound.channel_count, 2)

    def from_file_test(self):
        sound = Sound.from_file(os.path.join(dirname, 'sounds/A440_mono.wav'))
        self.assertEqual(sound.channel_count, 1)
        self.assertEqual(sound.sample_rate, 44100)
        self.assertEqual(sound.sample_count, 441)

        sound = Sound.from_file(os.path.join(dirname, 'sounds/A440_stereo.wav'))
        self.assertEqual(sound.channel_count, 2)
        self.assertEqual(sound.sample_rate, 44100)
        self.assertEqual(sound.sample_count, 441)

        sound = Sound.from_file(os.path.join(dirname, 'sounds/A440_mono.mp3'))
        self.assertEqual(sound.channel_count, 1)
        self.assertEqual(sound.sample_rate, 44100)
        self.assertTrue(sound.sample_count > 44100 and sound.sample_count < 50000)

    def to_file_test(self):
        sound = Sound.from_file(os.path.join(dirname, 'sounds/A440_mono.wav'))
        sound.to_file('/tmp/to_file_test.mp3')
        sound = Sound.from_file('/tmp/to_file_test.mp3')

        self.assertEqual(sound.channel_count, 1)
        self.assertEqual(sound.sample_rate, 44100)
        self.assertTrue(sound.sample_count >= 441)

    def mix_test(self):
        sound = Sound([[1, 0.5, 0.5], [2, 0.4, 0.4], [3, 0.3, 0.3], [4, 0.2, 0.2], [5, 0.1, 0.1], [6, 0, 0], [7, -0.1, -0.1], [8, -0.2, -0.2]], sample_rate=2)
        mixed = sound.mix()
        self.assertTrue(isinstance(mixed, Sound))
        self.assertEqual(mixed.index, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        self.assertEqual(np.round(mixed.icol(0), 4), np.round([2.0, 2.8, 3.6, 4.4, 5.2, 6.0, 6.8, 7.6], 4))

        sound = Sound([[1], [2], [3], [4], [5], [6], [7], [8]], sample_rate=2)
        mixed = sound.mix()
        self.assertTrue(isinstance(mixed, Sound))
        self.assertEqual(mixed.index, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        self.assertEqual(mixed.icol(0), [1, 2, 3, 4, 5, 6, 7, 8])

    def time_stretch_test(self):
        # length : 0.010
        sound = Sound.from_file(os.path.join(dirname, 'sounds/A440_stereo.wav'))

        stretched = sound.time_stretch(0.005)
        self.assertEqual(np.round(stretched.length, 4), 0.005)

        stretched = sound.time_stretch(0.003)
        self.assertEqual(np.round(stretched.length, 4), 0.003)

    def fade_test(self):
        sound = Sound({0: np.ones(22050), 1: np.ones(22050)}, sample_rate=44100)
        sound = sound.fade(in_dur=0.088, out_dur=0.4)
        if True or plot_opt:
            sound.plot()
            pylab.show()

    def echonest_test(self):
        sound = Sound.from_file(os.path.join(dirname, 'sounds/directions.mp3'))
        self.assertTrue(len(sound.echonest.bars) > 0)
