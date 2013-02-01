import sys, os
modpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(modpath)
import unittest
import numpy as np


plot_opt = True

sounddir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sounds'))
A440_MONO_16B = os.path.join(sounddir, 'A440_mono_16B.wav')
A440_STEREO_16B = os.path.join(sounddir, 'A440_stereo_16B.wav')
A440_MONO_MP3 = os.path.join(sounddir, 'A440_mono.mp3')
MILES_MP3 = os.path.join(sounddir, 'directions.mp3')

class PychedelicTestCase(unittest.TestCase):

    def assertEqual(self, first, second):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            first = np.array(first)
            second = np.array(second)
            super(PychedelicTestCase, self).assertEqual(first.shape, second.shape)
            ma = (first == second)
            if isinstance(ma, bool):
                return self.assertTrue(ma)            
            return self.assertTrue(ma.all())
        else:
            return super(PychedelicTestCase, self).assertEqual(first, second)
