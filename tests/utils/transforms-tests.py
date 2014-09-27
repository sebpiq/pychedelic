from unittest import TestCase

import numpy as np

from pychedelic.utils import transforms
from __init__ import PychedelicTestCase


class fix_channels_Test(PychedelicTestCase):
    
    def identity_test(self):
        samples = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        self.assertEqual(transforms.fix_channels(samples, 2), samples)

    def up_mix_test(self):
        samples = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        up_mixed_samples = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]).transpose()
        self.assertEqual(transforms.fix_channels(samples, 3), up_mixed_samples)

    def down_mix_test(self):
        samples = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        down_mixed_samples = np.array([[0, 1, 2, 3, 4]]).transpose()
        self.assertEqual(transforms.fix_channels(samples, 1), down_mixed_samples)