import unittest

import numpy

from pychedelic import block


class fix_channels_Test(unittest.TestCase):
    
    def identity_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(block.fix_channels(samples, 2), samples)

    def up_mix_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        up_mixed_samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(block.fix_channels(samples, 3), up_mixed_samples)

    def down_mix_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        down_mixed_samples = numpy.array([[0, 1, 2, 3, 4]]).transpose()
        numpy.testing.assert_array_equal(block.fix_channels(samples, 1), down_mixed_samples)