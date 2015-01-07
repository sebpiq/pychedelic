import unittest

import numpy
import scipy.io.wavfile as sp_wavfile

from pychedelic import chunk
from pychedelic import config

from .__init__ import STEPS_STEREO_16B


class ramp_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_ramp_test(self):
        config.frame_rate = 4
        config.block_size = 2
        ramp_samples = chunk.ramp(1, (2, 1), (0, 1))
        numpy.testing.assert_array_equal(numpy.round(ramp_samples, 4), numpy.round([
            [1], [1.33333], [1.66666], [2],
            [2], [1.33333], [0.66666], [0]
        ], 4))

    def rounding_duration_test(self):
        """
        Because the durations for each ramp are rounded, we might need to adjust the number
        of samples to make it right.
        """
        ramp_samples = chunk.ramp(1, (1, 0.004), (1, 0.012), (0, 0.004))
        self.assertEqual(ramp_samples.shape, (int(0.02 * 44100), 1))


class fix_channel_count_Test(unittest.TestCase):
    
    def identity_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_channel_count(samples, 2), samples)

    def up_mix_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        up_mixed_samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_channel_count(samples, 3), up_mixed_samples)

    def down_mix_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        down_mixed_samples = numpy.array([[0, 1, 2, 3, 4]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_channel_count(samples, 1), down_mixed_samples)


class fix_frame_count_Test(unittest.TestCase):

    def identity_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_frame_count(samples, 5), samples)
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_frame_count(samples, -5), samples)

    def pad_test(self):
        samples = numpy.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]).transpose()
        padded_samples = numpy.array([[1, 2, 3, 4, 5, 0, 0], [5, 6, 7, 8, 9, 0, 0]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_frame_count(samples, 7), padded_samples)

    def crop_test(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        cropped_samples = numpy.array([[0, 1, 2], [5, 6, 7]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_frame_count(samples, 3), cropped_samples)

    def pad_test_before(self):
        samples = numpy.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]).transpose()
        padded_samples = numpy.array([[0, 0, 1, 2, 3, 4, 5], [0, 0, 5, 6, 7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_frame_count(samples, -7), padded_samples)

    def crop_test_before(self):
        samples = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).transpose()
        cropped_samples = numpy.array([[2, 3, 4], [7, 8, 9]]).transpose()
        numpy.testing.assert_array_equal(chunk.fix_frame_count(samples, -3), cropped_samples)


class read_wav_Test(unittest.TestCase):

    def simple_file_test(self):
        samples, infos = chunk.read_wav(STEPS_STEREO_16B)

        self.assertEqual(samples.shape, (92610, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(open(STEPS_STEREO_16B, 'r'))
        numpy.testing.assert_array_equal(samples[:10,:].round(4), (samples_test[:10,:] / float(2**15)).round(4))
