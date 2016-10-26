from tempfile import NamedTemporaryFile
import unittest

import scipy.io.wavfile as sp_wavfile
import numpy

from pychedelic.core import pcm, errors
from .__init__ import STEPS_MONO_RAW_16B, STEPS_STEREO_RAW_16B


def _generate_samples(channel_count):
    frame_rate = 44100
    samples = None
    for val in numpy.arange(-1, 1.1, 0.1):
        block = numpy.ones((frame_rate / 10, channel_count)) * val
        if not samples is None:
            samples = numpy.concatenate([samples, block])
        else: samples = block
    if channel_count == 2:
        samples[:, 1] *= -1
    return samples


class samples_to_string_Test(unittest.TestCase):

    def mono_test(self):
        with open(STEPS_MONO_RAW_16B, 'r') as fd:
            expected_string = fd.read()
        samples = _generate_samples(1)
        self.assertEqual(pcm.samples_to_string(samples), expected_string)

    def stereo_test(self):
        with open(STEPS_STEREO_RAW_16B, 'r') as fd:
            expected_string = fd.read()
        samples = _generate_samples(2)
        self.assertEqual(pcm.samples_to_string(samples), expected_string)        


class string_to_samples_Test(unittest.TestCase):

    def mono_test(self):
        with open(STEPS_MONO_RAW_16B, 'r') as fd:
            string = fd.read()
        expected_samples = _generate_samples(1)
        numpy.testing.assert_array_equal(
            pcm.string_to_samples(string, 1).round(4), 
            expected_samples.round(4)
        )

    def stereo_test(self):
        with open(STEPS_STEREO_RAW_16B, 'r') as fd:
            string = fd.read()
        expected_samples = _generate_samples(2)
        numpy.testing.assert_array_equal(
            pcm.string_to_samples(string, 2).round(4), 
            expected_samples.round(4)
        )

    def incomplete_string_test(self):
        with open(STEPS_MONO_RAW_16B, 'r') as fd:
            string_mono = fd.read()
        
        # We take 10 samples (2 * 10 bytes) and add one incomplete sample
        incomplete_sample_string = string_mono[0:2*10 + 1]
        self.assertRaises(errors.PcmDecodeError, pcm.string_to_samples, incomplete_sample_string, 1)

        with open(STEPS_STEREO_RAW_16B, 'r') as fd:
            string_stereo = fd.read()
        # We take 10 frames (2 * 2 * 10 bytes) and add one incomplete frame (2 * 1 byte)
        incomplete_frame_string = string_stereo[0:4*10 + 2]
        self.assertRaises(errors.PcmDecodeError, pcm.string_to_samples, incomplete_frame_string, 2)