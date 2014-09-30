import os
import types
from tempfile import NamedTemporaryFile
import unittest

import numpy
import scipy.io.wavfile as sp_wavfile

from __init__ import A440_MONO_16B, A440_STEREO_16B
from pychedelic import generators
from pychedelic.utils import wav


class read_wav_Test(unittest.TestCase):

    def blocks_size_test(self):
        blocks = generators.read_wav(A440_STEREO_16B, block_size=50)
        self.assertEqual(type(blocks), types.GeneratorType)
        blocks = list(blocks)
        self.assertEqual([len(b) for b in blocks], [50, 50, 50, 50, 50, 50, 50, 50, 41])
        self.assertEqual(blocks[0].shape, (50, 2))

        actual = numpy.concatenate(blocks)
        frame_rate, expected = sp_wavfile.read(A440_STEREO_16B)
        expected = expected / float(2**15)
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))

    def block_size_bigger_than_slice_to_read_test(self):
        """
        Read only a segment of the file, block_size bigger than segment to read.
        """
        blocks = generators.read_wav(A440_MONO_16B, start=0.002, end=0.004, block_size=1000)
        self.assertEqual(type(blocks), types.GeneratorType)
        blocks = list(blocks)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].shape, (88, 1))

        actual = numpy.concatenate(blocks)
        frame_rate, expected = sp_wavfile.read(A440_MONO_16B)
        expected = numpy.array([expected[0.002*44100:0.004*44100] / float(2**15)]).transpose()
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))

    def last_block_too_small_test(self):
        """
        Ommit end, not an exact count of block_size.
        """
        blocks = generators.read_wav(A440_MONO_16B, start=0.002, block_size=20)
        self.assertEqual(type(blocks), types.GeneratorType)
        blocks = list(blocks)
        self.assertEqual([len(b) for b in blocks], [20] * 17 + [13])
        self.assertEqual(blocks[0].shape, (20, 1))

        actual = numpy.concatenate(blocks)
        frame_rate, expected = sp_wavfile.read(A440_MONO_16B)
        expected = numpy.array([expected[0.002*44100:] / float(2**15)]).transpose()
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))


class to_wav_file_Test(unittest.TestCase):

    def simple_write_test(self):
        temp_file = NamedTemporaryFile()
        blocks = []

        def source():
            for i in range(0, 5):
                block = numpy.ones((44100, 1)) * i * 0.1
                blocks.append(block)
                yield block

        sink = generators.to_wav_file(source(), temp_file, 44100)
        #list(sink) # pull audio

        expected = numpy.concatenate(blocks)
        frame_rate, actual = sp_wavfile.read(temp_file.name)
        actual = numpy.array([actual / float(2**15)]).transpose()
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))


    def chain_test(self):
        """
        Test that if one generator raises StopIteration up the chain, the sink catches it.
        """
        temp_file = NamedTemporaryFile()
        blocks = []

        def source():
            for i in range(0, 5):
                block = numpy.ones((44100, 2)) * i * 0.1
                blocks.append(block)
                yield block

        def double(source):
            while True:
                yield source.next() * 2

        sink = generators.to_wav_file(double(source()), temp_file, 44100)

        expected = numpy.concatenate(blocks) * 2
        frame_rate, actual = sp_wavfile.read(temp_file.name)
        actual = actual / float(2**15)
        self.assertEqual(actual.shape, (44100 * 5, 2))
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))