import os
import types
from tempfile import NamedTemporaryFile
import unittest

import numpy
import scipy.io.wavfile as sp_wavfile

from __init__ import A440_MONO_16B, A440_STEREO_16B
from pychedelic import stream
from pychedelic import config


class ramp_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_ramp_test(self):
        config.frame_rate = 4
        config.block_size = 2
        ramp_gen = stream.ramp(1, (2, 1), (0, 2)) 
        numpy.testing.assert_array_equal(ramp_gen.next(), [[1], [1.25]])
        numpy.testing.assert_array_equal(ramp_gen.next(), [[1.5], [1.75]])
        numpy.testing.assert_array_equal(ramp_gen.next(), [[2], [1.75]])
        numpy.testing.assert_array_equal(ramp_gen.next(), [[1.5], [1.25]])
        numpy.testing.assert_array_equal(ramp_gen.next(), [[1], [0.75]])
        numpy.testing.assert_array_equal(ramp_gen.next(), [[0.5], [0.25]])
        numpy.testing.assert_array_equal(ramp_gen.next(), [[0]])
        self.assertRaises(StopIteration, ramp_gen.next)


class Mixer_test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def dynamic_mixing_test(self):
        config.frame_rate = 4
        config.block_size = 2

        def source_stereo1():
            for i in range(0, 3):
                yield numpy.ones((1, 2)) * 1 * (i + 1)

        def source_stereo2():
            for i in range(0, 2):
                yield numpy.ones((2, 2)) * 0.1 * (i + 1)

        def source_mono1():
            for i in range(0, 3):
                yield numpy.ones((3, 1)) * 0.01 * (i + 1)

        mixer = stream.Mixer()
        mixer.plug(source_stereo1())
        mixer.plug(source_mono1())
        numpy.testing.assert_array_equal(mixer.next(), [
            [1 + 0.01, 1],
            [2 + 0.01, 2]
        ])
        numpy.testing.assert_array_equal(mixer.next(), [
            [3 + 0.01, 3],
            [0.02, 0]
        ])
        numpy.testing.assert_array_equal(mixer.next(), [
            [0.02],
            [0.02]
        ])
        mixer.plug(source_stereo2())
        numpy.testing.assert_array_equal(mixer.next(), [
            [0.1 + 0.03, 0.1],
            [0.1 + 0.03, 0.1]
        ])
        numpy.testing.assert_array_equal(mixer.next(), [
            [0.2 + 0.03, 0.2],
            [0.2, 0.2]
        ])
        self.assertRaises(StopIteration, mixer.next)


class read_wav_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def blocks_size_test(self):
        config.block_size = 50
        blocks = stream.read_wav(A440_STEREO_16B)
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
        config.block_size = 1000
        blocks = stream.read_wav(A440_MONO_16B, start=0.002, end=0.004)
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
        config.block_size = 20
        blocks = stream.read_wav(A440_MONO_16B, start=0.002)
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

        sink = stream.to_wav_file(source(), temp_file)

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

        sink = stream.to_wav_file(double(source()), temp_file)

        expected = numpy.concatenate(blocks) * 2
        frame_rate, actual = sp_wavfile.read(temp_file.name)
        actual = actual / float(2**15)
        self.assertEqual(actual.shape, (44100 * 5, 2))
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))