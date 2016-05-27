import os
import types
from tempfile import TemporaryFile, NamedTemporaryFile
import unittest

import numpy
import scipy.io.wavfile as sp_wavfile

from .__init__ import A440_MONO_16B, A440_STEREO_16B, STEPS_MONO_16B
from pychedelic import stream_functions
from pychedelic import config
from pychedelic.core import wav as core_wav


class ramp_test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_ramp_test(self):
        config.frame_rate = 4
        config.block_size = 2
        ramp_gen = stream_functions.ramp(1, (2, 1), (0, 1)) 
        numpy.testing.assert_array_equal(
            numpy.round(next(ramp_gen), 4), 
            numpy.round([[1], [1.33333]], 4)
        )
        numpy.testing.assert_array_equal(
            numpy.round(next(ramp_gen), 4),
            numpy.round([[1.66666], [2]], 4)
        )
        numpy.testing.assert_array_equal(
            numpy.round(next(ramp_gen), 4),
            numpy.round([[2], [1.33333]], 4)
        )
        numpy.testing.assert_array_equal(
            numpy.round(next(ramp_gen), 4),
            numpy.round([[0.66666], [0]], 4)
        )
        self.assertRaises(StopIteration, next, ramp_gen)


class resampler_test(unittest.TestCase):

    def upsample1_test(self):
        """
        Testing upsampling with the following configurations :
        IN:  |     |     |
        OUT: | | | | | | |
        """

        def gen():
            """
            [[0], [2], [4]] [[6], [8], [10]] ...
            """
            for i in range(0, 2):
                yield numpy.arange(i * 2 * 3, (i + 1) * 2 * 3, 2).reshape(3, 1)

        resampler = stream_functions.resample(gen())
        resampler.set_ratio(3.0)

        # IN:  0     1     2     3     4     5
        # OUT: 0 1 2 3 4 5 6 7 8 9 a b c d e f
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(resampler).round(8),
            (numpy.array([
                [0], [1], [2], [3], [4], [5], [6], [7], [8], 
                [9], [10], [11], [12], [13], [14], [15]
            ]) * 2.0 / 3).round(8)
        )

    def upsample2_test(self):
        """
        Here testing upsampling with the following configurations (+ testing stereo):
        IN:  |     |     |     |
        OUT: |   |   |   |   |
        """

        def gen():
            """
            [[0, 0], [-2, 2], [-4, 4]] [[-6, 6], [-8, 8], [-10, 10]] ...
            """
            for i in range(0, 2):
                block_in = numpy.vstack([
                    numpy.arange(-i * 2 * 3, -(i + 1) * 2 * 3, -2),
                    numpy.arange(i * 2 * 3, (i + 1) * 2 * 3, 2)
                ]).transpose()
                yield block_in

        resampler = stream_functions.resample(gen())
        ratio = 3.0 / 2
        resampler.set_ratio(ratio)


        # IN:  0     1     2     3     4     5
        # OUT: 0   1   2   3   4   5   6   7
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(resampler).round(8),
            (numpy.array([
                [0, 0], [-1, 1], [-2, 2], [-3, 3], 
                [-4, 4], [-5, 5], [-6, 6], [-7, 7]
            ]) * 2 / ratio).round(8)
        )


    def downsample1_test(self):
        """
        Testing downsampling with the following configurations:
        IN:  |  |  |  |  |  |
        OUT: |      |      |
        """

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 5):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream_functions.resample(gen())
        ratio = 3.0 / 7
        resampler.set_ratio(ratio)

        # IN:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        # OUT: 0      1      2      3      4      5      6
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(resampler).round(8),
            (numpy.array([[0], [1], [2], [3], [4], [5]]) * (0.5 / ratio)).round(8)
        )

    def downsample2_test(self):
        """
        Testing downsampling with the following configurations:
        # IN:  | | | | | | | | |
        # OUT: |       |       |
        """

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 3):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream_functions.resample(gen())
        ratio = 1.0 / 4
        resampler.set_ratio(ratio)

        # IN:  0 1 2 3 4 5 6 7 8
        # OUT: 0       1       2
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(resampler).round(8),
            (numpy.array([[0], [1], [2]]) * 0.5 / ratio).round(8)
        )

    def downsample3_test(self):
        """
        Testing high downsampling, several blocks of incoming data fetched for one frame out.
        """

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 6):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream_functions.resample(gen())
        ratio = 1.0 / 8
        resampler.set_ratio(ratio)

        # IN:  0 1 2 3 4 5 6 7 8 9 a b c d e f g h
        # OUT: 0               1               2
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(resampler).round(8),
            (numpy.array([[0], [1], [2]]) * (0.5 / ratio)).round(8)
        )

    def ratio1_test(self):
        """
        Ratio 1 test.
        """

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 2):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream_functions.resample(gen())

        numpy.testing.assert_array_equal(
            stream_functions.concatenate(resampler).round(8),
            (numpy.array([[0], [1], [2], [3], [4], [5]]) * 0.5).round(8)
        )

    def sanity_check_test(self):
        """
        Test that something's not fundamentally wrong.
        """
        def zcr_f0(samples):
            """
            Calculate frequency using zero-crossings method.
            """
            frame_rate = config.frame_rate
            frame_count = len(samples)
            
            crossings = (numpy.diff(numpy.sign(samples)) != 0)
            time = (numpy.ones(frame_count) / frame_rate).cumsum() - 1 / frame_rate
            half_oscillation_times = numpy.diff(time[crossings])
            self.assertTrue(half_oscillation_times.std() < 0.00005)
            return 0.5 / half_oscillation_times.mean()

        frame_count = 44100 * 20
        config.block_size = frame_count

        f0 = 440
        ratio = 1/0.99999

        time = numpy.arange(0, frame_count) / float(config.frame_rate)
        samples = numpy.cos(2 * numpy.pi * f0 * time)
        def gen():
            yield samples.reshape(len(samples), 1)
        resampler = stream_functions.resample(gen())
        resampler.set_ratio(ratio)
        self.assertEqual(round(zcr_f0(samples), 3), round(f0, 3))

        samples2 = next(resampler)[:,0]
        self.assertEqual(round(zcr_f0(samples2), 3), round(f0 / ratio, 3))


class mixer_test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def dynamic_plug_test(self):
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

        mixer = stream_functions.mixer(2)
        mixer.plug(source_stereo1())
        mixer.plug(source_mono1())
        numpy.testing.assert_array_equal(next(mixer), [
            [1 + 0.01, 1],
            [2 + 0.01, 2]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [3 + 0.01, 3],
            [0.02, 0]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.02, 0],
            [0.02, 0]
        ])
        mixer.plug(source_stereo2())
        numpy.testing.assert_array_equal(next(mixer), [
            [0.1 + 0.03, 0.1],
            [0.1 + 0.03, 0.1]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.2 + 0.03, 0.2],
            [0.2, 0.2]
        ])
        self.assertRaises(StopIteration, next, mixer)

    def schedule_plug_test(self):
        config.frame_rate = 4
        config.block_size = 4

        def source_stereo():
            for i in range(0, 2):
                yield numpy.ones((2, 2)) * 0.1 * (i + 1)

        def source_mono():
            for i in range(0, 3):
                yield numpy.ones((3, 1)) * 0.01 * (i + 1)

        mixer = stream_functions.mixer(2)
        mixer.plug(source_mono())
        mixer.clock.run_after(1.5, mixer.plug, args=[source_stereo()])

        numpy.testing.assert_array_equal(next(mixer), [
            [0.01, 0], [0.01, 0], [0.01, 0], [0.02, 0]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.02, 0], [0.02, 0]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.1 + 0.03, 0.1],
            [0.1 + 0.03, 0.1],
            [0.2 + 0.03, 0.2],
            [0.2, 0.2]
        ])
        self.assertRaises(StopIteration, next, mixer)

    def unplug_test(self):
        config.frame_rate = 4
        config.block_size = 2

        def source_stereo():
            for i in range(0, 3):
                yield numpy.ones((1, 2)) * 1 * (i + 1)

        def source_mono():
            for i in range(0, 3):
                yield numpy.ones((3, 1)) * 0.01 * (i + 1)

        mixer = stream_functions.mixer(2)
        src1 = source_mono()
        src2 = source_stereo()
        mixer.plug(src2)
        mixer.plug(src1)
        numpy.testing.assert_array_equal(next(mixer), [
            [1 + 0.01, 1], [2 + 0.01, 2]
        ])
        mixer.unplug(src2)
        numpy.testing.assert_array_equal(next(mixer), [
            [0.01, 0], [0.02, 0]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.02, 0], [0.02, 0]
        ])
        mixer.unplug(src1)
        self.assertRaises(StopIteration, next, mixer)

    def stop_when_empty_test(self):
        config.frame_rate = 4
        config.block_size = 2

        def source_mono():
            for i in range(0, 3):
                yield numpy.ones((1, 1)) * 1 * (i + 1)

        mixer = stream_functions.mixer(1, stop_when_empty=False)
        
        numpy.testing.assert_array_equal(next(mixer), [
            [0],
            [0]
        ])

        mixer.plug(source_mono())
        numpy.testing.assert_array_equal(next(mixer), [
            [1],
            [2]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [3],
            [0]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0],
            [0]
        ])


class window_test(unittest.TestCase):

    def no_pad_test(self):
        def gen():
            yield numpy.array([[0, 0]])
            yield numpy.array([[1, 1]])
            yield numpy.array([[2, 2]])
        window = stream_functions.window(gen(), 2, 1, pad=False)
        numpy.testing.assert_array_equal(next(window), [[0, 0], [1, 1]])
        numpy.testing.assert_array_equal(next(window), [[1, 1], [2, 2]])
        self.assertRaises(StopIteration, next, window)

    def no_pad_decimal_hop_size_test(self):
        def gen():
            yield numpy.array([[0, 0]])
            yield numpy.array([[1, 1]])
            yield numpy.array([[2, 2]])
        window = stream_functions.window(gen(), 2, 0.5, pad=False)
        # read position = 0
        numpy.testing.assert_array_equal(next(window), [[0, 0], [1, 1]])
        # read position = 0.5
        numpy.testing.assert_array_equal(next(window), [[0, 0], [1, 1]])
        # read position = 1.0       
        numpy.testing.assert_array_equal(next(window), [[1, 1], [2, 2]])
        # read position = 1.5
        numpy.testing.assert_array_equal(next(window), [[1, 1], [2, 2]])
        # read position = 2
        self.assertRaises(StopIteration, next, window)

    def pad_test(self):
        def gen():
            yield numpy.array([[0, 0]])
            yield numpy.array([[1, 1]])
            yield numpy.array([[2, 2]])
        window = stream_functions.window(gen(), 3, 1)
        numpy.testing.assert_array_equal(next(window), [[0, 0], [1, 1], [2, 2]])
        numpy.testing.assert_array_equal(next(window), [[1, 1], [2, 2], [0, 0]])
        numpy.testing.assert_array_equal(next(window), [[2, 2], [0, 0], [0, 0]])
        self.assertRaises(StopIteration, next, window)

    def win_size_exact_and_pad_test(self):
        """
        Test when padding is True, and last window falls exactly, without actual need for padding. 
        """
        def gen():
            for i in range(2):
                yield numpy.array([[i * 11, i * 11]])
        window = stream_functions.window(gen(), 1, 1, pad=True)
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(window), 
            [[0, 0], [11, 11]]
        )

    def overlap_cut_test(self):
        """
        Test overlap with pulled blocks smaller than source blocks.
        """
        def gen():
            yield numpy.tile(numpy.arange(0, 6), (2, 1)).transpose()
        window = stream_functions.window(gen(), 3, 1)
        blocks = [next(window) for i in range(0, 4)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]], [[3, 3], [4, 4], [5, 5]],
        ])

    def overlap_concatenate_test(self):
        """
        Test overlap with pulled blocks bigger than source blocks.
        """
        def gen():
            for i in range(6):
                yield numpy.array([[i * 11, i * 11]])
        window = stream_functions.window(gen(), 2, 1)
        blocks = [next(window) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [11, 11]], [[11, 11], [22, 22]],
            [[22, 22], [33, 33]], [[33, 33], [44, 44]],
            [[44, 44], [55, 55]]
        ])

    def overlap_almost_static_test(self):
        """
        Test with such a big overlap that same block is returned several times
        """
        def gen():
            for i in range(6):
                yield numpy.array([[i * 11]])
        window = stream_functions.window(gen(), 3, 0.5)
        numpy.testing.assert_array_equal(next(window), [[0], [11], [22]])
        numpy.testing.assert_array_equal(next(window), [[0], [11], [22]])
        numpy.testing.assert_array_equal(next(window), [[11], [22], [33]])

    def hop_size_bigger_than_win_size_test(self):
        def gen():
            for i in range(6):
                yield numpy.array([[i * 11, i * 11]])
        window = stream_functions.window(gen(), 2, 3, pad=True)
        numpy.testing.assert_array_equal(
            stream_functions.concatenate(window), 
            [[0, 0], [11, 11], [33, 33], [44, 44]]
        )


class iter_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream_functions.iter(samples)

        numpy.testing.assert_array_equal(next(iter_gen), [[0, 0], [1, 2]])
        numpy.testing.assert_array_equal(next(iter_gen), [[2, 4], [3, 6]])
        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8], [5, 10]])
        numpy.testing.assert_array_equal(next(iter_gen), [[6, 12], [7, 14]])
        numpy.testing.assert_array_equal(next(iter_gen), [[8, 16], [9, 18]])
        self.assertRaises(StopIteration, next, iter_gen)

    def start_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream_functions.iter(samples, start=4.0/config.frame_rate)

        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8], [5, 10]])
        numpy.testing.assert_array_equal(next(iter_gen), [[6, 12], [7, 14]])
        numpy.testing.assert_array_equal(next(iter_gen), [[8, 16], [9, 18]])
        self.assertRaises(StopIteration, next, iter_gen)

    def end_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream_functions.iter(samples, start=4.0/config.frame_rate, end=5.0/config.frame_rate)

        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8]])
        self.assertRaises(StopIteration, next, iter_gen)

    def pad_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream_functions.iter(samples, start=4.0/config.frame_rate, end=5.0/config.frame_rate, pad=True)

        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8], [0, 0]])
        self.assertRaises(StopIteration, next, iter_gen)


class concatenate_Test(unittest.TestCase):

    def simple_test(self):
        def source():
            for i in range(0, 3):
                yield numpy.ones([3, 1]) * i
        block = stream_functions.concatenate(source())
        numpy.testing.assert_array_equal(block, numpy.array([
            [0], [0], [0], [1], [1], [1], [2], [2], [2]
        ]))


class read_wav_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def blocks_size_test(self):
        config.block_size = 50
        blocks = stream_functions.read_wav(A440_STEREO_16B)
        self.assertEqual(blocks.infos['frame_rate'], 44100)
        self.assertEqual(blocks.infos['channel_count'], 2)

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
        blocks = stream_functions.read_wav(A440_MONO_16B, start=0.002, end=0.004)
        self.assertEqual(blocks.infos['frame_rate'], 44100)
        self.assertEqual(blocks.infos['channel_count'], 1)

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
        blocks = stream_functions.read_wav(A440_MONO_16B, start=0.002)
        self.assertEqual(blocks.infos['frame_rate'], 44100)
        self.assertEqual(blocks.infos['channel_count'], 1)

        blocks = list(blocks)
        self.assertEqual([len(b) for b in blocks], [20] * 17 + [13])
        self.assertEqual(blocks[0].shape, (20, 1))

        actual = numpy.concatenate(blocks)
        frame_rate, expected = sp_wavfile.read(A440_MONO_16B)
        expected = numpy.array([expected[0.002*44100:] / float(2**15)]).transpose()
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))

    def seek_test(self):
        config.block_size = 441
        blocks = stream_functions.read_wav(STEPS_MONO_16B, start=1.1, end=1.4)

        self.assertEqual(blocks.infos['frame_rate'], 44100)
        self.assertEqual(blocks.infos['channel_count'], 1)

        expected = numpy.ones([441, 1]) * 0.1
        samples = next(blocks)
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

        blocks.seek(1.3)
        expected = numpy.ones([441, 1]) * 0.3
        samples = next(blocks)
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

        blocks.seek(1.3)
        expected = numpy.ones([441, 1]) * 0.3
        samples = next(blocks)
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

        blocks.seek(0)
        expected = numpy.ones([441, 1]) * -1
        samples = next(blocks)
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

    def read_invalid_wav_test(self):
        # __file__ is obviously not a wav file ...
        self.assertRaises(core_wav.FormatError, stream_functions.read_wav, __file__)


class write_wav_Test(unittest.TestCase):

    def simple_write_test(self):
        temp_file = NamedTemporaryFile()
        blocks = []

        def source():
            for i in range(0, 5):
                block = numpy.ones((44100, 1)) * i * 0.1
                blocks.append(block)
                yield block

        sink = stream_functions.write_wav(source(), temp_file)
        self.assertEqual(sink.infos['frame_rate'], 44100)
        self.assertEqual(sink.infos['channel_count'], 1)

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
                yield next(source) * 2

        sink = stream_functions.write_wav(double(source()), temp_file)
        self.assertEqual(sink.infos['frame_rate'], 44100)
        self.assertEqual(sink.infos['channel_count'], 2)

        expected = numpy.concatenate(blocks) * 2
        frame_rate, actual = sp_wavfile.read(temp_file.name)
        actual = actual / float(2**15)
        self.assertEqual(actual.shape, (44100 * 5, 2))
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))

    def write_incorrect_channel_count_test(self):
        temp_file = NamedTemporaryFile()
        got_error = False

        def source():
            yield numpy.ones((44100, 2)) * 0.1
            yield numpy.ones((44100, 2)) * 0.1
            yield numpy.ones((44100, 1)) * 0.1

        try:
            stream_functions.write_wav(source(), temp_file)
        except ValueError:
            got_error = True

        self.assertTrue(got_error)
        expected = numpy.ones((44100 * 2, 2)) * 0.1
        frame_rate, actual = sp_wavfile.read(temp_file.name)
        actual = actual / float(2**15)
        self.assertEqual(actual.shape, (44100 * 2, 2))
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))

    @unittest.skip('temporarily disabled cause too slow')
    def reach_wav_size_limit_test(self):
        temp_file = TemporaryFile('w')
        
        def source():
            while True:
                yield numpy.zeros((2**20, 1))

        got_error = False
        try:
            stream_functions.write_wav(source(), temp_file)
        except core_wav.WavSizeLimitError:
            got_error = True 
        self.assertTrue(got_error)
