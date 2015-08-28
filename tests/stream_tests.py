import os
import types
from tempfile import NamedTemporaryFile
import unittest

import numpy
import scipy.io.wavfile as sp_wavfile

from .__init__ import A440_MONO_16B, A440_STEREO_16B, STEPS_MONO_16B
from pychedelic import stream
from pychedelic import config


class ramp_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_ramp_test(self):
        config.frame_rate = 4
        config.block_size = 2
        ramp_gen = stream.ramp(1, (2, 1), (0, 1)) 
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

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def upsample1_test(self):
        """
        Testing upsampling with the 2 following configurations :
        IN:  |     |     |     .
        OUT: | | | | | | | . . .

        IN:  |     |     |     |
        OUT: . | | | | | | | . .
        """
        config.block_size = 7

        def gen():
            """
            [[0], [2], [4]] [[6], [8], [10]] ...
            """
            for i in range(0, 4):
                yield numpy.arange(i * 2 * 3, (i + 1) * 2 * 3, 2).reshape(3, 1)

        resampler = stream.resample(gen())
        resampler.set_ratio(1 / 3.0)

        # IN:  0     1     2
        # OUT: 0 1 2 3 4 5 6
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[0], [1 * 2.0/3], [2 * 2.0/3], [2], [4 * 2.0/3], [5 * 2.0/3], [4]], 8)
        )
        # IN:  2     3     4     5
        # OUT:   7 8 9 a b c d
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[7 * 2.0/3], [8 * 2.0/3], [6], [10 * 2.0/3], [11 * 2.0/3], [8], [13 * 2.0/3]], 8)
        )
        # IN:  4     5     6     7
        # OUT:     e f g h i j k
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[14 * 2.0/3], [10], [16 * 2.0/3], [17 * 2.0/3], [12], [19 * 2.0/3], [20 * 2.0/3]], 8)
        )
        # IN:  7     8     9
        # OUT: l m n o p q r
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[14], [22 * 2.0/3], [23 * 2.0/3], [16], [25 * 2.0/3], [26 * 2.0/3], [18]], 8)
        )

    def upsample2_test(self):
        """
        Here testing upsampling with the following configuration (+ testing stereo):
        IN:  |     |
        OUT: |   |   .
        """
        config.block_size = 2

        def gen():
            """
            [[0, 0], [-2, 2], [-4, 4]] [[-6, 6], [-8, 8], [-10, 10]] ...
            """
            for i in range(0, 4):
                block_in = numpy.vstack([
                    numpy.arange(-i * 2 * 3, -(i + 1) * 2 * 3, -2),
                    numpy.arange(i * 2 * 3, (i + 1) * 2 * 3, 2)
                ]).transpose()
                yield block_in

        resampler = stream.resample(gen())
        ratio = 2 / 3.0
        resampler.set_ratio(ratio)


        # IN:  0     1
        # OUT: 0   1   .
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[0, 0], [1 * -2 * ratio, 1 * 2 * ratio]], 8)
        )
        # IN:  1     2
        # OUT:   2   3
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[2 * -2 * ratio, 2 * 2 * ratio], [3 * -2 * ratio, 3 * 2 * ratio]], 8)
        )


    def downsample1_test(self):
        """
        Testing downsampling with the following configurations:
        IN:  |  |  |  |  .  .
        OUT: |      |      .

        IN:  |   |   |   |
        OUT:   |      |  
        """
        config.block_size = 2

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 10):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream.resample(gen())
        ratio = 7/3.0
        resampler.set_ratio(ratio)

        # IN:  0  1  2  3  .  .
        # OUT: 0      1      .
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[0], [1 * 0.5 * ratio]], 8)
        )
        # IN:  3  4  5  6  7
        # OUT:      2      3
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[2 * 0.5 * ratio], [3 * 0.5 * ratio]], 8)
        )
        # IN:  9  a  b  c
        # OUT:  4      5
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[4 * 0.5 * ratio], [5 * 0.5 * ratio]], 8)
        )

    def downsample2_test(self):
        """
        Testing downsampling with the following configuration:
        # IN:  | | | | | | | | |
        # OUT: |       |       |
        """
        config.block_size = 3

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 10):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream.resample(gen())
        ratio = 4
        resampler.set_ratio(ratio)

        # IN:  0 1 2 3 4 5 6 7 8
        # OUT: 0       1       2
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[0], [1 * 0.5 * ratio], [2 * 0.5 * ratio]], 8)
        )
        # ...
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[3 * 0.5 * ratio], [4 * 0.5 * ratio], [5 * 0.5 * ratio]], 8)
        )

    def ratio1_test(self):
        """
        Ratio 1 test.
        """
        config.block_size = 3

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 10):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream.resample(gen())

        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[0], [1 * 0.5], [2 * 0.5]], 8)
        )
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[3 * 0.5], [4 * 0.5], [5 * 0.5]], 8)
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
        ratio = 0.99999

        time = numpy.arange(0, frame_count) / float(config.frame_rate)
        samples = numpy.cos(2 * numpy.pi * f0 * time)
        def gen():
            yield samples.reshape(len(samples), 1)
        resampler = stream.resample(gen())
        resampler.set_ratio(ratio)
        self.assertEqual(round(zcr_f0(samples), 3), round(f0, 3))

        samples2 = next(resampler)[:,0]
        self.assertEqual(round(zcr_f0(samples2), 3), round(f0 * ratio, 3))

    def source_exhausted_test(self):
        """
        Testing when the source is exhausted:
        # IN:  | | | | | | | | | | | | x
        # OUT: |       |       |       x
        """
        config.block_size = 2

        def gen():
            """
            [[0], [0.5], [1]] [[1.5], [2], [2.5]] ...
            """
            for i in range(0, 4):
                yield numpy.arange(i * 3 * 0.5, (i + 1) * 3 * 0.5, 0.5).reshape(3, 1)

        resampler = stream.resample(gen())
        ratio = 4
        resampler.set_ratio(ratio)

        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[0], [1 * 0.5 * ratio]], 8)
        )
        numpy.testing.assert_array_equal(
            next(resampler).round(8),
            numpy.round([[2 * 0.5 * ratio], [0]], 8)
        )
        self.assertRaises(StopIteration, next, resampler)


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

        mixer = stream.mixer()
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
            [0.02],
            [0.02]
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

        mixer = stream.mixer()
        mixer.plug(source_mono())
        mixer.clock.run_after(1.5, mixer.plug, args=[source_stereo()])

        numpy.testing.assert_array_equal(next(mixer), [
            [0.01], [0.01], [0.01], [0.02]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.02], [0.02]
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

        mixer = stream.mixer()
        src1 = source_mono()
        src2 = source_stereo()
        mixer.plug(src2)
        mixer.plug(src1)
        numpy.testing.assert_array_equal(next(mixer), [
            [1 + 0.01, 1], [2 + 0.01, 2]
        ])
        mixer.unplug(src2)
        numpy.testing.assert_array_equal(next(mixer), [
            [0.01], [0.02]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0.02], [0.02]
        ])
        mixer.unplug(src1)
        self.assertRaises(StopIteration, next, mixer)

    def stop_when_empty_test(self):
        config.frame_rate = 4
        config.block_size = 2

        def source_stereo():
            for i in range(0, 3):
                yield numpy.ones((1, 2)) * 1 * (i + 1)

        mixer = stream.mixer(stop_when_empty=False)
        
        numpy.testing.assert_array_equal(next(mixer), [
            [0],
            [0]
        ])

        mixer.plug(source_stereo())
        numpy.testing.assert_array_equal(next(mixer), [
            [1, 1],
            [2, 2]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [3, 3],
            [0, 0]
        ])
        numpy.testing.assert_array_equal(next(mixer), [
            [0],
            [0]
        ])


class iter_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream.iter(samples)

        numpy.testing.assert_array_equal(next(iter_gen), [[0, 0], [1, 2]])
        numpy.testing.assert_array_equal(next(iter_gen), [[2, 4], [3, 6]])
        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8], [5, 10]])
        numpy.testing.assert_array_equal(next(iter_gen), [[6, 12], [7, 14]])
        numpy.testing.assert_array_equal(next(iter_gen), [[8, 16], [9, 18]])
        self.assertRaises(StopIteration, next, iter_gen)

    def start_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream.iter(samples, start=4.0/config.frame_rate)

        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8], [5, 10]])
        numpy.testing.assert_array_equal(next(iter_gen), [[6, 12], [7, 14]])
        numpy.testing.assert_array_equal(next(iter_gen), [[8, 16], [9, 18]])
        self.assertRaises(StopIteration, next, iter_gen)

    def end_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream.iter(samples, start=4.0/config.frame_rate, end=5.0/config.frame_rate)

        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8]])
        self.assertRaises(StopIteration, next, iter_gen)

    def pad_test(self):
        config.block_size = 2

        samples = numpy.vstack([numpy.arange(0, 10), numpy.arange(0, 10) * 2]).transpose()
        iter_gen = stream.iter(samples, start=4.0/config.frame_rate, end=5.0/config.frame_rate, pad=True)

        numpy.testing.assert_array_equal(next(iter_gen), [[4, 8], [0, 0]])
        self.assertRaises(StopIteration, next, iter_gen)


class concatenate_Test(unittest.TestCase):

    def simple_test(self):
        def source():
            for i in range(0, 3):
                yield numpy.ones([3, 1]) * i
        block = stream.concatenate(source())
        numpy.testing.assert_array_equal(block, numpy.array([
            [0], [0], [0], [1], [1], [1], [2], [2], [2]
        ]))


class read_wav_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def blocks_size_test(self):
        config.block_size = 50
        blocks = stream.read_wav(A440_STEREO_16B)
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
        blocks = stream.read_wav(A440_MONO_16B, start=0.002, end=0.004)
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
        blocks = stream.read_wav(A440_MONO_16B, start=0.002)
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
        blocks = stream.read_wav(STEPS_MONO_16B, start=1.1, end=1.4)

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


class write_wav_Test(unittest.TestCase):

    def simple_write_test(self):
        temp_file = NamedTemporaryFile()
        blocks = []

        def source():
            for i in range(0, 5):
                block = numpy.ones((44100, 1)) * i * 0.1
                blocks.append(block)
                yield block

        sink = stream.write_wav(source(), temp_file)
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

        sink = stream.write_wav(double(source()), temp_file)
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
            stream.write_wav(source(), temp_file)
        except ValueError:
            got_error = True

        self.assertTrue(got_error)
        expected = numpy.ones((44100 * 2, 2)) * 0.1
        frame_rate, actual = sp_wavfile.read(temp_file.name)
        actual = actual / float(2**15)
        self.assertEqual(actual.shape, (44100 * 2, 2))
        numpy.testing.assert_array_equal(expected.round(4), actual.round(4))
