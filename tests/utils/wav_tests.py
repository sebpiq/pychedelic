from tempfile import NamedTemporaryFile
import unittest

import scipy.io.wavfile as sp_wavfile
import numpy

from pychedelic.utils import wav
from __init__ import STEPS_MONO_16B, STEPS_STEREO_16B


class wav_read_Test(unittest.TestCase):

    def read_all_mono_test(self):
        wfile, infos = wav.open_read_mode(STEPS_MONO_16B)
        samples = wav.read_all(wfile)
        
        self.assertEqual(samples.shape, (92610, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(STEPS_MONO_16B)
        numpy.testing.assert_array_equal(samples[:10,0].round(4), (samples_test[:10] / float(2**15)).round(4))

    def read_all_stereo_test(self):
        wfile, infos = wav.open_read_mode(STEPS_STEREO_16B)
        samples = wav.read_all(wfile)

        self.assertEqual(samples.shape, (92610, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(open(STEPS_STEREO_16B, 'r'))
        numpy.testing.assert_array_equal(samples[:10,:].round(4), (samples_test[:10,:] / float(2**15)).round(4))

    def read_block_mono_test(self):
        wfile, infos = wav.open_read_mode(STEPS_MONO_16B)
        samples = wav.read_block(wfile, 4410)

        self.assertEqual(samples.shape, (4410, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)

        expected = numpy.ones([4410, 1]) * -1
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

    def seek_and_read_mono_block_test(self):
        wfile, infos = wav.open_read_mode(STEPS_MONO_16B)
        frames = wav.seek(wfile, 0.4, 0.5)
        samples = wav.read_block(wfile, 100)

        self.assertEqual(frames, numpy.round((0.5 - 0.4) * 44100, 6))
        self.assertEqual(samples.shape, (100, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)

        expected = numpy.ones([100, 1]) * -0.6
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

    def read_block_stereo_test(self):
        wfile, infos = wav.open_read_mode(STEPS_STEREO_16B)
        samples = wav.read_block(wfile, 4410)

        self.assertEqual(samples.shape, (4410, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)

        expected = numpy.ones([4410, 2])
        expected[:, 0] *= -1
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))

    def seek_and_read_stereo_block_test(self):
        wfile, infos = wav.open_read_mode(STEPS_STEREO_16B)
        frames = wav.seek(wfile, 0.6)
        samples = wav.read_block(wfile, 100)

        self.assertEqual(frames, (infos['duration'] - 0.6) * 44100)
        self.assertEqual(samples.shape, (100, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)

        expected = numpy.ones([100, 2])
        expected[:, 0] *= -0.4
        expected[:, 1] *= 0.4
        numpy.testing.assert_array_equal(expected.round(3), samples.round(3))


class wav_write_Test(unittest.TestCase):

    def write_mono_test(self):
        samples = numpy.ones((4410, 1)) / 44100.0 * 2 * numpy.pi * 440
        dest_file = NamedTemporaryFile(delete=True)
        wfile, infos = wav.open_write_mode(dest_file.name, 44100, 1)
        wav.write_block(wfile, samples)
        wfile._file.flush() # To force the file to be written to the disk

        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(frame_rate, 44100)
        self.assertEqual(samples_written.shape, (4410,))
        numpy.testing.assert_array_equal(samples[:10,0].round(3), (samples_written[:10] / float(2**15)).round(3))
        dest_file.close()

    def write_stereo_test(self):
        samples = numpy.ones((4410, 2)) / 44100.0 * 2 * numpy.pi * 440
        dest_file = NamedTemporaryFile(delete=True)
        wfile, infos = wav.open_write_mode(dest_file.name, 44100, 2)
        wav.write_block(wfile, samples)
        wfile._file.flush() # To force the file to be written to the disk

        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(frame_rate, 44100)
        self.assertEqual(samples_written.shape, (4410, 2))
        numpy.testing.assert_array_equal(samples[:10,:].round(3), (samples_written[:10,:] / float(2**15)).round(3))
        dest_file.close()

    def test_write_edge_values(self):
        # Write edge values 1.0
        samples = numpy.ones((441, 1), dtype=numpy.float32)
        dest_file = NamedTemporaryFile(delete=True)
        wfile, infos = wav.open_write_mode(dest_file.name, 44100, 1)
        wav.write_block(wfile, samples)
        wfile._file.flush() # To force the file to be written to the disk

        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        numpy.testing.assert_array_equal(samples_written, numpy.array([2**15 - 1] * 441, dtype=numpy.int16))
        dest_file.close()

        # Write value 2.0, clipped to 1.0
        samples = numpy.ones((441, 1), dtype=numpy.float32) * 2.0
        dest_file = NamedTemporaryFile(delete=True)
        wfile, infos = wav.open_write_mode(dest_file.name, 44100, 1)
        wav.write_block(wfile, samples)
        wfile._file.flush() # To force the file to be written to the disk

        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        numpy.testing.assert_array_equal(samples_written, numpy.array([2**15 - 1] * 441, dtype=numpy.int16))
        dest_file.close()

        # Write edge values -1.0
        samples = numpy.ones((441, 1), dtype=numpy.float32) * -1
        dest_file = NamedTemporaryFile(delete=True)
        wfile, infos = wav.open_write_mode(dest_file.name, 44100, 1)
        wav.write_block(wfile, samples)
        wfile._file.flush() # To force the file to be written to the disk

        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        numpy.testing.assert_array_equal(samples_written, numpy.array([-2**15] * 441, dtype=numpy.int16))
        dest_file.close()