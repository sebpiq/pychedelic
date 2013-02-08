import types
from tempfile import NamedTemporaryFile

import scipy.io.wavfile as sp_wavfile
import numpy as np

from pychedelic.utils.files import read_wav, write_wav
from __init__ import PychedelicTestCase, A440_MONO_16B, A440_STEREO_16B, A440_MONO_MP3


class read_write_wave_Test(PychedelicTestCase):

    def read_wav_test(self):
        samples, infos = read_wav(A440_MONO_16B)
        self.assertEqual(samples.shape, (441, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(A440_MONO_16B)
        self.assertEqual(samples[:10,0].round(4), (samples_test[:10] / float(2**15)).round(4))

        samples, infos = read_wav(A440_STEREO_16B)
        self.assertEqual(samples.shape, (441, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(open(A440_STEREO_16B, 'r'))
        self.assertEqual(samples[:10,:].round(4), (samples_test[:10,:] / float(2**15)).round(4))

        # Read only a segment of the file
        samples, infos = read_wav(A440_MONO_16B, start=0.002, end=0.004)
        self.assertEqual(samples.shape, (88, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)

        samples, infos = read_wav(A440_STEREO_16B, start=0.002, end=0.004)
        self.assertEqual(samples.shape, (88, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)

        # Omitting `end`
        samples, infos = read_wav(A440_STEREO_16B, start=0.002)
        self.assertEqual(samples.shape, (352, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)

        # Omitting `start`
        samples, infos = read_wav(A440_MONO_16B, end=0.006)
        self.assertEqual(samples.shape, (264, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)

        # Omitting `start`
        samples, infos = read_wav(A440_MONO_16B, end=0.006)
        self.assertEqual(samples.shape, (264, 1))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)

        # Returning a generator
        blocks, infos = read_wav(A440_STEREO_16B, block_size=50)
        self.assertEqual(type(blocks), types.GeneratorType)
        blocks = list(blocks)
        self.assertEqual([len(b) for b in blocks], [50, 50, 50, 50, 50, 50, 50, 50, 41])
        self.assertEqual(blocks[0].shape, (50, 2))
        self.assertEqual(infos['frame_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)
        # Sanity check
        frame_rate, samples_test = sp_wavfile.read(A440_STEREO_16B)
        self.assertEqual(blocks[0][:10,:].round(4), (samples_test[:10,:] / float(2**15)).round(4))

    def write_wav_test(self):
        # Write mono
        samples = np.sin(np.linspace(0, 0.1, 4410) * 2 * np.pi * 440)
        samples = samples.reshape(4410, 1)
        dest_file = NamedTemporaryFile(delete=True)
        write_wav(dest_file, samples, frame_rate=44100)
        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(frame_rate, 44100)
        self.assertEqual(samples_written.shape, (4410,))
        self.assertEqual(samples[:10,0].round(3), (samples_written[:10] / float(2**15)).round(3))
        dest_file.close()

        # Write stereo
        samples = np.sin(np.linspace(0, 0.1, 4410) * 2 * np.pi * 440)
        samples = samples.reshape(4410, 1)
        samples = np.hstack((samples, samples))
        dest_file = NamedTemporaryFile(delete=True)
        write_wav(dest_file, samples, frame_rate=44100)
        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(frame_rate, 44100)
        self.assertEqual(samples_written.shape, (4410, 2))
        self.assertEqual(samples[:10,:].round(3), (samples_written[:10,:] / float(2**15)).round(3))
        dest_file.close()

        # Write edge values 1.0
        samples = np.array([1.0] * 441, dtype=np.float32)
        samples = samples.reshape(441, 1)
        dest_file = NamedTemporaryFile(delete=True)
        write_wav(dest_file, samples, frame_rate=44100)
        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(samples_written, np.array([2**15 - 1] * 441, dtype=np.int16))
        dest_file.close()

        # Write value 2.0, clipped to 1.0
        samples = np.array([2.0] * 441, dtype=np.float32)
        samples = samples.reshape(441, 1)
        dest_file = NamedTemporaryFile(delete=True)
        write_wav(dest_file, samples, frame_rate=44100)
        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(samples_written, np.array([2**15 - 1] * 441, dtype=np.int16))
        dest_file.close()

        # Write edge values -1.0
        samples = np.array([-1.0] * 441, dtype=np.float32)
        samples = samples.reshape(441, 1)
        dest_file = NamedTemporaryFile(delete=True)
        write_wav(dest_file, samples, frame_rate=44100)
        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(samples_written, np.array([-2**15] * 441, dtype=np.int16))
        dest_file.close()

        # Write with generator as input
        samples = np.sin(np.linspace(0, 0.1, 4410) * 2 * np.pi * 440)
        samples = samples.reshape(4410, 1)
        samples = np.hstack((samples, samples))
        dest_file = NamedTemporaryFile(delete=True)
        gen = (samples[i*100:(i+1)*100,:] for i in range(45))
        write_wav(dest_file, gen, frame_rate=44100)
        frame_rate, samples_written = sp_wavfile.read(dest_file.name)
        self.assertEqual(frame_rate, 44100)
        self.assertEqual(samples_written.shape, (4410, 2))
        self.assertEqual(samples[:10,:].round(3), (samples_written[:10,:] / float(2**15)).round(3))
        dest_file.close()