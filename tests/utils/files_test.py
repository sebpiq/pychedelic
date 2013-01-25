import scipy.io.wavfile as sp_wavfile

from pychedelic.utils.files import read_wav, write_wav

from __init__ import PychedelicTestCase, A440_MONO_16B, A440_STEREO_16B, A440_MONO_MP3



class read_write_wave_Test(PychedelicTestCase):

    def read_wav_test(self):
        samples, infos = read_wav(A440_MONO_16B)
        self.assertEqual(samples.shape, (441, 1))
        self.assertEqual(infos['sample_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)
        # Sanity check
        sample_rate, samples_test = sp_wavfile.read(A440_MONO_16B)
        self.assertEqual(samples[:10,0].round(4), (samples_test[:10] / float(2**15)).round(4))

        samples, infos = read_wav(A440_STEREO_16B)
        self.assertEqual(samples.shape, (441, 2))
        self.assertEqual(infos['sample_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)
        # Sanity check
        sample_rate, samples_test = sp_wavfile.read(A440_STEREO_16B)
        self.assertEqual(samples[:10,:].round(4), (samples_test[:10,:] / float(2**15)).round(4))

        # Read only a segment of the file
        samples, infos = read_wav(A440_MONO_16B, start=0.002, end=0.004)
        self.assertEqual(samples.shape, (88, 1))
        self.assertEqual(infos['sample_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)

        samples, infos = read_wav(A440_STEREO_16B, start=0.002, end=0.004)
        self.assertEqual(samples.shape, (88, 2))
        self.assertEqual(infos['sample_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)

        # Omitting `end`
        samples, infos = read_wav(A440_STEREO_16B, start=0.002)
        self.assertEqual(samples.shape, (352, 2))
        self.assertEqual(infos['sample_rate'], 44100)
        self.assertEqual(infos['channel_count'], 2)

        # Omitting `start`
        samples, infos = read_wav(A440_MONO_16B, end=0.006)
        self.assertEqual(samples.shape, (264, 1))
        self.assertEqual(infos['sample_rate'], 44100)
        self.assertEqual(infos['channel_count'], 1)
