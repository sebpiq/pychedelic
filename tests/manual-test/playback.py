import numpy
import scipy.io.wavfile as sp_wavfile

from __init__ import A440_MONO_16B, A440_STEREO_16B
from pychedelic import generators
from pychedelic.utils import pcm

if __name__ == '__main__':
    frame_rate = 44100


    def fm_sine(freq):
        """
        generates blocks of 1 second of frequency modulated sine wave
        """
        count = 0
        last = 0

        while count < 5:
            if type(freq) is int: f0 = freq
            else: f0 = freq.next()

            time = (numpy.arange(0, frame_rate) + count * frame_rate) / float(frame_rate)
            time = numpy.array([time]).transpose()
            yield numpy.sin(2 * numpy.pi * time * f0)
            count += 1
    

    def mult(source, val):
        while True:
            yield source.next() * val


    def add(source, val):
        while True:
            yield source.next() + val


    generators.playback(fm_sine(
        add(
            mult(
                fm_sine(5)
            , 10)
        , 300)
    ), frame_rate, 1)