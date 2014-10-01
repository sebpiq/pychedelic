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
        acc = 0

        while count < 5:
            if type(freq) is int:
                f0 = numpy.ones((frame_rate, 1)) * freq
            else:
                f0 = freq.next()

            phases = 2 * numpy.pi * f0 / float(frame_rate)
            phases = acc + numpy.cumsum(phases, axis=0)
            acc = phases[-1][0]
            yield numpy.sin(phases)
            count += 1
    

    def mult(source, val):
        while True:
            yield source.next() * val


    def add(source, val):
        while True:
            yield source.next() + val

    def ramp(start, end, duration):
        samp_count = duration * frame_rate
        samples = numpy.linspace(start, end, samp_count).reshape((samp_count, 1))
        count = 0
        while True:
            yield samples[count*44100:(count+1)*44100,0:1]
            count += 1

    generators.playback(fm_sine(
        add(
            mult(
                fm_sine(
                    ramp(1, 100, 5)
                )
            , 70)
        , 300)
    ), frame_rate)