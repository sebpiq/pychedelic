import numpy

from . import errors


def samples_to_string(samples):
    """
    Takes a float32 numpy array, containing audio samples in the range [-1, 1],
    returns the equivalent wav byte string.
    `samples` can be stereo, mono, or a one-dimensional array (thus mono).
    """
    return float_to_int(samples).tostring()
    

def string_to_samples(string, channel_count):
    """
    Takes a byte string of int16 raw PCM data and returns a float32 numpy array containing
    audio samples in range [-1, 1].
    """
    # check that we have an exact number of frames
    # % 2 because we have 2 bytes per sample
    remainder = len(string) % (2 * channel_count)
    if remainder != 0: raise errors.PcmDecodeError()
    samples = numpy.fromstring(string, dtype='int16')
    samples = int_to_float(samples)
    frame_count = samples.size / channel_count
    return samples.reshape([frame_count, channel_count])


def float_to_int(samples):
    samples = samples.astype(numpy.float32) * 2**15
    return samples.clip(-2**15, 2**15 - 1).astype(numpy.int16)


def int_to_float(samples):
    samples = samples / float(2**15)
    samples.astype(numpy.float32)
    return samples