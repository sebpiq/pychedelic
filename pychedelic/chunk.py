import numpy

from pychedelic import config


def ramp(initial, *values):
    """
    Generates a ramp array.

    The following example generates a ramp that starts from 0, go to 1 in 10 seconds,
    then go back to 0 in 5 seonds: 

        >>> chunk.ramp(0, (1, 10), (0, 5))
    """
    ramp_blocks = []

    for start, step, frame_count in _iter_ramps(initial, values):
        block = numpy.ones((frame_count, 1)) * step
        block = start - step + numpy.cumsum(block, axis=0)
        ramp_blocks.append(block)

    ramp_blocks.append(numpy.array([[values[-1][0]]], dtype='float32'))
    return numpy.concatenate(ramp_blocks)


def _iter_ramps(initial, values):
    values = list(values)
    previous_target = initial

    while len(values):
        target, duration = values.pop(0)
        frame_count = round(duration * config.frame_rate)
        step = (target - previous_target) / float(frame_count)
        yield previous_target, step, frame_count
        previous_target = target


def fix_channel_count(samples, channel_count):
    """
    Up-mix / down-mix `samples` to `channel_count` channels.
    If `samples` has too many channels, the extra channels are simply cropped,
    If `samples` doesn't have enough channels, the extra channels are copied from
    the last channel available.
    """
    if samples.shape[1] == channel_count: return samples
    elif samples.shape[1] > channel_count:
        return samples[:,:channel_count]
    elif samples.shape[1] < channel_count:
        return numpy.hstack([samples, samples[:,-1:]])


def fix_frame_count(samples, frame_count, default_val):
    """
    Fix the number of frames in a block of samples, bringing it to `frame_count`.
    If there is too many frames, simply crop the block, if there is not enough, it is padded
    with `default_val`
    """
    if samples.shape[0] == frame_count: return samples
    elif samples.shape[0] < frame_count:
        extra_samples = numpy.ones((frame_count - samples.shape[0], samples.shape[1])) * default_val
        return numpy.vstack([samples, extra_samples])
    elif samples.shape[0] > frame_count:
        return samples[:frame_count,:]