import numpy


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
    if samples.shape[0] == frame_count: return samples
    elif samples.shape[0] < frame_count:
        extra_samples = numpy.ones((frame_count - samples.shape[0], samples.shape[1])) * default_val
        return numpy.vstack([samples, extra_samples])
    elif samples.shape[0] > frame_count:
        raise ValueError('too many samples')