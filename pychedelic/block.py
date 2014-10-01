import numpy as np

def fix_channels(samples, channel_count):
    """
    Up-mix / down-mix `samples` to `channel_count` channels.
    If `samples` has too many channels, the extra channels are simply cropped,
    If `samples` doesn't have enough channels, the extra channels are copied from
    the last channel available.
    """
    if samples.shape[1] is channel_count: return samples
    elif samples.shape[1] > channel_count:
        return samples[:,:channel_count]
    elif samples.shape[1] < channel_count:
        return np.hstack([samples, samples[:,-1:]])
