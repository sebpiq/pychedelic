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
    return numpy.concatenate(ramp_blocks)


def _iter_ramps(initial, values):
    values = list(values)
    previous_target = initial
    # This is used to fix the total duration of the ramps, and avoid accumulating
    # rounding errors.
    frames_offset = 0 

    while len(values):
        target, duration = values.pop(0)
        frame_count = duration * config.frame_rate
        frames_offset += frame_count % 1
        frame_count = int(frame_count) + int(frames_offset)
        frames_offset = frames_offset % 1

        step = (target - previous_target) / float(frame_count - 1)
        yield previous_target, step, frame_count
        previous_target = target


def fix_channel_count(block, channel_count):
    """
    Up-mix / down-mix `block` to `channel_count` channels.
    If `block` has too many channels, the extra channels are simply cropped,
    If `block` doesn't have enough channels, the extra channels are copied from
    the last channel available.
    """
    if block.shape[1] == channel_count: return block
    elif block.shape[1] > channel_count:
        return block[:,:channel_count]
    elif block.shape[1] < channel_count:
        return numpy.hstack([block, block[:,-1:]])


def fix_frame_count(block, frame_count):
    """
    Fix the number of frames, bringing it to `frame_count` by adding or removing 
    frames at the end of `block`.
    """
    sign = numpy.sign(frame_count)
    frame_count = abs(frame_count)
    if block.shape[0] == frame_count: return block
    elif block.shape[0] < frame_count:
        extra_frames = numpy.zeros((frame_count - block.shape[0], block.shape[1]))
        if (sign == 1):
            return numpy.vstack([block, extra_frames])
        else:
            return numpy.vstack([extra_frames, block])
    elif block.shape[0] > frame_count:
        if (sign == 1):
            return block[:frame_count,:]
        else:
            return block[-frame_count:,:]


def reshape(block, channel_count=None, frame_count=None):
    """
    Just combines `fix_frame_count` and `fix_channel_count` in one more handy function.
    """
    if channel_count != None:
        block = fix_channel_count(block, channel_count)
    if frame_count != None:
        block = fix_frame_count(block, frame_count)
    return block