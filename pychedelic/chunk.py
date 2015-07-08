import numpy
import math

from .config import config
from .core import wav


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


def resample(block, ratio):
    """
    Resamples `block`, returning a new block that has a play rate of `ratio`.
    """
    if ratio == 1: return block

    frame_count_in = block.shape[0]
    frame_count_out = math.floor((frame_count_in - 1) / ratio) + 1
    x_in = numpy.arange(0, frame_count_in)
    x_out = numpy.arange(0, frame_count_out) * ratio
    
    block_out = []
    for block_ch in block.T:
        block_out.append(numpy.interp(x_out, x_in, block_ch))
    block_out = numpy.vstack(block_out).transpose()

    return block_out


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


def read_wav(filelike, start=0, end=None):
    """
    Reads a whole wav file. Returns a tuple `(<samples>, <infos>)`.
    """
    wfile, infos = wav.open_read_mode(filelike)
    start_frame = start * infos['frame_rate']
    if start_frame > infos['frame_count']:
        return numpy.empty([0, infos['channel_count']]), infos
    frame_count = wav.seek(wfile, start, end)
    return wav.read_block(wfile, frame_count), infos
    

def write_wav(block, filelike):
    """
    Writes `block` to a wav file, replacing the whole content. 
    """
    channel_count = block.shape[1]
    wfile, infos = wav.open_write_mode(filelike, config.frame_rate, channel_count)
    wav.write_block(wfile, block)
    wfile.close() # To force writing