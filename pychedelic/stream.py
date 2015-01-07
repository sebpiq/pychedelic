import time
import math
from contextlib import contextmanager

import numpy
try:
    import pyaudio
except ImportError:
    print('Please install pyAudio if you want to play back audio')

from .core import wav
from .core import pcm
from .core import buffering
from .core import scheduling
from . import chunk
from .config import config


def ramp(initial, *values):
    """
    Returns a ramp generator.

    The following example generates a ramp that starts from 0, go to 1 in 10 seconds,
    then go back to 0 in 5 seonds: 

        >>> stream.ramp(0, (1, 10), (0, 5))
    """
    for start, step, frame_count in chunk._iter_ramps(initial, values):
        counter = 0
        acc = start - step

        while counter < frame_count:
            next_size = min(frame_count - counter, config.block_size)
            block = numpy.ones((next_size, 1)) * step
            block = acc + numpy.cumsum(block, axis=0)
            counter += next_size
            acc = block[-1,0]
            yield block
            

class Resampler(object):

    def __init__(self, source):
        self.source = buffering.Buffer(source)
        self.set_ratio(1)

    def set_ratio(self, val):
        self.ratio = val
        self.frame_in = 0
        self.frame_out = -val

    def __iter__(self):
        return self

    def __next__(self):
        if self.ratio == 1: return self.source.pull(config.block_size)
        overlap = 1 # We always keep the last `frame_in` for next iteration

        x_out = self.frame_out + (numpy.ones(config.block_size) * self.ratio).cumsum()
        x_in = numpy.arange(self.frame_in, math.ceil(x_out[-1]) + 1)

        self.frame_out = x_out[-1]
        next_size = math.ceil(len(x_in))
        # If next `frame_out` is in interval [x_in[-2], x_in[-1]),
        # it means we'll need x_in[-2] for next iteration
        overlap += (self.frame_out + self.ratio) < x_in[-1]
        self.frame_in = x_in[-1] + 1 - overlap

        block_in = self.source.pull(next_size, overlap=overlap, pad=True)
        block_out = []
        for block_ch in block_in.T:
            block_out.append(numpy.interp(x_out, x_in, block_ch))
        block_out = numpy.vstack(block_out).transpose()

        return block_out
Resampler.next = Resampler.__next__ # Compatibility Python 2


class Mixer(object):

    def __init__(self):
        self.sources = []
        self.clock = scheduling.Clock()

    def plug(self, source):
        self.sources.append(buffering.Buffer(source))

    def unplug(self, source):
        self.sources = filter(lambda buf: not(buf.source is source), self.sources)

    def __iter__(self):
        return self

    def __next__(self):
        block_channels = []
        empty_sources = []
        next_size = self.clock.advance(config.block_size)

        # Iterating through all the sources and do the mixing
        for buf in self.sources:
            try:
                block = buf.pull(next_size, pad=True)
            except StopIteration:
                empty_sources.append(buf)
            else:
                # If the source is empty, the buffer might return a smaller block than expected 
                for ch in range(0, block.shape[1]):
                    if len(block_channels) < (ch + 1):
                        block_channels.append(numpy.zeros(next_size, dtype='float32'))
                    block_channels[ch] = numpy.sum([block_channels[ch], block[:,ch]], axis=0)
                
        # Forget empty sources
        for buf in empty_sources:
            self.sources.remove(buf)
        if len(self.sources) is 0: raise StopIteration

        return numpy.column_stack(block_channels)
Mixer.next = Mixer.__next__ # Compatibility Python 2


def read_wav(f, start=0, end=None):
    wfile, infos = wav.open_read_mode(f)
    current_time = wav.seek(wfile, start, end)

    read = 0
    while read < current_time:
        next_size = min(config.block_size, current_time - read)
        block = wav.read_block(wfile, next_size)
        read += next_size
        yield block


def to_wav_file(source, f):
    with _until_StopIteration():
        block = next(source)
        channel_count = block.shape[1]
        wfile, infos = wav.open_write_mode(f, config.frame_rate, channel_count)

        while True:
            wav.write_block(wfile, block)
            block = next(source)


def to_raw(source):
    with _until_StopIteration(): 
        while True:
            yield pcm.samples_to_string(next(source))


def playback(source):
    buf = buffering.Buffer(source)
    channel_count = buf.fill(1).shape[1]

    def callback(in_data, current_time, time_info, status):
        block = buf.pull(current_time)
        block_size = block.shape[0]
        if block_size == current_time:
            return (pcm.float_to_int(block), pyaudio.paContinue)
        elif block_size > 0:
            return (pcm.float_to_int(block), pyaudio.paComplete)
        else:
            return (None, pyaudio.paComplete)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(2), # Only format supported right now 16bits
        channels=channel_count, 
        rate=config.frame_rate,
        output=True,
        stream_callback=callback
    )

    stream.start_stream()
    while stream.is_active():
        time.sleep(0.05)

    stream.stop_stream()
    stream.close()
    p.terminate()


@contextmanager
def _until_StopIteration():
    try:
        yield
    except StopIteration:
        pass
