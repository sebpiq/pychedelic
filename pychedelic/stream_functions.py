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
from . import block_functions
from .config import config


def ramp(initial, *values):
    """
    Returns a ramp generator.

    The following example generates a ramp that starts from 0, go to 1 in 10 seconds,
    then go back to 0 in 5 seonds: 

        >>> stream_functions.ramp(0, (1, 10), (0, 5))
    """
    for start, step, frame_count in block_functions._iter_ramps(initial, values):
        counter = 0
        acc = start - step

        while counter < frame_count:
            next_size = min(frame_count - counter, config.block_size)
            block = numpy.ones((next_size, 1)) * step
            block = acc + numpy.cumsum(block, axis=0)
            counter += next_size
            acc = block[-1,0]
            yield block
            

class resample(object):

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
resample.next = resample.__next__ # Compatibility Python 2


class mixer(object):
    """
    Mixes several streams of audio into one.
    """

    def __init__(self, channel_count, stop_when_empty=True):
        self.sources = []
        self.clock = scheduling.Clock()
        self.channel_count = channel_count
        self.stop_when_empty = stop_when_empty

    def plug(self, source):
        self.sources.append(buffering.Buffer(source))

    def unplug(self, source):
        self.sources = filter(lambda buf: not(buf.source is source), self.sources)

    def __iter__(self):
        return self

    def __next__(self):
        empty_sources = []
        next_size = self.clock.advance(config.block_size)
        block_channels = [numpy.zeros(next_size, dtype='float32') for ch in range(self.channel_count)]

        # Iterating through all the sources and do the mixing
        for buf in self.sources:
            try:
                block = buf.pull(next_size, pad=True)
            except StopIteration:
                empty_sources.append(buf)
            else:
                # !!! If the source is empty, the buffer might return a smaller block than expected 
                # Also, if not same number of channels, the block is down-mixed / up-mixed here
                for ch in range(0, min(block.shape[1], self.channel_count)):
                    block_channels[ch] = numpy.sum([block_channels[ch], block[:,ch]], axis=0)
        
        # Forget empty sources
        for buf in empty_sources:
            self.sources.remove(buf)

        # Handle case when all sources are empty
        if len(self.sources) is 0:
            if self.stop_when_empty:
                raise StopIteration
            elif len(block_channels) is 0:
                return numpy.zeros((next_size, self.channel_count), dtype='float32')

        return numpy.column_stack(block_channels)
mixer.next = mixer.__next__ # Compatibility Python 2


class iter(object):
    """
    Creates a simple generator which will iter blocks from `samples`.
    Each ouput block is guaranteed to have `config.block_size` frames, if pad is `True`.
    """

    def __init__(self, samples, pad=False, start=0, end=None):
        self.samples = samples
        self.pad = pad
        self.end = end
        self.seek(start)

    def seek(self, position):
        def _source():
            if self.end is None:
                yield self.samples[math.floor(position * config.frame_rate):,:]
            else:
                start_frame = math.floor(position * config.frame_rate)
                end_frame = math.floor(self.end * config.frame_rate)
                yield self.samples[start_frame:end_frame,:]
        self.buffer = buffering.Buffer(_source())

    def __iter__(self):
        return self

    def __next__(self):
        return self.buffer.pull(config.block_size, pad=self.pad)
iter.next = iter.__next__ # Compatibility Python 2


class read_wav(object):

    def __init__(self, filelike, start=0, end=None):
        self.wfile, self.infos = wav.open_read_mode(filelike)
        self.end = end
        self.seek(start)
        self.frames_read = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frames_read < self.frames_to_read:
            next_size = min(config.block_size, self.frames_to_read - self.frames_read)
            block = wav.read_block(self.wfile, next_size)
            self.frames_read += next_size
            return block
        else: raise StopIteration

    def seek(self, position):
        """
        Seek `position` in seconds in the wav file.
        """
        self.frames_to_read = wav.seek(self.wfile, position, self.end)
read_wav.next = read_wav.__next__ # Compatibility Python 2


class write_wav(object):

    def __init__(self, source, filelike):
        self.source = source
        self._block = next(source)
        channel_count = self._block.shape[1]
        self.wfile, self.infos = wav.open_write_mode(filelike, config.frame_rate, channel_count)
        # Pull all audio
        for i in self: pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._block.shape[1] != self.infos['channel_count']:
            raise ValueError('Received block with %s channels, while writing wav file with %s channels' 
              % (self._block.shape[1], self.infos['channel_count'])) 

        wav.write_block(self.wfile, self._block)
        try:
            self._block = next(self.source)
        except StopIteration:
            self.wfile.close() # To force writing
            raise
write_wav.next = write_wav.__next__ # Compatibility Python 2


def to_raw(source):
    with _until_StopIteration(): 
        while True:
            yield pcm.samples_to_string(next(source))


def concatenate(source):
    """
    Concatenates all the blocks generated by source until exhaustion into one single block,
    and returns it. 
    """
    blocks = []
    with _until_StopIteration(): 
        while True:
            blocks.append(next(source))
    return numpy.concatenate(blocks)


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
