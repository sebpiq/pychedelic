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
        self.source = source
        self.previous_frame = None
        self.set_ratio(1)

    def set_ratio(self, val):
        """
        Ratio by which the output frame rate will be changed. 
        For example if `2`, the output will have twice as much frames as the input.
        """
        self.ratio = float(val)
        self.counter_out = -1 / self.ratio
        self.counter_in = 0

    def __iter__(self):
        return self

    def __next__(self):
        block_out_size = 0

        # Get a new block, concatenate with last frame from previous block for the needs of
        # interpolation.
        block_in = next(self.source)
        if not self.previous_frame is None:
            block_in = numpy.concatenate([self.previous_frame, block_in])

        while True:
            # Size of the returned (and interpolated) block
            # (<space to place interpolated frames> - <offset of next frame out>) * <ratio>
            block_out_size = int(1 + self.ratio * (
                (block_in.shape[0] - 1) 
                - ((self.counter_out - self.counter_in) + 1 / self.ratio)
            ))
            
            # If not enough data to interpolate on at least one frame, fetch more.
            if block_out_size: break
            block_in = numpy.concatenate([ block_in, next(self.source) ])

        times_out = self.counter_out + (numpy.ones(block_out_size) / self.ratio).cumsum()
        times_in = self.counter_in + numpy.arange(0, block_in.shape[0])

        # For each column of the block received, do the interpolation and concatenate 
        # to form a new block to return.
        block_out = []
        for block_col in block_in.T:
            block_out.append(numpy.interp(times_out, times_in, block_col))
        block_out = numpy.vstack(block_out).T

        # Prepare next iteration
        self.counter_in = times_in[-1]
        self.counter_out = times_out[-1]
        self.previous_frame = block_in[-1:, :]

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
        self.sources.append(buffering.StreamControl(source))

    def unplug(self, source):
        self.sources = filter(lambda stream: not(stream.source is source), self.sources)

    def __iter__(self):
        return self

    def __next__(self):
        empty_sources = []
        next_size = self.clock.advance(config.block_size)
        block_channels = [numpy.zeros(next_size, dtype='float32') for ch in range(self.channel_count)]

        # Iterating through all the sources and do the mixing
        for stream in self.sources:
            try:
                block = stream.pull(next_size, pad=True)
            except StopIteration:
                empty_sources.append(stream)
            else:
                # !!! If the source is empty, the buffer might return a smaller block than expected 
                # Also, if not same number of channels, the block is down-mixed / up-mixed here
                for ch in range(0, min(block.shape[1], self.channel_count)):
                    block_channels[ch] = numpy.sum([block_channels[ch], block[:,ch]], axis=0)
        
        # Forget empty sources
        for stream in empty_sources:
            self.sources.remove(stream)

        # Handle case when all sources are empty
        if len(self.sources) is 0:
            if self.stop_when_empty:
                raise StopIteration
            elif len(block_channels) is 0:
                return numpy.zeros((next_size, self.channel_count), dtype='float32')

        return numpy.column_stack(block_channels)
mixer.next = mixer.__next__ # Compatibility Python 2


class window(object):
    """
    If `pad` is `False`, when the source has not enough data left for a windows, 
    `StopIteration` will be thrown.

    `hop_size` can be a decimal number.
    `archive_size` is used to keep data in memory even after it has been handled.
    The data in memory can be used in conjunction with `get_archive`. 
    """

    def __init__(self, source, window_size, hop_size, pad=True, archive_size=0):
        self.source = source
        self.hop_size = hop_size
        self.window_size = window_size
        self.archive_size = archive_size
        self.pad = pad
        
        self._buffer = buffering.Buffer()       # The underlying buffer that will hold the data
        self._offset = 0                        # The read offset in the buffer
        self._pad_count = False                 # Counts how many frame of padding have been generated

    def __iter__(self):
        return self

    def __next__(self):
        # If we're not padding, just fetch some data from the source
        # and add it to the buffer
        if self._pad_count is False:
            while self._available_frame_count < self.window_size:
                try:
                    block_in = next(self.source)

                except StopIteration:
                    if not self.pad or self._available_frame_count == 0: 
                        raise StopIteration 
                    self._pad_count = 0
                    block_in = self._fetch_pad()
                
                self._buffer.push(block_in)

        # Source is exhausted, so we need to add padding (if `pad` is `True`)
        else:
            if (self._available_frame_count - self._pad_count) <= 0:
                raise StopIteration
            self._buffer.push(self._fetch_pad())

        # Create `block_out`, move from `hop_size`
        block_out = self._buffer.read(int(self._offset), self.window_size)
        self._offset += self.hop_size

        # Discard used data
        if (self._offset > self.archive_size):
            throw_away = int(self._offset - self.archive_size)
            self._buffer.shift(throw_away)
            self._offset -= throw_away

        return block_out

    def get_archive(self, block_out_size):
        if block_out_size > self.archive_size:
            raise ValueError('cannot get more than `archive_size`')
        return self._buffer.read(int(self._offset) - block_out_size, block_out_size)

    def _fetch_pad(self):
        missing = numpy.ceil(self.window_size - (self._buffer.size - self._offset))
        channel_count = self._buffer._blocks[0].shape[1]
        pad_block = numpy.zeros(( missing, channel_count ))
        self._pad_count += missing
        return pad_block

    @property
    def _available_frame_count(self):
        return self._buffer.size - int(self._offset)
window.next = window.__next__ # Compatibility Python 2


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
        self._stream = buffering.StreamControl(_source())

    def __iter__(self):
        return self

    def __next__(self):
        return self._stream.pull(config.block_size, pad=self.pad)
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
    stream = buffering.StreamControl(source)
    channel_count = stream.fill(1).shape[1]

    def callback(in_data, current_time, time_info, status):
        block = stream.pull(current_time)
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
