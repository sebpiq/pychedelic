import time
from contextlib import contextmanager

import numpy
try:
    import pyaudio
except ImportError:
    print('Please install pyAudio if you want to play back audio')

from .core import wav
from .core import pcm
from .core import buffering
from . import chunk
from pychedelic import config


def ramp(initial, *values):
    values = list(values)
    previous_target = initial

    while len(values):
        target, time = values.pop(0)
        frame_count = round(time * config.frame_rate)
        step = (target - previous_target) / float(frame_count)

        counter = 0
        acc = previous_target - step
        while counter < frame_count:
            next_size = min(frame_count - counter, config.block_size)
            block = numpy.ones((next_size, 1)) * step
            block = acc + numpy.cumsum(block, axis=0)
            counter += next_size
            acc = block[-1,0]
            yield block
        previous_target = target
    yield numpy.array([[target]], dtype='float32')


class Mixer(object):

    def __init__(self):
        self.sources = []

    def plug(self, gen):
        self.sources.append(buffering.Buffer(gen))

    def __iter__(self):
        return self

    def __next__(self):
        block_channels = []
        empty_sources = []

        # Iterating through all the sources and do the mixing
        for buf in self.sources:
            try:
                block = buf.pull(config.block_size)
            except StopIteration:
                empty_sources.append(buf)
            else:
                # If the source is empty, the buffer might return a smaller block than expected 
                block = chunk.fix_frame_count(block, config.block_size, 0) 
                for ch in range(0, block.shape[1]):
                    if len(block_channels) < (ch + 1):
                        block_channels.append(numpy.zeros(config.block_size, dtype='float32'))
                    block_channels[ch] = numpy.sum([block_channels[ch], block[:,ch]], axis=0)
                
        # Forget empty sources
        for buf in empty_sources:
            self.sources.remove(buf)
        if len(self.sources) is 0: raise StopIteration

        return numpy.column_stack(block_channels)
Mixer.next = Mixer.__next__ # Compatibility Python 2


def read_wav(f, start=0, end=None):
    wfile, infos = wav.open_read_mode(f)
    frame_count = wav.seek(wfile, start, end)

    read = 0
    while read < frame_count:
        next_size = min(config.block_size, frame_count - read)
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

    def callback(in_data, frame_count, time_info, status):
        block = buf.pull(frame_count)
        block_size = block.shape[0]
        if block_size == frame_count:
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
