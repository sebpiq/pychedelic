import wave
from contextlib import contextmanager

import numpy
try:
    import pyaudio
except ImportError:
    print 'Please install pyAudio if you want to play back audio'

from utils import wav, pcm


def read_wav(f, start=0, end=None, block_size=1024):
    wfile, infos = wav.open_read_mode(f)
    frame_count = wav.seek(wfile, start, end)

    read = 0
    while read < frame_count:
        next_size = min(block_size, frame_count - read)
        block = wav.read_block(wfile, next_size)
        read += next_size
        yield block


def to_wav_file(source, f, frame_rate):
    with _until_StopIteration():
        block = source.next()
        channel_count = block.shape[1]
        wfile, infos = wav.open_write_mode(f, frame_rate, channel_count)

        while True:
            wav.write_block(wfile, block)
            block = source.next()


def to_raw(source):
    with _until_StopIteration(): 
        while True:
            yield pcm.samples_to_string(source.next())


#TODO callback instead of blocking call
def playback(source, frame_rate):
    with _until_StopIteration():
        block = source.next()
        channel_count = block.shape[1]

        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(2), # Only format supported right now 16bits
            channels=channel_count, 
            rate=frame_rate,
            output=True
        )

        while True:
            stream.write(pcm.float_to_int(block))
            block = source.next()


@contextmanager
def _until_StopIteration():
    try:
        yield
    except StopIteration:
        pass


#stream.stop_stream()
#stream.close()

#p.terminate()
