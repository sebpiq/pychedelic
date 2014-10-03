# TODO: support for 8-bit wavs ?
import wave

from . import pcm


def open_write_mode(f, frame_rate, channel_count):
    wfile = wave.open(f, mode='wb')
    wfile.setsampwidth(2)
    wfile.setframerate(frame_rate)
    wfile.setnchannels(channel_count)
    return wfile, _get_file_infos(wfile)


def open_read_mode(f):
    wfile = wave.open(f, 'rb')
    sample_width = wfile.getsampwidth()       # Sample width in byte
    if sample_width != 2: raise ValueError('Wave format not supported')
    return wfile, _get_file_infos(wfile)


def seek(wfile, position, end=None):
    end_frame = wfile.getnframes()
    if end != None: end_frame = min(end * wfile.getframerate(), end_frame)
    position_frame = position * wfile.getframerate()
    wfile.setpos(int(round(position_frame)))
    return int(round(end_frame - position_frame))


def read_all(wfile):
    start_frame = wfile.tell()
    end_frame = wfile.getnframes()
    frame_count = end_frame - start_frame
    return pcm.string_to_samples(wfile.readframes(frame_count), wfile.getnchannels())


def read_block(wfile, block_size):
    start_frame = wfile.tell()
    end_frame = min(start_frame + block_size, wfile.getnframes())
    frame_count = end_frame - start_frame
    return pcm.string_to_samples(wfile.readframes(frame_count), wfile.getnchannels())


def write_block(wfile, block):
    wfile.writeframes(pcm.samples_to_string(block))


def _get_file_infos(wfile):
    frame_rate = wfile.getframerate()
    return {
        'frame_rate': frame_rate,
        'channel_count': wfile.getnchannels(),
        'frame_count': wfile.getnframes(),
        'duration': wfile.getnframes() / float(frame_rate)
    }