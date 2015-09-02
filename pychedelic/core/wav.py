# TODO: support for 8-bit wavs ?
import wave
import struct

from . import pcm


def open_write_mode(f, frame_rate, channel_count):
    wfile = wave.open(f, mode='wb')
    wfile.setsampwidth(2)
    wfile.setframerate(frame_rate)
    wfile.setnchannels(channel_count)
    return wfile, _get_file_infos(wfile)


def open_read_mode(f):
    try:
        wfile = wave.open(f, mode='rb')
    except wave.Error as err:
        raise FormatError(err)
    sample_width = wfile.getsampwidth()       # Sample width in byte
    if sample_width != 2:
        raise FormatError('Sample width %s not supported yet' % sample_width)
    return wfile, _get_file_infos(wfile)


def seek(wfile, position, end=None):
    """
    Seeks `position` in `wfile`, and returns the number of frames to read until `end`.
    """
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
    try:
        wfile.writeframes(pcm.samples_to_string(block))
    except struct.error:
        if wfile.getnframes() * wfile.getsampwidth() >= 2**32:
            raise WavSizeLimitError
        else: raise


def _get_file_infos(wfile):
    frame_rate = wfile.getframerate()
    return {
        'frame_rate': frame_rate,
        'channel_count': wfile.getnchannels(),
        'frame_count': wfile.getnframes(),
        'duration': wfile.getnframes() / float(frame_rate),
        'bit_depth': wfile.getsampwidth() * 8
    }


class FormatError(Exception):
    """
    Raised when attempting to read a wave file failed.
    """
    pass


class WavSizeLimitError(Exception):
    """
    Raised when the size limit for wav files has been reached.
    """
    pass
