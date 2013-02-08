# TODO: support for 8-bit wavs ?
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO
from tempfile import NamedTemporaryFile
import os
import math
import numpy as np
import wave
import subprocess
import types


def write_wav(f, samples, frame_rate=44100):
    """
    Writes audio samples to a wav file.
    `f` can be a file object or a filename.
    `samples` can be a numpy array or a generator yielding numpy arrays.
    """
    fd = wave.open(f, mode='wb')
    fd.setsampwidth(2)
    fd.setframerate(frame_rate)
    if isinstance(samples, types.GeneratorType):
        chunk = samples.next()
        fd.setnchannels(chunk.shape[1])
        fd.writeframes(samples_to_string(chunk))
        for chunk in samples: fd.writeframes(samples_to_string(chunk))
    else:
        fd.setnchannels(samples.shape[1])
        fd.writeframes(samples_to_string(samples))


def read_wav(f, start=None, end=None, block_size=None):
    """
    Reads audio samples from a wav file and returns `samples, infos`.
    If `block_size` is defined, `samples` is a generator yielding
    blocks of at least `block_size` frames.
    `f` can be a file object or a filename.
    """
    # Opening the file and getting infos
    raw = wave.open(f, 'rb')
    channel_count = raw.getnchannels()
    sample_width = raw.getsampwidth()       # Sample width in byte
    if sample_width != 2: raise ValueError('Wave format not supported')
    frame_rate = raw.getframerate()
    infos = {'frame_rate': frame_rate, 'channel_count': channel_count}

    # Calculating start position and end position
    # for reading the samples
    if start is None: start = 0
    start_frame = start * frame_rate
    if end is None: end_frame = raw.getnframes()
    else: end_frame = end * frame_rate
    frame_count = int(end_frame - start_frame)

    # Reading samples between `start` and `end`.
    raw.setpos(int(start_frame))
    if block_size is None:
        samples = raw.readframes(frame_count)
        samples = string_to_samples(samples, channel_count)
        return samples, infos
    else:
        def gen():
            read = 0
            while read < frame_count:
                block = raw.readframes(block_size)
                yield string_to_samples(block, channel_count)
                read += block_size
        return gen(), infos


def samples_to_string(samples):
    """
    Takes a float32 numpy array, containing audio samples in the range [-1, 1],
    returns the equivalent wav byte string.
    `samples` can be stereo, mono, or a one-dimensional array (thus mono).
    """
    samples = samples.astype(np.float32) * 2**15
    samples = samples.clip(-2**15, 2**15 - 1)
    return samples.astype(np.int16).tostring()
    

def string_to_samples(string, channel_count):
    """
    Takes a byte string of raw PCM data and returns a float32 numpy array containing
    audio samples in range [-1, 1].
    """
    samples = np.fromstring(string, dtype='int16')
    samples = samples / float(2**15)
    samples.astype(np.float32)
    frame_count = samples.size / channel_count
    return samples.reshape([frame_count, channel_count])


def guess_fileformat(filename):
    """
    Guess the format of a sound file.
    """
    try:
        return filename.split('.')[-1]
    except IndexError:
        raise ValueError('unknown file format')


def convert_file(filename, to_format, to_filename=None):
    """
    Returns None if the file is already of the desired format.
    """
    fileformat = guess_fileformat(filename)
    if fileformat == to_format:
        if to_filename:
            shutil.copy(filename, to_filename)
            return to_filename
        else:
            return filename

    # Copying source file to a temporary file
    # TODO: why copying ?
    origin_file = NamedTemporaryFile(mode='wb', delete=True)
    with open(filename, 'r') as fd:
        while True:
            copy_buffer = fd.read(1024*1024)
            if copy_buffer: origin_file.write(copy_buffer)
            else: break
    origin_file.flush()

    # Converting the file to wav
    if to_filename is None:
        dest_file = NamedTemporaryFile(mode='rb', delete=False)
        to_filename = dest_file.name
    avconv_call = ['avconv', '-y',
                    '-f', fileformat,
                    '-i', origin_file.name,  # input options (filename last)
                    '-vn',  # Drop any video streams if there are any
                    '-f', to_format,  # output options (filename last)
                    to_filename
                  ]
    # TODO: improve to report error properly
    try:
        subprocess.check_call(avconv_call, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
    except subprocess.CalledProcessError:
        raise 
    origin_file.close()
    return to_filename

