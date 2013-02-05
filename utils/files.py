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


def samples_to_string(samples):
    """
    Takes a float32 numpy array, containing audio data in the range [-1, 1],
    returns the equivalent wav byte string.
    `samples` can be stereo, mono, or a one-dimensional array (thus mono).
    """
    samples = samples.astype(np.float32) * 2**15
    samples = samples.clip(-2**15, 2**15 - 1)
    return samples.astype(np.int16).tostring()
    

def string_to_samples(string, channel_count):
    """
    Takes a byte string of raw PCM data and returns a numpy array containing
    audio data in range [-1, 1].
    """
    # TODO: test
    samples = np.fromstring(string, dtype='int16')
    samples = samples / float(2**15)
    samples.astype(np.float32)
    frame_count = samples.size / channel_count
    return samples.reshape([frame_count, channel_count])


def write_wav(f, data, frame_rate=44100):
    fd = wave.open(f, mode='wb')
    fd.setsampwidth(2)
    fd.setframerate(frame_rate)
    if isinstance(data, types.GeneratorType):
        chunk = data.next()
        fd.setnchannels(chunk.shape[1])
        fd.writeframes(samples_to_string(chunk))
        for chunk in data: fd.writeframes(samples_to_string(chunk))
    else:
        fd.setnchannels(data.shape[1])
        fd.writeframes(samples_to_string(data))


def read_wav(f, start=None, end=None):
    # Opening the file and getting infos
    raw = wave.open(f, 'rb')
    channels = raw.getnchannels()
    sample_width = raw.getsampwidth()       # Sample width in byte
    if sample_width != 2: raise ValueError('Wave format not supported')
    frame_rate = raw.getframerate()

    # Calculating start position and end position
    # for reading the data
    if start is None: start = 0
    start_frame = start * frame_rate
    if end is None: end_frame = raw.getnframes()
    else: end_frame = end * frame_rate
    frame_count = end_frame - start_frame

    # Reading only the data between `start` and `end`,
    # putting this data to a numpy array 
    raw.setpos(int(start_frame))
    data = raw.readframes(int(frame_count))
    data = string_to_samples(data)
    return data, {'frame_rate': frame_rate, 'channel_count': channels}


def guess_fileformat(filename):
    # Get the format of the file
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


class OggEncoder(object):
    
    def __init__(self):
        #self._in = file()
        self._out = StringIO.StringIO()
        self.oggenc = subprocess.Popen(['oggenc', '-', '-r', '-C', '1', '-B', '16', '-R', '44100'],
                        stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        
    def encode(self, data):
        import time
        data = StringIO.StringIO(samples_to_string(data))
        out = ''
        while True:
            stdin_data = data.read(64)
            if not stdin_data: break
            self.oggenc.stdin.write(input=stdin_data)
            stdout_data # TODO
            out += stdout_data
        return out


if __name__ == '__main__':
    import math
    phase = 0
    K = 2 * math.pi * 440 / 44100
    def next_frame():
        global phase, K
        while(True):
            phase += K
            yield math.cos(phase)
    data_gen = next_frame()
    data = [data_gen.next() for i in range(44100)]

    write_wav('test.wav', data)
