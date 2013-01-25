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


def samples_to_string(data):
    data = (data.astype(np.float32).clip(-1, 1) + 1) * 2**15
    return data.astype(np.uint16).tostring()
    

def write_wav(f, data, sample_rate=44100):
    fd = wave.open(f, mode='wb')
    fd.setnchannels(data.shape[1])
    fd.setsampwidth(2)
    fd.setframerate(sample_rate)
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
    sample_count = frame_count * channels

    # Reading only the data between `start` and `end`,
    # putting this data to a numpy array 
    raw.setpos(int(start_frame))
    data = raw.readframes(int(frame_count))
    data = np.fromstring(data, dtype='int16')
    data = data / float(2**15)
    data.astype(np.float32)
    data = data.reshape([frame_count, channels])
    return data, {'sample_rate': frame_rate, 'channel_count': channels}


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
    avconv_call = ['ffmpeg', '-y',
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
