import wave

import numpy as np

from utils.files import string_to_samples, samples_to_string


class BaseNode(object):

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError()


class AudioOutMixin(object):

    def __init__(self, block_size=0):
        self.block_size = block_size
        self.buffer = []
        self.exhausted = False

    def next(self):
        if self.exhausted: raise StopIteration()

        # First, get as much chunk of data as needed to
        # make a block.
        size = sum([s.shape[0] for s in self.buffer])
        while size < self.block_size:
            try:
                chunk = self.block()
            except StopIteration:
                self.exhausted = True
                chunk = np.zeros((self.block_size - size, 1))
            self.buffer.append(chunk)
            size += chunk.shape[0]
        
        # If there's too much data in the buffer, we need
        # to cut the last chunk and keep the remainder for next iteration
        extra = sum([s.shape[0] for s in self.buffer]) - self.block_size
        new_buffer = []
        if extra > 0:
            last_chunk = self.buffer.pop(-1)
            self.buffer.append(last_chunk[:extra])
            new_buffer = [last_chunk[extra:]]

        # Finally concatenate the chunks in the buffer,
        # prepare buffer or next time and return the block
        block = np.concatenate(self.buffer, axis=0)
        self.buffer = new_buffer
        return block

    def block(self):
        raise NotImplementedError()


class AudioInMixin(object):

    def __init__(self):
        self.input = None

    def plug_in(self, node):
        self.input = node


class PipeNode(AudioOutMixin, AudioInMixin, BaseNode):
    
    def __init__(self, *args, **kwargs):
        AudioOutMixin.__init__(self, *args, **kwargs)
        AudioInMixin.__init__(self, *args, **kwargs)


class SourceNode(AudioOutMixin, BaseNode):
    pass


class SinkNode(AudioInMixin, BaseNode):
    pass
    

class SoundFile(SourceNode):

    def __init__(self, filelike, start=None, end=None, block_size=0):
        super(SoundFile, self).__init__(block_size=block_size)
        # We open just to get the name of the file
        with open(filelike, 'r') as fd:
            self.filename = fd.name

        # Opening the file and getting infos
        self.fd = fd = wave.open(filelike, 'rb')
        sample_width = fd.getsampwidth()       # Sample width in byte
        if sample_width != 2: raise ValueError('Wave format not supported')
        self.channel_count = fd.getnchannels()
        self.frame_rate = fd.getframerate()

        # Calculating start position and end position
        # for reading the data
        if start is None: start = 0
        self.start_frame = start * self.frame_rate
        self.end_frame = fd.getnframes() if end is None else end * self.frame_rate
        self.frame_count = int(self.end_frame - self.start_frame)

    def __iter__(self):
        # Reading only the data between `start` and `end`,
        # putting this data to a numpy array 
        self.fd.setpos(int(self.start_frame))
        return super(SoundFile, self).__iter__()

    def block(self):
        data = self.fd.readframes(self.block_size)
        if not data: raise StopIteration()
        return string_to_samples(data, self.channel_count)


class ToRaw(SinkNode):

    def next(self):
        return samples_to_string(self.input.next())


if __name__ == '__main__':
    import pdb
    import wave 

    soundfile = SoundFile('tests/sounds/A440_mono_16B.wav', block_size=10)
    blocks = list(soundfile)
    block_lengths = np.array([b.shape[0] for b in blocks])
    print np.all(block_lengths == 10)

    soundfile = SoundFile('tests/sounds/A440_mono_16B.wav', block_size=10)
    to_raw = ToRaw()
    to_raw.plug_in(soundfile)
    raw_blocks = list(to_raw)
    raw = ''.join(raw_blocks)
    raw_test_fd = wave.open('tests/sounds/A440_mono_16B.wav', 'rb')
    raw_test = raw_test_fd.readframes(raw_test_fd.getnframes())
    print raw_test == raw, raw.startswith(raw_test)
    
