import wave

import numpy as np

from utils.files import string_to_samples


class BaseNode(object):

    def __init__(self, block_size=0):
        self.block_size = block_size
        self.buffer = []
        self.exhausted = False

    def __iter__(self):
        return self
        
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
        pass


class Node(object):

    def __init__(self, block_size=0):
        super(Node, self).__init__(block_size=block_size)
        self.input = None

    def plug_in(self, node):
        self.input = node


class SourceNode(BaseNode):
    pass


class FileReader(SourceNode):

    def __init__(self, filelike, start=None, end=None, block_size=0):
        super(FileReader, self).__init__(block_size=block_size)
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
        return super(FileReader, self).__iter__()

    def block(self):
        data = self.fd.readframes(self.block_size)
        if not data: raise StopIteration()
        return string_to_samples(data, self.channel_count)
        

if __name__ == '__main__':
    import pdb
    f_reader = FileReader('tests/sounds/A440_mono_16B.wav', block_size=10)
    blocks = list(f_reader)
    block_lengths = np.array([b.shape[0] for b in blocks])
    print np.all(block_lengths == 10)
