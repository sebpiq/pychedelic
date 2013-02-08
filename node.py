import wave

import numpy as np

from utils.files import string_to_samples, samples_to_string
from utils.stream import reblock


class Node(object):

    def __iter__(self):
        self.start()
        return self

    def start(self):
        """
        Method called when the processing starts.
        """
        pass

    def next(self):
        raise NotImplementedError()


class HasOutput(object):

    def __gt__(self, other):
        """
        Allows to connect nodes with `>` operator.
        """
        if isinstance(other, Node):
            if isinstance(other, HasInput):
                other.input = self
                return other
            else:
                raise ValueError('%s has no input' % other)
        else:
            raise ValueError('cannot connect to %s' % other)


class HasInput(object):

    def __init__(self, in_block_size=None):
        self.in_block_size = in_block_size
        self._input = None
    
    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, gen):
        gen = iter(gen)
        if self.in_block_size is None: self._input = gen
        else: self._input = reblock(gen, self.in_block_size)


class PipeNode(HasOutput, HasInput, Node):
    
    pass


class SourceNode(HasOutput, Node):
    pass


class SinkNode(HasInput, Node):
    pass


class SoundFile(SourceNode):

    def __init__(self, filelike, start=None, end=None, out_block_size=1024):
        super(SoundFile, self).__init__()
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

    def start(self):
        # Reading only the data between `start` and `end`,
        # putting this data to a numpy array 
        self.fd.setpos(int(self.start_frame))

    def block(self):
        data = self.fd.readframes(self.out_block_size)
        if not data: raise StopIteration()
        return string_to_samples(data, self.channel_count)


class ToRaw(SinkNode):

    def next(self):
        return samples_to_string(self.input.next())    
