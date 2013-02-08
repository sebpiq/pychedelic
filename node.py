import wave

import numpy as np

from utils.files import samples_to_string, read_wav
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

    def __init__(self, filelike, start=None, end=None, block_size=1024):
        super(SoundFile, self).__init__()
        # We open just to get the name of the file
        with open(filelike, 'r') as fd:
            self.filename = fd.name
        gen, infos = read_wav(filelike, start=start, end=end, block_size=block_size)
        self._gen = gen

    def next(self):
        return self._gen.next()


class ToRaw(SinkNode):

    def next(self):
        return samples_to_string(self.input.next())
