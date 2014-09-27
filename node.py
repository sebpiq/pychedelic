import wave

import numpy as np

from utils.files import samples_to_string, read_wav, write_wav
from utils.stream import reblock


class _Node(object):

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError()


class _HasOutput(object):

    def __gt__(self, other):
        """
        Allows to connect nodes with `>` operator.
        """
        if isinstance(other, _Node):
            if isinstance(other, _HasInput):
                other.input = self
                other.on_connection()
                return other
            else:
                raise ValueError('%s has no input' % other)
        else:
            raise ValueError('cannot connect to %s' % other)


class _HasInput(object):

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

    def on_connection(self):
        """
        Method called when the node received a connection.
        """
        pass


class PipeNode(_HasOutput, _HasInput, _Node):
    pass


class SourceNode(_HasOutput, _Node):
    pass


class SinkNode(_HasInput, _Node):
    pass


class FromFile(SourceNode):

    def __init__(self, filelike, start=None, end=None, block_size=1024):
        super(FromFile, self).__init__()
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


class ToFile(SinkNode):

    def __init__(self, filelike, sample_rate=44100, **kwargs):
        super(ToFile, self).__init__(**kwargs)
        self.filelike = filelike
        self.sample_rate = sample_rate

    def on_connection(self):
        write_wav(self.filelike, iter(self.input), self.sample_rate)