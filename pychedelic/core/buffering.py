import numpy


class Buffer(object):

    def __init__(self, source):
        self.source = source
        self._buffer = [] # a list of blocks of samples
        self._size = 0

    def fill(self, to_size):
        while self._size < to_size:
            try:
                block = next(self.source)

            # Source exhausted
            except StopIteration:
                break

            else:
                self._buffer.append(block)
                self._size += block.shape[0]

        if len(self._buffer):
            block = numpy.concatenate(self._buffer, axis=0)
            return block
        else: raise StopIteration

    def pull(self, block_size, overlap=0, pad=False):
        if overlap and overlap >= block_size:
            raise ValueError('overlap cannot be more than block_size')

        # First, get as much blocks of data as needed.
        while self._size < block_size:
            try:
                block = next(self.source)

            # Source exhausted
            except StopIteration:
                if len(self._buffer):
                    block = numpy.concatenate(self._buffer, axis=0)
                    self._buffer = []
                    self._size = 0
                    if pad is True:
                        zeros = numpy.zeros((block_size - block.shape[0], block.shape[1]))
                        return numpy.vstack([block, zeros])
                    else: return block
                else: raise StopIteration

            else:
                self._buffer.append(block)
                self._size += block.shape[0]
        
        # If there's too much data in the buffer, we need
        # to cut the last block and keep the remainder for next iteration
        new_buffer = []
        extra = self._size - block_size
        if extra > 0:
            last_chunk = self._buffer.pop(-1)
            self._buffer.append(last_chunk[:-extra])
            new_buffer = [last_chunk[-extra:]]
        block = numpy.concatenate(self._buffer, axis=0)

        # Preparing next iteration
        self._size = extra
        if overlap:
            new_buffer.insert(0, block[-overlap:])
            self._size += overlap
        self._buffer = new_buffer

        return block

    def pull_all(self):
        blocks = []
        while True:
            try:
                blocks.append(next(self.source))
            except StopIteration:
                break
        return numpy.concatenate(blocks, axis=0)