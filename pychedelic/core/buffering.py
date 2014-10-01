import numpy


class Buffer(object):

    def __init__(self, source):
        self.source = source
        self._buffer = [] # a list of blocks of samples
        self._size = 0

    def fill(self, to_size):
        while self._size < to_size:
            try:
                chunk = self.source.next()

            # Source exhausted
            except StopIteration:
                break

            else:
                self._buffer.append(chunk)
                self._size += chunk.shape[0]

        if len(self._buffer):
            block = numpy.concatenate(self._buffer, axis=0)
            return block
        else: return None

    def pull(self, block_size, overlap=0):
        if overlap and overlap >= block_size:
            raise ValueError('overlap cannot be more than block_size')

        # First, get as much chunk of data as needed to
        # make a block.
        while self._size < block_size:
            try:
                chunk = self.source.next()

            # Source exhausted
            except StopIteration:
                if len(self._buffer):
                    block = numpy.concatenate(self._buffer, axis=0)
                    self._buffer = []
                    self._size = 0
                    return block
                else: return None

            else:
                self._buffer.append(chunk)
                self._size += chunk.shape[0]
        
        # If there's too much data in the buffer, we need
        # to cut the last chunk and keep the remainder for next iteration
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