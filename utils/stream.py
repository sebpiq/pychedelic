import numpy


class Buffer(object):

    def __init__(self, source):
        self.source = source
        self._buffer = [] # a list of blocks of samples
        self._size = 0

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
                    return block.shape[0], block
                else: return 0, None

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

        return block.shape[0], block


def reblock(source, block_size, when_exhausted='pad', overlap=0):
    """
    This function takes a generator which generates blocks of sound,
    and cuts/concatenates them in order to yield blocks of size `block_size`.

    `when_exhausted` controls the behaviour when `source` is exhausted,
    and the last block is not enough to yield `block_size` frames.
        - `'pad'` pads the last block with zeros
        - `'drop'` drops the last block
    """
    if when_exhausted not in ['pad', 'drop']:
        raise ValueError('invalid value for when_exhausted : %s' % when_exhausted)
    if overlap and overlap >= block_size:
        raise ValueError('overlap cannot be more than block_size')
        
    exhausted = False
    buf = []
    size = 0
    while not exhausted:
        # First, get as much chunk of data as needed to
        # make a block.
        while size < block_size:
            try:
                chunk = source.next()
            except StopIteration:
                # If `pad`, but `buf` is empty, there's nothing to pad
                if when_exhausted == 'pad' and len(buf):
                    exhausted = True
                    channel_count = buf[0].shape[1]
                    chunk = numpy.zeros((block_size - size, channel_count))
                else: raise
            buf.append(chunk)
            size += chunk.shape[0]
        
        # If there's too much data in the buffer, we need
        # to cut the last chunk and keep the remainder for next iteration
        new_buffer = []
        extra = size - block_size
        if extra > 0:
            last_chunk = buf.pop(-1)
            buf.append(last_chunk[:-extra])
            new_buffer = [last_chunk[-extra:]]
        block = numpy.concatenate(buf, axis=0)
        yield block

        # Preparing next iteration
        size = extra
        if overlap:
            new_buffer.insert(0, block[-overlap:])
            size += overlap
        buf = new_buffer
