import numpy


class Buffer(object):
    """
    Helper to lump together several blocks in one single block, 
    without having to reallocate new arrays.
    """

    def __init__(self):
        self.size = 0
        self._blocks = []       # a list of blocks of samples
        self._read_pos = 0      # position offset in `_blocks`

    def push(self, block):
        self._blocks.append(block)
        self.size += block.shape[0]

    def shift(self, offset):
        self._read_pos += offset
        self.size -= offset
        while self._blocks:
            block = self._blocks[0]
            if block.shape[0] > self._read_pos: break
            self._read_pos -= self._blocks.pop(0).shape[0]

    def read(self, offset, block_size):
        channel_count = self._blocks[0].shape[1]
        block_out = numpy.zeros((block_size, channel_count))
        read_start_pos = self._read_pos + offset

        # First, find the first block that we can use to read data from
        i = 0
        block_start_pos = 0
        while True:
            block = self._blocks[i]
            if read_start_pos < (block_start_pos + block.shape[0]): break
            i += 1
            block_start_pos += block.shape[0]
        
        # Then iterate blocks and write data `block_out`
        write_pos = 0
        while write_pos < block_size:
            block = self._blocks[i]

            to_read_offset = max(read_start_pos - block_start_pos, 0)
            to_read = min(block.shape[0] - to_read_offset, block_size - write_pos)

            block_out[write_pos:write_pos+to_read,:] = block[to_read_offset:to_read_offset+to_read,:]
            write_pos += to_read
            i += 1
            block_start_pos += block.shape[0]

        return block_out


class StreamControl(object):
    """
    Object to control a stream of blocks and being able to pull a give amount of frames.
    For example to get a block `1024` frames from a stream whose blocks might be of
    different size :

       >>> stream = StreamControl(block_generator)
       >>> stream.pull(1024)
    """

    def __init__(self, source):
        self.source = source
        self._buffer = Buffer()
        self._source_exhausted = False  # True when the source raised StopIteration

    def pull(self, block_size, pad=False):
        # First, get as much blocks of data as needed.
        if not self._source_exhausted:
            while self._buffer.size < block_size:
                try:
                    block = next(self.source)

                except StopIteration:
                    self._source_exhausted = True
                    break

                else:
                    self._buffer.push(block)

        if self._source_exhausted and self._buffer.size <= 0: 
            raise StopIteration

        # If there is not enough frames, but pad is True we just pad the output with zeros.
        block_out = self._buffer.read(0, min(block_size, self._buffer.size))
        if block_out.shape[0] < block_size and pad is True:
            block_out = numpy.concatenate([
                block_out,
                numpy.zeros((block_size - block_out.shape[0], block_out.shape[1]))
            ])

        # Discard used blocks
        self._buffer.shift(block_size)
        
        return block_out