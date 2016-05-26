import numpy


class Buffer(object):

    def __init__(self, source):
        self.source = source
        self._blocks = []               # a list of blocks of samples
        self._size = 0                  # number of available frames in `_blocks`
        self._read_pos = 0              # position offset in `_blocks`
        self._source_exhausted = False  # True when the source raised StopIteration

    def pull(self, block_size, pad=False):
        # First, get as much blocks of data as needed.
        if not self._source_exhausted:
            while self._size < block_size:
                try:
                    block = next(self.source)

                except StopIteration:
                    self._source_exhausted = True
                    break

                else:
                    self._blocks.append(block)
                    self._size += block.shape[0]

        if self._source_exhausted and self._size <= 0: 
            raise StopIteration

        # If there is not enough frames, but pad is True we just pad the output with zeros.
        block_out = self._make_block_out(min(block_size, self._size))
        if block_out.shape[0] < block_size and pad is True:
            block_out = numpy.concatenate([
                block_out,
                numpy.zeros((block_size - block_out.shape[0], block_out.shape[1]))
            ])

        # Update positions
        self._read_pos += block_size
        self._size -= block_size

        # Discard used blocks
        while self._blocks:
            block = self._blocks[0]
            if block.shape[0] > self._read_pos: break
            self._read_pos -= self._blocks.pop(0).shape[0]
        
        return block_out

    def _make_block_out(self, block_size):
        """
        Helper function to create the output block by consuming data from `_blocks` at `_read_pos`.
        """
        channel_count = self._blocks[0].shape[1]
        block_out = numpy.zeros((block_size, channel_count))
        read_pos = numpy.floor(self._read_pos)
        write_pos = 0
        i = 0

        while write_pos < block_size:
            block = self._blocks[i]
            to_read = min(block.shape[0] - read_pos, block_size - write_pos)
            block_out[write_pos:write_pos+to_read,:] = block[read_pos:read_pos+to_read,:]
            write_pos += to_read
            read_pos = 0
            i += 1

        return block_out