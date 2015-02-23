import numpy


class Buffer(object):

    def __init__(self, source):
        self.source = source
        self._blocks = []               # a list of blocks of samples
        self._size = 0                  # number of available frames in `_blocks`
        self._read_pos = 0              # position offset in `_blocks`
        self._source_exhausted = False  # True when the source raised StopIteration

    def fill(self, to_size):
        while self._size < to_size:
            try:
                block = next(self.source)

            # Source exhausted
            except StopIteration:
                break

            else:
                self._blocks.append(block)
                self._size += block.shape[0]

        if len(self._blocks):
            return self._make_block_out(self._size)
        else: raise StopIteration

    def pull(self, block_size, overlap=0, pad=False):
        if overlap and overlap > block_size:
            raise ValueError('overlap cannot be more than block_size')

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
        
        # Not the 
        block_out_size = block_size

        # If the source is exhausted, but pad is True, and there is still some samples,
        # we just pad the output with zeros.
        if self._source_exhausted:
            if self._size > 0:
                if pad is True:
                    channel_count = self._blocks[0].shape[1]
                    self._blocks.append(numpy.zeros((block_size - self._size, channel_count)))
                else: block_out_size = numpy.ceil(self._size)
            else:
                raise StopIteration

        block_out = self._make_block_out(block_out_size)

        # Update positions
        self._read_pos += block_size - overlap
        self._size -= block_size - overlap

        # Discard used blocks
        block = self._blocks[0]
        while block.shape[0] <= self._read_pos:
            self._blocks.pop(0)
            self._read_pos -= block.shape[0]
            if not self._blocks: break
            else: block = self._blocks[0]
        return block_out

    def pull_all(self):
        blocks = []
        while True:
            try:
                blocks.append(next(self.source))
            except StopIteration:
                break
        return numpy.concatenate(blocks, axis=0)

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