import numpy as np


def reblock(block_gen, block_size, when_exhausted='pad'):
    """
    This function takes a generator which generate blocks of sound,
    and cuts/concatenates them in order to yield blocks of size `block_size`.

    `when_exhausted` controls the behaviour when `block_gen` is exhausted,
    and the last block is not enough to yield `block_size` frames.
        - `'pad'` pads the last block with zeros
        - `'drop'` drops the last block
    """
    if when_exhausted not in ['pad', 'drop']:
        raise ValueError('invalid value for when_exhausted : %s' % when_exhausted)
    exhausted = False
    buf = []
    while not exhausted:
        # First, get as much chunk of data as needed to
        # make a block.
        size = sum([s.shape[0] for s in buf])
        while size < block_size:
            try:
                chunk = block_gen.next()
            except StopIteration:
                # If `pad`, but `buf` is empty, there's nothing to pad
                if when_exhausted == 'pad' and len(buf):
                    exhausted = True
                    channel_count = buf[0].shape[1]
                    chunk = np.zeros((block_size - size, channel_count))
                else: raise
            buf.append(chunk)
            size += chunk.shape[0]
        
        # If there's too much data in the buffer, we need
        # to cut the last chunk and keep the remainder for next iteration
        extra = sum([s.shape[0] for s in buf]) - block_size
        new_buffer = []
        if extra > 0:
            last_chunk = buf.pop(-1)
            buf.append(last_chunk[:-extra])
            new_buffer = [last_chunk[-extra:]]

        # Finally concatenate the chunks in the buffer,
        # prepare buffer for next time and yield the block
        block = np.concatenate(buf, axis=0)
        buf = new_buffer
        yield block
