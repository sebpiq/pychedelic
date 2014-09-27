import numpy as np


def reblock(block_gen, block_size, when_exhausted='pad', overlap=0):
    """
    This function takes a generator which generates blocks of sound,
    and cuts/concatenates them in order to yield blocks of size `block_size`.

    `when_exhausted` controls the behaviour when `block_gen` is exhausted,
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
        new_buffer = []
        extra = size - block_size
        if extra > 0:
            last_chunk = buf.pop(-1)
            buf.append(last_chunk[:-extra])
            new_buffer = [last_chunk[-extra:]]
        try:
            block = np.concatenate(buf, axis=0)
        except Exception as err:
            import pdb; pdb.set_trace()
        yield block

        # Preparing next iteration
        size = extra
        if overlap:
            new_buffer.insert(0, block[-overlap:])
            size += overlap
        buf = new_buffer
