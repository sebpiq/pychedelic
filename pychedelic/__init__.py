class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

config = Config(
  frame_rate = 44100,

  # This is a **recommended** block size for stream processing.  
  block_size = 1024
)