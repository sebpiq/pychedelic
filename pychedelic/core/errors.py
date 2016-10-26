class PcmDecodeError(Exception):
    """
    Raised when attempt to decode pcm string has failed
    """
    pass


class WavFormatError(Exception):
    """
    Raised when attempting to read a wave file failed.
    """
    pass


class WavSizeLimitError(Exception):
    """
    Raised when the size limit for wav files has been reached.
    """
    pass
