from tempfile import NamedTemporaryFile
import os
import math
import numpy as np
import subprocess
import types


def guess_fileformat(filename):
    """
    Guess the format of a sound file.
    """
    try:
        return filename.split('.')[-1]
    except IndexError:
        raise ValueError('unknown file format')


def convert_file(filename, to_format, to_filename=None):
    """
    Returns None if the file is already of the desired format.
    """
    fileformat = guess_fileformat(filename)
    if fileformat == to_format:
        if to_filename:
            shutil.copy(filename, to_filename)
            return to_filename
        else:
            return filename

    # Copying source file to a temporary file
    # TODO: why copying ?
    origin_file = NamedTemporaryFile(mode='wb', delete=True)
    with open(filename, 'r') as fd:
        while True:
            copy_buffer = fd.read(1024*1024)
            if copy_buffer: origin_file.write(copy_buffer)
            else: break
    origin_file.flush()

    # Converting the file to wav
    if to_filename is None:
        dest_file = NamedTemporaryFile(mode='rb', delete=False)
        to_filename = dest_file.name
    avconv_call = ['avconv', '-y',
                    '-f', fileformat,
                    '-i', origin_file.name,  # input options (filename last)
                    '-vn',  # Drop any video streams if there are any
                    '-f', to_format,  # output options (filename last)
                    to_filename
                  ]
    # TODO: improve to report error properly
    try:
        subprocess.check_call(avconv_call, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
    except subprocess.CalledProcessError:
        raise 
    origin_file.close()
    return to_filename

