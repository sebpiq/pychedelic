import subprocess
import os
from tempfile import NamedTemporaryFile


def convert_file(filename, to_format):
    """
    Returns None if the file is already of the desired format.
    """
    fileformat = guess_fileformat(filename)
    if fileformat == to_format: return

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
    dest_file = NamedTemporaryFile(mode='rb', delete=True)
    ffmpeg_call = ['ffmpeg', '-y',
                    '-f', fileformat,
                    '-i', origin_file.name,  # input options (filename last)
                    '-vn',  # Drop any video streams if there are any
                    '-f', to_format,  # output options (filename last)
                    dest_file.name
                  ]
    subprocess.check_call(ffmpeg_call, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
    origin_file.close()
    return dest_file


def guess_fileformat(filename):
    # Get the format of the file
    try:
        return filename.split('.')[-1]
    except IndexError:
        raise ValueError('unknown file format')
