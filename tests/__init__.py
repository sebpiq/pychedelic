import sys, os
modpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(modpath)

sounddir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sounds'))
STEPS_MONO_16B = os.path.join(sounddir, 'steps-mono-16b-44khz.wav')
STEPS_STEREO_16B = os.path.join(sounddir, 'steps-stereo-16b-44khz.wav')
A440_MONO_16B = os.path.join(sounddir, 'A440-mono-16b-44khz.wav')
A440_STEREO_16B = os.path.join(sounddir, 'A440-stereo-16b-44khz.wav')