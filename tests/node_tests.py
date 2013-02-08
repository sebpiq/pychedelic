from __init__ import PychedelicTestCase, A440_MONO_16B, A440_STEREO_16B, A440_MONO_MP3, MILES_MP3

from pychedelic.node import *


class Node_Test(PychedelicTestCase):

    pass


class Node_Test(PychedelicTestCase):

    def plugin_test(self):
        # plug-in source to sink directly
        source = SourceNode()
        sink = SinkNode()
        result = source > sink
        self.assertEqual(result, sink)
        self.assertEqual(sink.input, source)

        # plug-in with pipes in between
        pipe = PipeNode()
        pipe2 = PipeNode()
        result = source > pipe > pipe2 > sink
        self.assertEqual(result, sink)
        self.assertEqual(sink.input, pipe2)
        self.assertEqual(pipe2.input, pipe)
        self.assertEqual(pipe.input, source)

        # Invalid plug-ins
        self.assertRaises(ValueError, pipe.__gt__, source)
        self.assertRaises(ValueError, source.__gt__, 1)

    def in_block_size_test(self):
        # resizing blocks is handled by `reblock` so no need
        # to test extensively
        class ControlNode(PipeNode):

            def start(self):
                self.blocks = []

            def next(self):
                block = self.input.next()
                self.blocks.append(block)
                return block

        class DummySource(SourceNode):

            def next(self):
                if hasattr(self, 'exit'): raise StopIteration()
                data = np.tile(np.arange(0, 10), (2, 1)).transpose()
                self.exit = 1
                return data

        source = DummySource()
        control1 = ControlNode()
        control2 = ControlNode(in_block_size=2)
        control3 = ControlNode(in_block_size=3)

        sink = source > control1 > control2 > control3
        blocks3 = list(sink)
        blocks2 = control2.blocks
        blocks1 = control1.blocks
        
        self.assertEqual(self.blocks_to_list(blocks1), [[
            [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
            [6, 6], [7, 7], [8, 8], [9, 9]
        ]])
        self.assertEqual(self.blocks_to_list(blocks2), [
            [[0, 0], [1, 1]], [[2, 2], [3, 3]], [[4, 4], [5, 5]],
            [[6, 6], [7, 7]], [[8, 8], [9, 9]]
        ])
        self.assertEqual(self.blocks_to_list(blocks3), [
            [[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]], [[9, 9], [0, 0], [0, 0]]
        ])
        
'''
class SoundFile_Test(PychedelicTestCase):

    soundfile = SoundFile('tests/sounds/A440_mono_16B.wav', out_block_size=10)
    blocks = list(soundfile)
    block_lengths = np.array([b.shape[0] for b in blocks])
    print np.all(block_lengths == 10)

    soundfile = SoundFile('tests/sounds/A440_mono_16B.wav', out_block_size=10)
    to_raw = ToRaw()
    to_raw.plug_in(soundfile)
    raw_blocks = list(to_raw)
    raw = ''.join(raw_blocks)
    raw_test_fd = wave.open('tests/sounds/A440_mono_16B.wav', 'rb')
    raw_test = raw_test_fd.readframes(raw_test_fd.getnframes())
    print raw_test == raw, raw.startswith(raw_test)'''
