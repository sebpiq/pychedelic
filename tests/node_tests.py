from tempfile import NamedTemporaryFile

from __init__ import PychedelicTestCase, A440_MONO_16B, A440_STEREO_16B, A440_MONO_MP3, MILES_MP3
from pychedelic.node import *
from pychedelic.utils.files import write_wav, read_wav


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
        

class SoundFile_Test(PychedelicTestCase):

    def simple_read_test(self):
        # Read all file
        soundfile = SoundFile(A440_MONO_16B, block_size=10)
        blocks = list(soundfile)
        self.assertTrue(all([b.shape[0] == 10 for b in blocks[:-1]]))
        self.assertEqual(len(blocks[-1]), 1)
        self.assertEqual(sum([b.shape[0] for b in blocks]), 441)
        test_blocks, infos = read_wav(A440_MONO_16B, block_size=10)
        self.assertEqual(self.blocks_to_list(blocks), self.blocks_to_list(test_blocks))

        # Read only a segment of the file
        soundfile = SoundFile(A440_STEREO_16B, start=0.002, end=0.004)
        blocks = list(soundfile)
        samples = blocks[0]
        self.assertTrue(len(blocks), 1)
        self.assertEqual(samples.shape[0], 88)
        self.assertEqual(samples.shape[1], 2)
        test_samples, infos = read_wav(A440_STEREO_16B, start=0.002, end=0.004)
        self.assertEqual(samples, test_samples)
