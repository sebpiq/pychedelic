import os
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

            def __init__(self, in_block_size=10):
                super(ControlNode, self).__init__(in_block_size=in_block_size)
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
        
    def simple_process_test(self):
        connections = []

        class CountSource(SourceNode):
            def __init__(self):
                super(CountSource, self).__init__()
                self.count = 0
            def on_connection(self):
                connections.append('source')
            def next(self):
                self.count += 1
                return np.ones([5, 1]) * self.count

        class HalfPipe(PipeNode):
            def __init__(self):
                super(HalfPipe, self).__init__(in_block_size=5)
            def on_connection(self):
                connections.append('pipe')
            def next(self):
                block = self.input.next()
                return block * 0.5

        class ConcatSink(SinkNode):
            def __init__(self):
                super(ConcatSink, self).__init__(in_block_size=5)
                self.all_blocks = []
            def on_connection(self):
                connections.append('sink')
            def next(self):
                self.all_blocks.append(self.input.next())

        count = CountSource()
        halve = HalfPipe()
        concat = ConcatSink()

        count > halve > concat
        concat.next()
        concat.next()
        concat.next()
        concat.next()

        self.assertEqual(connections, ['pipe', 'sink'])
        
        self.assertEqual(len(concat.all_blocks), 4)
        self.assertEqual(concat.all_blocks[0], np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]).transpose())
        self.assertEqual(concat.all_blocks[1], np.array([[1, 1, 1, 1, 1]]).transpose())
        self.assertEqual(concat.all_blocks[2], np.array([[1.5, 1.5, 1.5, 1.5, 1.5]]).transpose())
        self.assertEqual(concat.all_blocks[3], np.array([[2, 2, 2, 2, 2]]).transpose())


class FromFile_Test(PychedelicTestCase):

    def simple_read_test(self):
        # Read all file
        soundfile = FromFile(A440_MONO_16B, block_size=10)
        blocks = list(soundfile)
        self.assertTrue(all([b.shape[0] == 10 for b in blocks[:-1]]))
        self.assertEqual(len(blocks[-1]), 1)
        self.assertEqual(sum([b.shape[0] for b in blocks]), 441)
        test_blocks, infos = read_wav(A440_MONO_16B, block_size=10)
        self.assertEqual(self.blocks_to_list(blocks), self.blocks_to_list(test_blocks))

        # Read only a segment of the file
        soundfile = FromFile(A440_STEREO_16B, start=0.002, end=0.004)
        blocks = list(soundfile)
        samples = blocks[0]
        self.assertTrue(len(blocks), 1)
        self.assertEqual(samples.shape[0], 88)
        self.assertEqual(samples.shape[1], 2)
        test_samples, infos = read_wav(A440_STEREO_16B, start=0.002, end=0.004)
        self.assertEqual(samples, test_samples)


class ToFile_Test(PychedelicTestCase):

    def simple_write_test(self):
        temp_file = NamedTemporaryFile()

        fromfile = FromFile(A440_MONO_16B, block_size=10)
        tofile = ToFile(temp_file)

        fromfile > tofile

        temp_file.seek(0)
        actual_samples, infos = read_wav(temp_file)
        expected_samples, infos = read_wav(A440_MONO_16B)

        self.assertEqual(actual_samples, expected_samples)