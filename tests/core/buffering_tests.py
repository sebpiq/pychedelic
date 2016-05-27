import unittest

import numpy

from pychedelic.core.buffering import Buffer, StreamControl


class Buffer_test(unittest.TestCase):

    def read_with_offset_test(self):
        buf = Buffer()
        buf.push(numpy.array([ numpy.arange(0, 10), numpy.arange(0, 10) ]).T)
        buf.push(numpy.array([ numpy.arange(10, 20), numpy.arange(10, 20) ]).T)

        numpy.testing.assert_array_equal(
            buf.read(0, 10),
            numpy.array([ numpy.arange(0, 10), numpy.arange(0, 10) ]).T
        )
        numpy.testing.assert_array_equal(
            buf.read(5, 10),
            numpy.array([ numpy.arange(5, 15), numpy.arange(5, 15) ]).T
        )
        numpy.testing.assert_array_equal(
            buf.read(0, 20),
            numpy.array([ numpy.arange(0, 20), numpy.arange(0, 20) ]).T
        )

    def read_offset_several_blocks_test(self):
        """
        Test for a bug fix
        """
        buf = Buffer()
        buf.push(numpy.array([ numpy.arange(0, 3) ]).T)
        buf.push(numpy.array([ numpy.arange(3, 6) ]).T)
        buf.push(numpy.array([ numpy.arange(6, 9) ]).T)
        buf.push(numpy.array([ numpy.arange(9, 12) ]).T)
        buf.push(numpy.array([ numpy.arange(12, 15) ]).T)

        numpy.testing.assert_array_equal(
            buf.read(4, 1),
            numpy.array([ [4] ])
        )
        numpy.testing.assert_array_equal(
            buf.read(8, 1),
            numpy.array([ [8] ])
        )
        numpy.testing.assert_array_equal(
            buf.read(13, 1),
            numpy.array([ [13] ])
        )

    def shift_test(self):
        buf = Buffer()
        buf.push(numpy.array([ numpy.arange(0, 10), numpy.arange(0, 10) ]).T)
        buf.push(numpy.array([ numpy.arange(10, 20), numpy.arange(10, 20) ]).T)
        
        buf.shift(2)
        self.assertEqual(len(buf._blocks), 2)

        numpy.testing.assert_array_equal(
            buf.read(0, 4),
            numpy.array([ numpy.arange(2, 6), numpy.arange(2, 6) ]).T
        )

        buf.shift(5)
        self.assertEqual(len(buf._blocks), 2)

        numpy.testing.assert_array_equal(
            buf.read(1, 4),
            numpy.array([ numpy.arange(8, 12), numpy.arange(8, 12) ]).T
        )
        
        buf.shift(5)
        self.assertEqual(len(buf._blocks), 1)

        numpy.testing.assert_array_equal(
            buf.read(3, 3),
            numpy.array([ numpy.arange(15, 18), numpy.arange(15, 18) ]).T
        )


class StreamControl_test(unittest.TestCase):
    
    def cut_into_smaller_blocks_test(self):
        def gen():
            yield numpy.tile(numpy.arange(0, 10), (2, 1)).transpose()
        stream = StreamControl(gen())
        blocks = [stream.pull(2) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [1, 1]], [[2, 2], [3, 3]], [[4, 4], [5, 5]],
            [[6, 6], [7, 7]], [[8, 8], [9, 9]]
        ])

    def get_bigger_blocks_test(self):
        def gen():
            for i in range(10):
                yield numpy.array([[i * 11]])
        stream = StreamControl(gen())
        blocks = [stream.pull(2) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0], [11]], [[22], [33]], [[44], [55]],
            [[66], [77]], [[88], [99]]
        ])    

    def receive_variable_block_size_test(self):
        def gen():
            for i in range(5):
                yield numpy.array([[i * 1] * 3])
            yield numpy.tile(numpy.arange(5, 10), (3, 1)).transpose()
        stream = StreamControl(gen())
        blocks = [stream.pull(2) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5]], [[6, 6, 6], [7, 7, 7]],
            [[8, 8, 8], [9, 9, 9]]
        ])

    def pull_variable_block_size_test(self):
        def gen():
            for i in range(20):
                yield numpy.array([[i * 1] * 2])
        stream = StreamControl(gen())
        numpy.testing.assert_array_equal(
            stream.pull(3),
            [[0, 0], [1, 1], [2, 2]]
        )
        numpy.testing.assert_array_equal(
            stream.pull(3),
            [[3, 3], [4, 4], [5, 5]]
        )
        numpy.testing.assert_array_equal(
            stream.pull(6),
            [[6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]]
        )

    def exhausted_test(self):
        def gen():
            yield numpy.tile(numpy.arange(0, 11), (2, 1)).transpose()
        stream = StreamControl(gen())
        blocks = [stream.pull(3) for i in range(0, 3)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [1, 1], [2, 2]],
            [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]]
        ])

        block = stream.pull(3)
        numpy.testing.assert_array_equal(block, [[9, 9], [10, 10]])
        self.assertRaises(StopIteration, stream.pull, 2)

    def pad_test(self):
        def gen():
            yield numpy.tile(numpy.arange(0, 5), (2, 1)).transpose()
        stream = StreamControl(gen())
        block = stream.pull(8, pad=True)
        numpy.testing.assert_array_equal(block, [
            [0, 0], [1, 1], [2, 2], [3, 3],
            [4, 4], [0, 0], [0, 0], [0, 0]
        ])