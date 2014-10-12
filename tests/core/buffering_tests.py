import unittest

import numpy

from pychedelic.core.buffering import Buffer


class Buffer_Test(unittest.TestCase):
    
    def cut_into_smaller_blocks_test(self):
        def gen():
            yield numpy.tile(numpy.arange(0, 10), (2, 1)).transpose()
        buf = Buffer(gen())
        blocks = [buf.pull(2) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [1, 1]], [[2, 2], [3, 3]], [[4, 4], [5, 5]],
            [[6, 6], [7, 7]], [[8, 8], [9, 9]]
        ])

    def get_bigger_blocks_test(self):
        def gen():
            for i in range(10):
                yield numpy.array([[i * 11]])
        buf = Buffer(gen())
        blocks = [buf.pull(2) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0], [11]], [[22], [33]], [[44], [55]],
            [[66], [77]], [[88], [99]]
        ])    

    def variable_block_size_test(self):
        def gen():
            for i in range(5):
                yield numpy.array([[i * 1] * 3])
            yield numpy.tile(numpy.arange(5, 10), (3, 1)).transpose()
        buf = Buffer(gen())
        blocks = [buf.pull(2) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5]], [[6, 6, 6], [7, 7, 7]],
            [[8, 8, 8], [9, 9, 9]]
        ])

    def exhausted_test(self):
        def gen():
            yield numpy.tile(numpy.arange(0, 11), (2, 1)).transpose()
        buf = Buffer(gen())
        blocks = [buf.pull(3) for i in range(0, 3)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [1, 1], [2, 2]],
            [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]]
        ])

        block = buf.pull(3)
        numpy.testing.assert_array_equal(block, [[9, 9], [10, 10]])
        self.assertRaises(StopIteration, buf.pull, 2)

    def pad_test(self):
        def gen():
            yield numpy.tile(numpy.arange(0, 5), (2, 1)).transpose()
        buf = Buffer(gen())
        block = buf.pull(8, pad=True)
        numpy.testing.assert_array_equal(block, [
            [0, 0], [1, 1], [2, 2], [3, 3],
            [4, 4], [0, 0], [0, 0], [0, 0]
        ])

    def overlap_cut_test(self):
        """
        Test overlap with pulled blocks smaller than source blocks.
        """
        def gen():
            yield numpy.tile(numpy.arange(0, 6), (2, 1)).transpose()
        buf = Buffer(gen())
        blocks = [buf.pull(3, overlap=2) for i in range(0, 4)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]], [[3, 3], [4, 4], [5, 5]],
        ])

    def overlap_concatenate_test(self):
        """
        Test overlap with pulled blocks bigger than source blocks.
        """
        def gen():
            for i in range(6):
                yield numpy.array([[i * 11, i * 11]])
        buf = Buffer(gen())
        blocks = [buf.pull(2, overlap=1) for i in range(0, 5)]
        numpy.testing.assert_array_equal(blocks, [
            [[0, 0], [11, 11]], [[11, 11], [22, 22]],
            [[22, 22], [33, 33]], [[33, 33], [44, 44]],
            [[44, 44], [55, 55]]
        ])

    def buffer_fill_test(self):
        def gen():
            for i in range(6):
                yield numpy.array([[i * 11]])
        buf = Buffer(gen())
        blocks = [buf.fill(2) for i in range(0, 2)]
        numpy.testing.assert_array_equal(blocks, [
            [[0], [11]], [[0], [11]]
        ])

        buf.pull(3)
        blocks = [buf.fill(3) for i in range(0, 2)]
        numpy.testing.assert_array_equal(blocks, [
            [[33], [44], [55]], [[33], [44], [55]]
        ])

        buf.pull(2)
        blocks = [buf.fill(3) for i in range(0, 2)]
        numpy.testing.assert_array_equal(blocks, [
            [[55]], [[55]]
        ])

        buf.pull(2)
        self.assertRaises(StopIteration, buf.fill, 3)