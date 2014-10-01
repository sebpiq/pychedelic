import unittest

import numpy

from pychedelic.utils.stream import Buffer


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
        self.assertEqual(buf.pull(2), None)
        self.assertEqual(buf.pull(2), None)

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
        blocks = [buf.fill(3) for i in range(0, 2)]
        numpy.testing.assert_array_equal(blocks, [None, None])



'''
class Stream_Test(PychedelicTestCase):
    
    def reblock_test(self):
        # Cut big blocks into smaller
        def gen():
            yield numpy.tile(numpy.arange(0, 10), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2)), [
            [[0, 0], [1, 1]], [[2, 2], [3, 3]], [[4, 4], [5, 5]],
            [[6, 6], [7, 7]], [[8, 8], [9, 9]]
        ])

        # Concatenate small blocks into bigger
        def gen():
            for i in range(10):
                yield numpy.array([[i * 11]])
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2)), [
            [[0], [11]], [[22], [33]], [[44], [55]],
            [[66], [77]], [[88], [99]]
        ])

        # Mix of the 2
        def gen():
            for i in range(5):
                yield numpy.array([[i * 1] * 3])
            yield numpy.tile(numpy.arange(5, 10), (3, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2)), [
            [[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5]], [[6, 6, 6], [7, 7, 7]],
            [[8, 8, 8], [9, 9, 9]]
        ])

    def reblock_exhausted_test(self):
        # Pad with zeros
        def gen():
            yield numpy.tile(numpy.arange(0, 11), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 3, when_exhausted='pad')), [
            [[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]], [[9, 9], [10, 10], [0, 0]]
        ])

        # drop block
        def gen():
            yield numpy.tile(numpy.arange(0, 11), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 3, when_exhausted='drop')), [
            [[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]]
        ])

    def reblock_overlap_test(self):
        # With cut blocks
        def gen():
            yield numpy.tile(numpy.arange(0, 5), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 3, overlap=2)), [
            [[0, 0], [1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]], [[3, 3], [4, 4], [0, 0]],
        ])

        # With concatenated blocks
        def gen():
            for i in range(5):
                yield numpy.array([[i * 11, i * 11]])
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2, overlap=1)), [
            [[0, 0], [11, 11]], [[11, 11], [22, 22]],
            [[22, 22], [33, 33]], [[33, 33], [44, 44]],
            [[44, 44], [0, 0]]
        ])
'''