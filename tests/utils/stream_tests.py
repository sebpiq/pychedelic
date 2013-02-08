from unittest import TestCase

import numpy as np

from pychedelic.utils.stream import reblock
from __init__ import PychedelicTestCase


class Stream_Test(PychedelicTestCase):
    
    def reblock_test(self):
        # Cut big blocks into smaller
        def gen():
            yield np.tile(np.arange(0, 10), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2)), [
            [[0, 0], [1, 1]], [[2, 2], [3, 3]], [[4, 4], [5, 5]],
            [[6, 6], [7, 7]], [[8, 8], [9, 9]]
        ])

        # Concatenate small blocks into bigger
        def gen():
            for i in range(10):
                yield np.array([[i * 11]])
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2)), [
            [[0], [11]], [[22], [33]], [[44], [55]],
            [[66], [77]], [[88], [99]]
        ])

        # Mix of the 2
        def gen():
            for i in range(5):
                yield np.array([[i * 1] * 3])
            yield np.tile(np.arange(5, 10), (3, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2)), [
            [[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5]], [[6, 6, 6], [7, 7, 7]],
            [[8, 8, 8], [9, 9, 9]]
        ])

    def reblock_exhausted_test(self):
        # Pad with zeros
        def gen():
            yield np.tile(np.arange(0, 11), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 3, when_exhausted='pad')), [
            [[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]], [[9, 9], [10, 10], [0, 0]]
        ])

        # drop block
        def gen():
            yield np.tile(np.arange(0, 11), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 3, when_exhausted='drop')), [
            [[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]],
            [[6, 6], [7, 7], [8, 8]]
        ])

    def reblock_overlap_test(self):
        # With cut blocks
        def gen():
            yield np.tile(np.arange(0, 5), (2, 1)).transpose()
        self.assertEqual(self.blocks_to_list(reblock(gen(), 3, overlap=2)), [
            [[0, 0], [1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]], [[3, 3], [4, 4], [0, 0]],
        ])

        # With concatenated blocks
        def gen():
            for i in range(5):
                yield np.array([[i * 11, i * 11]])
        self.assertEqual(self.blocks_to_list(reblock(gen(), 2, overlap=1)), [
            [[0, 0], [11, 11]], [[11, 11], [22, 22]],
            [[22, 22], [33, 33]], [[33, 33], [44, 44]],
            [[44, 44], [0, 0]]
        ])
