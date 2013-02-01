import copy
import numpy as np
from __init__ import PychedelicTestCase
from pychedelic.base_data_frames import PychedelicSampledDataFrame, PychedelicDataFrame


class PychedelicDataFrame_Test(PychedelicTestCase):

    def maxima_test(self):
        channel1 = [1, 2, 1, 2, 5, 5, 3]
        index = np.arange(0, len(channel1)) * 0.5

        df = PychedelicDataFrame({1: channel1}, index=index)
        maxima = df.maxima(1)
        self.assertEqual(maxima, [2, 5])
        self.assertEqual(maxima.index, [0.5, 2])

        channel1 = [1, 2, 1, 2, 3, 5, 3]
        channel2 = [78, 5, 34, 33, 1, 4, 5]
        index = np.arange(0, len(channel1)) * 0.5

        df = PychedelicDataFrame({1: channel1, 2: channel2}, index=index)
        maxima = df.maxima(2, take_edges=False)
        self.assertEqual(maxima, [34])
        self.assertEqual(maxima.index, [1])

    def minima_test(self):
        channel1 = [1, 1, 2, 2, 3, 5, 3]
        index = np.arange(0, len(channel1)) * 0.5

        df = PychedelicDataFrame({1: channel1}, index=index)
        minima = df.minima(1)
        self.assertEqual(minima, [1, 3])
        self.assertEqual(minima.index, [0, 3])
        minima = df.minima(1, take_edges=False)
        self.assertEqual(minima, [1])
        self.assertEqual(minima.index, [0])

    def smooth_test(self):
        channel1 = [1, 2, 1, 3, 1, 4, 1, 5, 1, 6]
        channel2 = [4, 1, 3, 1, 4, 1, 8, 1, 7, 1]
        index = np.arange(0, len(channel1)) * 0.5

        df = PychedelicDataFrame({1: channel1, 2: channel2}, index=index)
        smoothen = df.smooth(1, window_size=4)
        

class PychedelicSampledDataFrame_Test(PychedelicTestCase):

    def init_test(self):
        channel1 = np.random.random(5)
        
        df = PychedelicSampledDataFrame({1: channel1}, frame_rate=4)
        self.assertEqual(df.index, [0, 0.25, 0.5, 0.75, 1])
        self.assertEqual(df.frame_count, 5)
