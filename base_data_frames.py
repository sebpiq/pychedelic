import numpy as np
import pandas as pnd

import algorithms as algos


class PychedelicDataFrame(pnd.DataFrame):

    def maxima(self, i, take_edges=True):
        return algos.maxima(self[i], take_edges=take_edges)

    def minima(self, i, take_edges=True):
        return algos.minima(self[i], take_edges=take_edges)

    def smooth(self, i, window_size=11, window_func='hanning'):
        return algos.smooth(self[i], window_size=window_size, window_func=window_func)

    def convolve(self, i, array, mode='full'):
        return pnd.Series(np.convolve(self[i], array, mode=mode))

    def _constructor(self, *args, **kwargs):
        """
        This is used by `pandas` to create a new `DataFrame` when doing any operation,
        for example slicing, ...
        PB is, it is not implemented everywhere yet : https://github.com/pydata/pandas/issues/60
        """
        return self.__class__(*args, **kwargs)


class PychedelicSampledDataFrame(PychedelicDataFrame):

    def __init__(self, data, **kwargs):
        try:
            sample_rate = kwargs.pop('sample_rate')
        except KeyError:
            raise TypeError('sample_rate kwarg is required')
        if 'index' in kwargs:
            raise TypeError('index is generated automatically with sample_rate')

        super(PychedelicSampledDataFrame, self).__init__(data, **kwargs)
        self.sample_rate = sample_rate
        self.index = np.arange(0, self.shape[0]) * 1.0 / sample_rate

    @property
    def sample_count(self):
        return self.shape[0]

    def _constructor(self, *args, **kwargs):
        kwargs.setdefault('sample_rate', self.sample_rate)
        return self.__class__(*args, **kwargs)
