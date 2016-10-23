import unittest

import numpy

from pychedelic.core import math


class LinearFunction_test(unittest.TestCase):

    def call_test(self):
        lin_func = math.LinearFunction(-2.2, 9)
        self.assertEqual(numpy.round(lin_func(-11), 10), -2.2 * -11 + 9)
        numpy.testing.assert_array_equal(
            lin_func(numpy.array([-1.1, 2.3, 35.99])).round(10), 
            numpy.array([-2.2 * -1.1 + 9, -2.2 * 2.3 + 9, -2.2 * 35.99 + 9]).round(10)
        )
        numpy.testing.assert_array_equal(
            lin_func(numpy.array([[-1.1, 2.3], [35.99, -124.8]])).round(10), 
            numpy.array([
                [-2.2 * -1.1 + 9, -2.2 * 2.3 + 9], 
                [-2.2 * 35.99 + 9, -2.2 * -124.8 + 9]
            ]).round(10)
        )

    def from2points_test(self):
        lin_func = math.LinearFunction.from2points([0.2, 1.08], [0.75, 1.025])
        self.assertEqual(numpy.round(lin_func(0.88), 10), 1.012)
        self.assertEqual(numpy.round(lin_func(10), 10), 0.1)
        numpy.testing.assert_array_equal(
            lin_func(numpy.array([1, 2, 3, 3.5])).round(10), 
            numpy.array([1.0, 0.9, 0.8, 0.75]).round(10)
        )


class ExpRatioFunction_test(unittest.TestCase):

    def call_test(self):
        exp_func = math.ExpRatioFunction(2.0)
        self.assertEqual(exp_func(0), 0)
        self.assertEqual(exp_func(1), 1)
        self.assertEqual(numpy.round(exp_func(0.5), 10), 0.2689414214)
        numpy.testing.assert_array_equal(
            exp_func(numpy.array([[0, 1], [0.2, 0.8]])).round(8),
            [[ 0.0,  1.0], [0.07697924, 0.61871932]]
        )