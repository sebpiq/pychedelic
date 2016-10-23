import numpy


class LinearFunction(object):
    """
    Takes any float or numpy array and apply the linear equation `ax + b`
    """

    @classmethod
    def from2points(cls, p1, p2):
        a = float(p2[1] - p1[1]) / (p2[0] - p1[0])
        b = float(p1[1]) - a * p1[0]
        return cls(a, b)

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, val):
        return val * self.a + self.b


class ExpRatioFunction(object):
    """
    Takes a float or numpy array `val` in [0, 1]. Maps this value to an exponential scale
    in [0, 1]. The higher `beta`, the steeper the slope. 
    This is useful for example for smooth volume fade in / fade out.
    """

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, val):
        return ((numpy.exp(val * self.beta) - numpy.exp(0)) 
            / (numpy.exp(self.beta) - numpy.exp(0)))
