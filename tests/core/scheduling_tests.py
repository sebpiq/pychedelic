import unittest

import numpy

from pychedelic.core.scheduling import Clock
from pychedelic import config


class Clock_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_scheduling_test(self):
        config.frame_rate = 44100
        clock = Clock()

        ran = []
        for i in range(0, 4):
            clock.run_after(i, lambda k: ran.append(k), args=[i])

        self.assertEqual(ran, [0])
        self.assertEqual(clock.current_time, 0 * config.frame_rate)

        self.assertEqual(clock.advance(22050), 22050)
        self.assertEqual(clock.advance(22050), 22050)
        self.assertEqual(clock.current_time, 1 * config.frame_rate)
        self.assertEqual(ran, [0])

        self.assertEqual(clock.advance(88200), 44100)
        self.assertEqual(ran, [0, 1])
        self.assertEqual(clock.current_time, 2 * config.frame_rate)

        self.assertEqual(clock.advance(44100), 44100)
        self.assertEqual(ran, [0, 1, 2])
        self.assertEqual(clock.current_time, 3 * config.frame_rate)

        self.assertEqual(clock.advance(1), 1)
        self.assertEqual(ran, [0, 1, 2, 3])
        self.assertEqual(clock.advance(100000000), 100000000)