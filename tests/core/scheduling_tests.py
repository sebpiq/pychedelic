import unittest

import numpy

from pychedelic.core.scheduling import Clock
from pychedelic import config


class Clock_Test(unittest.TestCase):

    def tearDown(self):
        config.frame_rate = 44100
        config.block_size = 1024

    def simple_scheduling_test(self):
        """Test that `advance` executes events on current frame and goes to frame with next event"""
        config.frame_rate = 44100
        clock = Clock()

        ran = []
        for i in range(0, 4):
            clock.run_after(i, lambda k: ran.append(k), args=[i])
        clock.run_after(1, lambda k: ran.append(k), args=['extra'])

        self.assertEqual(ran, [0])
        self.assertEqual(clock.current_frame, 0 * config.frame_rate)

        self.assertEqual(clock.advance(22050), 22050)
        self.assertEqual(clock.advance(22050), 22050)
        self.assertEqual(clock.current_frame, 1 * config.frame_rate)
        self.assertEqual(ran, [0])

        self.assertEqual(clock.advance(88200), 44100)
        self.assertEqual(ran, [0, 1, 'extra'])
        self.assertEqual(clock.current_frame, 2 * config.frame_rate)

        self.assertEqual(clock.advance(44100), 44100)
        self.assertEqual(ran, [0, 1, 'extra', 2])
        self.assertEqual(clock.current_frame, 3 * config.frame_rate)

        self.assertEqual(clock.advance(1), 1)
        self.assertEqual(ran, [0, 1, 'extra', 2, 3])
        self.assertEqual(clock.advance(100000000), 100000000)

    def overdue_events_test(self):
        """Test that overdue events are executed on next `advance`"""
        config.frame_rate = 44100
        clock = Clock()

        ran = []
        for i in range(0, 5):
            clock.run_after(i, lambda k: ran.append(k), args=[i])

        self.assertEqual(ran, [0])
        self.assertEqual(clock.current_frame, 0 * config.frame_rate)

        self.assertEqual(clock.advance(44100 * 2.9, force=True), 44100 * 2.9)
        self.assertEqual(clock.current_frame, 2.9 * config.frame_rate)
        self.assertEqual(ran, [0])

        self.assertEqual(clock.advance(44100 * 10.1, force=True), 44100 * 10.1)
        self.assertEqual(clock.current_frame, 13 * config.frame_rate)
        self.assertEqual(ran, [0, 1, 2])

        self.assertEqual(clock.advance(44100 * 10, force=True), 44100 * 10)
        self.assertEqual(clock.current_frame, 23 * config.frame_rate)
        self.assertEqual(ran, [0, 1, 2, 3, 4])

