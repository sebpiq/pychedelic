from ..config import config


class Clock(object):
    """
    Simple clock for scheduling things. Example of usage :

        def gain(source):
            clock = Clock()
            buffered_source = Buffer(source)
            block_size = 1024

            context = { 'gain': 1 }
            def change_gain(val): context['gain'] = val

            # We schedule the gain to change after 1 sec, 2 secs and 3 secs.
            clock.run_after(1, change_gain, args=[2])
            clock.run_after(2, change_gain, args=[3])
            clock.run_after(3, change_gain, args=[4])

            while True:
                # Executes the events scheduled at the current frame.
                # `next_size` if the number of frames before next event.
                next_size = clock.advance(block_size)

                # Pull all frames until next event
                yield buffered_source.pull(next_size) * context['gain']
    """

    def __init__(self):
        self.current_frame = 0
        self._events = []

    def run_after(self, dur, func, args=None, kwargs=None):
        """
        Runs `func(*args, **kwargs)` after `dur` seconds.
        """
        args = args or []
        kwargs = kwargs or {}
        if dur <= 0:
            func(*args, **kwargs)
            return 
        dur_frames = round(dur * config.frame_rate)
        self._events.append({
            'time': int(self.current_frame + dur_frames), 'func': func,
            'args': args, 'kwargs': kwargs
        })
        self._events.sort(key=lambda e: e['time'])

    def advance(self, max_frames, force=False):
        """
        Executes the events that are due (scheduled either on current frame or on a previous)
        and increments current frame until next scheduled event or `max_frames`.
        If `force` is `True`, the clock will be advanced from exactly `max_frames`, even if it causes
        some events to be overdue.
        """
        # Execute events whose time has come
        while self._events and self._events[0]['time'] <= self.current_frame:
            event = self._events.pop(0)
            event['func'](*event['args'], **event['kwargs'])

        # Calculate how many frames we can run before we meet the next event.
        if self._events and force is False:
            next_event = self._events[0]
            advance_frames = min(max_frames, next_event['time'] - self.current_frame)
        else: advance_frames = max_frames
        self.current_frame += advance_frames
        return advance_frames