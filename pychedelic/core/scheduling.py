from ..config import config


class Clock(object):

    def __init__(self):
        self.current_time = 0
        self._events = []

    def run_after(self, dur, func, args=None, kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        if dur <= 0:
            func(*args, **kwargs)
            return 
        dur_frames = round(dur * config.frame_rate)
        self._events.append({
            'time': int(self.current_time + dur_frames), 'func': func,
            'args': args, 'kwargs': kwargs
        })
        self._events.sort(key=lambda e: e['time'])

    def advance(self, max_frames):
        # Execute events whose time has come
        while self._events and self._events[0]['time'] == self.current_time:
            event = self._events.pop(0)
            event['func'](*event['args'], **event['kwargs'])

        # Calculate how many frames we can run before we meet the next event.
        if self._events:
            next_event = self._events[0]
            advance_frames = min(max_frames, next_event['time'] - self.current_time)
        else: advance_frames = max_frames
        self.current_time += advance_frames
        return advance_frames