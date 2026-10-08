"""Single-flight remote inference with observation-relative action alignment.

Only the worker touches the RPC client while an episode is running. Simulator
observations/geometry are captured on the simulation thread and copied before
submission. An empty queue waits without advancing simulation time.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter

import numpy as np


class AsyncActionQueue:
    def __init__(self, get_action, *, horizon=32, prefetch_step=16, clock=perf_counter):
        if not 0 < prefetch_step < horizon:
            raise ValueError('prefetch_step must be inside the prediction horizon')
        self.get_action = get_action
        self.horizon = horizon
        self.prefetch_step = prefetch_step
        self.clock = clock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-rpc')
        self.pending = None
        self.queue = deque()
        self.anchor_step = None
        self.last_step = -1
        self.events = []

    def _submit(self, step, observation_factory):
        event = dict(observation_step=step, geometry_start_sec=self.clock())
        observation = deepcopy(observation_factory())
        event['geometry_end_sec'] = self.clock()
        self.events.append(event)

        def predict():
            try:
                actions = np.asarray(self.get_action(observation), dtype=np.float32)
                if actions.ndim != 2 or len(actions) != self.horizon or not np.isfinite(actions).all():
                    raise ValueError(f'Async policy requires {self.horizon} finite actions; got {actions.shape}')
                return actions
            finally:
                event['rpc_end_sec'] = self.clock()

        self.pending = (self.pool.submit(predict), event)

    def action(self, step, observation_factory, *, episode_steps):
        if step != self.last_step + 1:
            raise ValueError('Async control steps must be contiguous')
        self.last_step = step
        # Propagate a worker failure even while old actions remain available.
        if self.pending is not None and self.pending[0].done():
            self.pending[0].result()
        if not self.queue:
            if self.pending is None:
                self._submit(step, observation_factory)
            future, event = self.pending
            wait_start = self.clock()
            actions = future.result()
            event['queue_wait_sec'] = self.clock() - wait_start
            elapsed = step - event['observation_step']
            if elapsed >= len(actions):
                raise RuntimeError('Entire async prediction expired before adoption')
            # Actions refer to the observation time, not their arrival time.
            self.queue.extend(actions[elapsed:])
            event.update(adopt_step=step, skipped_actions=elapsed,
                         retained_actions=len(actions)-elapsed)
            self.anchor_step = event['observation_step']
            self.pending = None
        # Age is relative to the observation used for this prediction. After
        # discarding an elapsed prefix, the next prefetch can be due immediately.
        if (self.pending is None and step - self.anchor_step >= self.prefetch_step
                and step + len(self.queue) < episode_steps):
            self._submit(step, observation_factory)
        return self.queue.popleft()

    def close(self):
        # Join before reset/close can send another request on the same socket.
        self.pool.shutdown(wait=True, cancel_futures=True)
