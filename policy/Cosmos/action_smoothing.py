"""Optional causal smoothing of executed absolute joint targets in radians."""
import numpy as np


class ActionSmoother:
    def __init__(self, config=None):
        config = config or {}
        self.mode = config.get('mode', 'none')
        if self.mode not in ('none', 'ema'):
            raise ValueError('action_smoothing.mode must be none or ema')
        self.alpha = float(config.get('alpha', .3))
        if self.mode == 'ema' and not 0 < self.alpha <= 1:
            raise ValueError('EMA alpha must be in (0, 1]')
        self.reset()

    def reset(self):
        self.previous = None
        self.joint_names = None

    def apply(self, actions, current_state, joint_names):
        if self.mode == 'none':
            return actions
        raw = np.asarray(actions, dtype=np.float32)
        state = np.asarray(current_state, dtype=np.float32)
        names = tuple(joint_names)
        if (raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1:] != state.shape
                or state.ndim != 1 or len(names) != len(state)
                or len(set(names)) != len(names)
                or not np.isfinite(raw).all() or not np.isfinite(state).all()):
            raise ValueError('Smoothing requires finite aligned joint targets and state')
        if self.joint_names is not None and self.joint_names != names:
            raise ValueError('Joint order changed mid-episode; reset smoothing first')
        previous = state.copy() if self.previous is None else self.previous.copy()
        output = np.empty_like(raw)
        for i, target in enumerate(raw):
            previous = target.copy() if self.alpha == 1 else previous + self.alpha * (target - previous)
            output[i] = previous
        self.previous = previous.copy()
        self.joint_names = names
        return output
