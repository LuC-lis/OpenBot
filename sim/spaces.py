"""极简的 Gym 风格"空间"定义。

用途：把观测和动作的**合法范围**写成代码，让"接口契约"可被检查。
如果装了 gymnasium / gym 就直接用它们的实现，否则用这个轻量替代。
"""

import numpy as np

try:
    from gymnasium import spaces as _gym_spaces  # type: ignore

    Box = _gym_spaces.Box
    Dict = _gym_spaces.Dict
    GYMNASIUM_AVAILABLE = True
except Exception:
    GYMNASIUM_AVAILABLE = False

    class Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            return (
                self.low + (self.high - self.low) * np.random.rand(*self.shape)
            ).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return bool(
                x.shape == self.shape
                and np.all(x >= self.low)
                and np.all(x <= self.high)
            )

        def __repr__(self):
            return f"Box(shape={self.shape}, low={self.low.min()}, high={self.high.max()})"

    class Dict:
        def __init__(self, spaces):
            self.spaces = dict(spaces)

        def sample(self):
            return {k: v.sample() for k, v in self.spaces.items()}

        def contains(self, x):
            return all(k in x and self.spaces[k].contains(x[k]) for k in self.spaces)

        def __getitem__(self, key):
            return self.spaces[key]

        def __repr__(self):
            return f"Dict({self.spaces})"
