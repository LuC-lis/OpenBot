"""模块 6：观测组装（Observation）

把"相机图像"和"指令"打包成 OpenBot 契约要求的观测格式。

契约（见 policy/openbot/models.py 与 Autopilot.java）：
    {
      "image": float32 (H, W, 3)，取值 [0, 1]     # 相机图像
      "cmd":   float32 (1,) 或标量               # 高层指令 {-1, 0, +1}
    }

这个模块故意做得非常薄：它只负责"形状和类型对齐"，
不做任何渲染或计算。这样契约一旦要改，只改这一个地方。
"""

import numpy as np


def make_observation(image, cmd):
    """组装一条观测。

    参数：
        image: (H, W, 3) 的图像，数值任意，会被转成 float32
        cmd:   标量指令（-1 / 0 / +1）
    返回：
        dict，符合 OpenBot 观测契约
    """
    return {
        "image": np.asarray(image, dtype=np.float32),
        "cmd": np.array([float(cmd)], dtype=np.float32),
    }
