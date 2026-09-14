"""模块 2：动力学（Dynamics）

只描述"给了左右轮指令，状态怎么变"。
它依赖模块 1 的 State，但不依赖画面、道路、奖励。

接口对齐真实 OpenBot（见 android/.../vehicle/Control.java）：
    动作 = [left, right]，每个值被夹到 [-1, 1]。

数学（差速驱动）：
    v_l   = MAX_WHEEL_SPEED * left
    v_r   = MAX_WHEEL_SPEED * right
    v     = (v_l + v_r) / 2                 # 车体线速度
    omega = (v_r - v_l) / WHEEL_BASE        # 车体角速度
    theta += omega * dt
    x     += v * cos(theta) * dt
    y     += v * sin(theta) * dt
"""

import math

from state import State, wrap_angle


# ----------------------------------------------------------------------
# 参数：来自真实 OpenBot，之后会统一挪到配置文件
# ----------------------------------------------------------------------
WHEEL_BASE = 0.14        # 左右轮中心间距（米）
MAX_WHEEL_SPEED = 1.2    # |指令| = 1 时车轮线速度（米/秒）
DT = 0.1                 # 控制周期（秒），即 10 Hz
ACTION_LIMIT = 1.0       # 动作范围 [-1, 1]


def clamp(v, lo=-ACTION_LIMIT, hi=ACTION_LIMIT):
    """把值夹到 [lo, hi]，对应 Control.java 里的 Math.max/min。"""
    return max(lo, min(hi, v))


class DiffDrive:
    """差速驱动动力学：作用在一个 State 上，原地更新它。"""

    def __init__(self, wheel_base=WHEEL_BASE, max_wheel_speed=MAX_WHEEL_SPEED, dt=DT):
        self.wheel_base = wheel_base
        self.max_wheel_speed = max_wheel_speed
        self.dt = dt

    def step(self, state, left, right):
        """执行一步，原地更新 state。

        参数：
            state: 模块 1 的 State 对象（会被修改）
            left, right: 左右轮指令，会被夹到 [-1, 1]
        返回：
            (v, omega): 本步车体线速度与角速度
        """
        left = clamp(left)
        right = clamp(right)

        v_l = self.max_wheel_speed * left
        v_r = self.max_wheel_speed * right

        v = 0.5 * (v_l + v_r)
        omega = (v_r - v_l) / self.wheel_base

        # 用中点角度积分，转弯时比直接用旧 theta 更准
        theta_mid = state.theta + 0.5 * omega * self.dt
        state.x += v * math.cos(theta_mid) * self.dt
        state.y += v * math.sin(theta_mid) * self.dt
        state.theta = wrap_angle(state.theta + omega * self.dt)

        return v, omega


# ----------------------------------------------------------------------
# 直接运行时，验证"动力学"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    car = DiffDrive()

    # 1) 两轮同速前进 -> 沿 +x 直走
    s = State()
    for _ in range(10):
        car.step(s, 1.0, 1.0)
    print("直行 10 步 (1, 1): ", s)      # 期望 x≈1.2

    # 2) 左轮快右轮慢 -> 向右转（theta 减小，y 变负）
    s = State()
    for _ in range(10):
        car.step(s, 1.0, 0.5)
    print("右转 10 步 (1, 0.5):", s)

    # 3) 两轮反向 -> 原地打转，位置不变
    s = State()
    for _ in range(10):
        car.step(s, -1.0, 1.0)
    print("原地打转 (-1, 1):  ", s)      # 期望 x≈0, y≈0
