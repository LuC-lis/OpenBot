"""模块 1：状态（State）

只描述"车在哪、朝哪"，不含任何运动逻辑、没有画面、没有奖励。
纯数据 + 少量便于使用的辅助方法。

世界坐标约定：
    +x     车头默认方向（theta = 0 时车朝 +x）
    +y     车的左侧
    theta  绕 z 轴的航向角，单位弧度，逆时针为正
"""

import math


def wrap_angle(a):
    """把角度归一化到 (-pi, pi]。

    这是 theta 的"不变量"：无论怎么转，theta 永远落在这个区间，
    这样状态才不会无限增大，比较角度时也不会出错。
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class State:
    """一辆车的状态：位置 (x, y) 与航向 theta。"""

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.reset(x, y, theta)

    def reset(self, x=0.0, y=0.0, theta=0.0):
        """直接设置状态，并保证 theta 已被归一化。"""
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_angle(float(theta))

    def pose(self):
        """返回当前状态 (x, y, theta)。"""
        return self.x, self.y, self.theta

    def copy(self):
        """返回一个副本（避免多个对象共享同一份状态）。"""
        return State(self.x, self.y, self.theta)

    def __repr__(self):
        return f"State(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"


# ----------------------------------------------------------------------
# 直接运行时，验证"状态"本身
# ----------------------------------------------------------------------
if __name__ == "__main__":
    s = State()
    print("初始:", s)

    s.reset(x=1.0, y=-0.5, theta=math.pi + 0.3)   # theta 超出范围，应被归一化
    print("设置后:", s, " (theta 已被 wrap 到 (-pi, pi])")

    t = s.copy()
    t.x = 99.0
    print("副本改了 x，原状态不受影响:", s, "|", t)
