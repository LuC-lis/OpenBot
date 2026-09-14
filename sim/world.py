"""模块 3：世界 / 地图（World）

只回答几何问题，不关心是谁在问、也不关心车怎么动：
    - 一个点 (x, y) 偏离道路中心多少？
    - 这个点在道路上吗？
    - 离线 / 离中心还有多远？

因此它**不依赖** state 或 dynamics。传给它数字 (x, y) 即可。

几何（M1 阶段最简单的情形）：
    一条沿 +x 的无限长直道，中心线是 y = 0，宽度为 2 * road_half_width。
    +y 为车的左侧，所以 lateral_offset > 0 表示偏左，< 0 表示偏右。
"""


class World:
    def __init__(self, road_half_width=0.9):
        # 道路半宽（米）。总宽度 = 2 * road_half_width。
        self.road_half_width = float(road_half_width)

    # ------------------------------------------------------------------
    # 几何查询
    # ------------------------------------------------------------------
    def centerline_y(self, x):
        """道路中心线在给定 x 处的 y 坐标。

        M1 是直道，所以恒为 0。留成函数是为了以后换成弯道时
        只需要改这里（例如返回 x 的正弦），其余代码不用动。
        """
        return 0.0

    def lateral_offset(self, x, y):
        """点 (x, y) 相对中心线的横向偏移（有符号）。

        正值 = 偏左，负值 = 偏右。
        """
        return y - self.centerline_y(x)

    def is_on_road(self, x, y):
        """点是否在道路上。"""
        return abs(self.lateral_offset(x, y)) <= self.road_half_width

    def distance_to_edge(self, x, y):
        """到最近道路边线的距离。

        道路上为正，道路外为负。
        """
        return self.road_half_width - abs(self.lateral_offset(x, y))

    def off_road_by(self, x, y):
        """冲出道路的距离：在路上返回 0，路外返回越界的正数。"""
        return max(0.0, abs(self.lateral_offset(x, y)) - self.road_half_width)

    def __repr__(self):
        return f"World(road_half_width={self.road_half_width})"


# ----------------------------------------------------------------------
# 直接运行时，验证几何查询
# ----------------------------------------------------------------------
if __name__ == "__main__":
    w = World(road_half_width=0.9)
    print(w)

    for y in (0.0, 0.5, 0.9, 1.0, -1.5):
        print(
            f"y={y:+.2f}  "
            f"lateral={w.lateral_offset(0.0, y):+.2f}  "
            f"on_road={w.is_on_road(0.0, y)!s:5}  "
            f"to_edge={w.distance_to_edge(0.0, y):+.2f}  "
            f"off_by={w.off_road_by(0.0, y):.2f}"
        )
