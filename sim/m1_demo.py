"""M1 组合演示：state + dynamics + world 第一次串起来。

让车从道路左侧出发、一直往右偏，观察它什么时候冲出道路。
这证明：世界模块让"位置"变得有意义了。

运行：
    cd OpenBot/sim && python3 m1_demo.py
"""

from state import State
from dynamics import DiffDrive
from world import World


def main():
    state = State(x=0.0, y=0.0, theta=0.0)   # 出发点：道路正中央
    car = DiffDrive()
    world = World(road_half_width=0.9)

    print(f"{world}\n")
    print(f"{'step':>4} | {'x':>6} {'y':>6} {'theta':>7} | {'lateral':>7} | on_road")

    for step in range(60):
        # 左轮快、右轮慢 -> 车向右偏（y 减小）
        car.step(state, left=1.0, right=0.95)

        x, y, theta = state.pose()
        on_road = world.is_on_road(x, y)
        lateral = world.lateral_offset(x, y)

        if step % 3 == 0 or not on_road:
            print(
                f"{step:>4} | {x:>6.2f} {y:>6.2f} {theta:>7.2f} | "
                f"{lateral:>7.2f} | {on_road}"
            )

        if not on_road:
            print(
                f"\n冲出道路！位置 (x={x:.2f}, y={y:.2f})，"
                f"越界 {world.off_road_by(x, y):.2f} m"
            )
            break


if __name__ == "__main__":
    main()
