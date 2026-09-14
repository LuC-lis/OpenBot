"""M2 组合演示：world + camera + observation。

渲染车在不同横向位置、不同朝向时"看到"的画面，拼成一张图。
并验证观测的形状与取值范围符合契约。

运行：
    cd OpenBot/sim && python3 m2_demo.py
输出：
    m2_views.png
"""

import numpy as np

from state import State
from world import World
from camera import Camera
from observation import make_observation


# (说明, y, theta)：横向位置 / 朝向
CASES = [
    ("居中直行",      0.00,  0.00),
    ("偏左 0.6m",     0.60,  0.00),
    ("偏右 0.6m",    -0.60,  0.00),
    ("车头朝左",      0.00,  0.30),
    ("车头朝右",      0.00, -0.30),
    ("快到左边界",    0.80,  0.00),
]


def main():
    world = World(road_half_width=0.9)
    camera = Camera(image_size=(96, 256))
    state = State()

    frames = []
    for _, y, theta in CASES:
        state.reset(x=0.0, y=y, theta=theta)
        image = camera.render(state, world)

        # 用观测组装模块打包，并校验契约
        obs = make_observation(image, cmd=0.0)
        assert obs["image"].shape == (96, 256, 3), obs["image"].shape
        assert obs["image"].dtype == np.float32
        assert 0.0 <= obs["image"].min() and obs["image"].max() <= 1.0
        assert obs["cmd"].shape == (1,)

        frames.append(obs["image"])

    print("图像形状:", frames[0].shape, " dtype:", frames[0].dtype)
    print("取值范围: [%.2f, %.2f]" % (frames[0].min(), frames[0].max()))
    print("观测键:", list(make_observation(frames[0], 0.0).keys()))

    # 拼图保存
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 让图里能显示中文（本机有 Noto CJK 字体）
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(12, 4))
    for ax, (name, _, _), img in zip(axes.ravel(), CASES, frames):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("m2_views.png", dpi=120)
    print("已保存 m2_views.png")


if __name__ == "__main__":
    main()
