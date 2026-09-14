"""迷你卷积演示（纯 numpy）

目的：亲眼看清 CNN 的核心运算 —— 卷积（convolution）到底在做什么。

内容：
  1) 用一个 3x3 的小滤波器，在 8x8 的小图上滑动，手算出结果
  2) 对比参数量：卷积 vs 全连接，看卷积有多"省"
  3) 拿我们仿真相机拍的真实画面，用边缘滤波器找车道线

运行：
    cd OpenBot/sim && python3 tiny_cnn.py
输出：
    tiny_cnn.png
"""

import numpy as np

from camera import Camera
from state import State
from world import World


# ======================================================================
# 1. 卷积：一个小窗口在图上滑动，每到一处做"逐元素相乘再求和"
# ======================================================================
def conv2d(image, kernel, stride=1):
    """最朴素的二维卷积。image: (H,W)，kernel: (kh,kw)。"""
    kh, kw = kernel.shape
    out_h = (image.shape[0] - kh) // stride + 1
    out_w = (image.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = np.sum(patch * kernel)   # 逐元素乘，再全部加起来 -> 一个数
    return out


def demo_small():
    print("=" * 60)
    print("1) 小实验：3x3 滤波器 在 8x8 图上滑动")
    print("=" * 60)

    # 一张小图：中间有一条亮竖线（模拟"车道线"）
    img = np.zeros((8, 8))
    img[:, 4] = 1.0
    print("\n输入图像（0=黑，1=白），第 4 列是亮竖线：")
    print(img.astype(int))

    # 竖直边缘滤波器：左边 -1，右边 +1
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], dtype=float)
    print("\n滤波器（检测'竖直边缘'）：")
    print(kernel.astype(int))

    out = conv2d(img, kernel)
    print("\n输出（特征图）：")
    print(out.astype(int))
    print("\n注意：亮竖线两侧出现了 +3 / -3 —— 滤波器'看到'了边缘！")
    print("这就是卷积：用一个模板去图里找它认识的图案。")

    # 参数量对比
    n_conv = kernel.size
    n_fc = img.size * out.size   # 若用全连接：输入 64 -> 输出 36
    print(f"\n参数量对比：")
    print(f"  卷积（3x3 滤波器，权重共享）      : {n_conv} 个权重")
    print(f"  全连接（64 输入 -> 36 输出）      : {n_fc} 个权重")
    print(f"  -> 卷积省了 {n_fc / n_conv:.0f} 倍，而且位置无关！")


# ======================================================================
# 2. 用真实相机画面做边缘检测
# ======================================================================
def demo_real_frame():
    print("\n" + "=" * 60)
    print("2) 真实仿真画面 + 边缘滤波")
    print("=" * 60)

    world = World(road_half_width=0.9)
    camera = Camera(image_size=(96, 256))
    state = State(x=0.0, y=0.0, theta=0.0)

    img = camera.render(state, world)          # (96, 256, 3) float [0,1]
    gray = img.mean(axis=2)                    # 转灰度 (96, 256)

    k_vert = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float)   # 竖直边缘
    k_horz = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float)   # 水平边缘

    fv = np.abs(conv2d(gray, k_vert))
    fh = np.abs(conv2d(gray, k_horz))
    print(f"输入灰度图形状: {gray.shape}")
    print(f"竖直边缘特征图形状: {fv.shape}  -> 高亮'车道线/路沿'")
    print(f"水平边缘特征图形状: {fh.shape}  -> 高亮'地平线/横向变化'")

    # 保存对比图
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(12, 2.4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("原始画面（灰度）", fontsize=10)
    axes[1].imshow(fv, cmap="hot")
    axes[1].set_title("竖直边缘滤波器输出", fontsize=10)
    axes[2].imshow(fh, cmap="hot")
    axes[2].set_title("水平边缘滤波器输出", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("tiny_cnn.png", dpi=120)
    print("已保存 tiny_cnn.png")


if __name__ == "__main__":
    demo_small()
    demo_real_frame()
