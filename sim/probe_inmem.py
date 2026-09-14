"""决定性实验：内存内数据 vs 文件数据集。

目的：定位"模型忽略图像"的原因。
  假设A：保存/读取数据集时，图像与标签错位了。
  假设B：相机图像 / cil_mobile 网络本身学不到这个映射。

做法：完全绕开 JPEG、文件路径、associate_frames，
      直接在内存里渲染图像并让老师给标签，用同样的网络训练。
      然后做横向扫描：固定 cmd，改变横向偏移，看输出的转向是否跟着变。

若【学会了】→ 是数据集保存/读取环节（假设 A）
若【仍不会】→ 是相机/网络环节（假设 B）

用法：
    cd OpenBot/sim && ../sim/gpu_python.sh probe_inmem.py --n 4000 --epochs 20
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "policy")))
from openbot import models as ob_models  # noqa: E402

from camera import Camera  # noqa: E402
from policies import TeacherPolicy  # noqa: E402
from state import State  # noqa: E402
from world import World  # noqa: E402


def gen_dataset(n, seed=0, zero_cmd=False):
    rng = np.random.default_rng(seed)
    world = World(0.9)
    cam = Camera((96, 256))
    teacher = TeacherPolicy()

    X = np.zeros((n, 96, 256, 3), np.float32)   # 图像
    C = np.zeros((n, 1), np.float32)            # 指令
    Y = np.zeros((n, 2), np.float32)            # 老师动作

    for i in range(n):
        y = rng.uniform(-0.72, 0.72)
        th = rng.uniform(-0.4, 0.4)
        cmd = float(rng.choice([-1.0, 0.0, 1.0]))
        if zero_cmd:
            cmd = 0.0
        st = State(0.0, y, th)
        img = cam.render(st, world)
        info = {"state": (0.0, y, th), "lateral": y, "heading": th,
                "target_lateral": -cmd * 0.6}
        X[i] = img
        C[i, 0] = cmd
        Y[i] = teacher.act({"image": img, "cmd": np.array([cmd], np.float32)}, info)
    return X, C, Y


def sweep(model, tag, ymean=None):
    print(f"\n--- 横向扫描（{tag}，模型应随 lateral 改变转向）---")
    world = World(0.9)
    cam = Camera((96, 256))
    teacher = TeacherPolicy()
    if ymean is None:
        ymean = np.zeros(2, np.float32)
    for cmd in (0.0, 1.0):
        print(f"  cmd={cmd}")
        for y in (-0.6, -0.3, 0.0, 0.3, 0.6):
            st = State(0.0, y, 0.0)
            img = cam.render(st, world)
            info = {"state": (0.0, y, 0.0), "lateral": y, "heading": 0.0,
                    "target_lateral": -cmd * 0.6}
            t = teacher.act({"image": img, "cmd": np.array([cmd], np.float32)}, info)
            p = model([img[None, ...], np.array([[cmd]], np.float32)]).numpy()[0] + ymean
            print(f"    lateral {y:+.1f} | 老师[{t[0]:.3f},{t[1]:.3f}] | 模型[{p[0]:.3f},{p[1]:.3f}] "
                  f"| 转向差 模型{p[1]-p[0]:+.3f} 老师{t[1]-t[0]:+.3f}")


def build_model(arch):
    if arch == "cil":
        return ob_models.cil_mobile(256, 96, bn=True)
    # simple: 小 CNN（无 dropout/bn）+ 拼接 cmd + 全连接
    base = ob_models.create_cnn(256, 96, 3)
    ii = tf.keras.Input((96, 256, 3))
    ci = tf.keras.Input((1,))
    f = base(ii)
    x = tf.keras.layers.Concatenate()([f, ci])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    o = tf.keras.layers.Dense(2, activation="linear")(x)
    return tf.keras.Model([ii, ci], o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zero-cmd", action="store_true")
    ap.add_argument("--center", action="store_true", help="标签去中心化（只学转向）")
    ap.add_argument("--arch", choices=["cil", "simple"], default="cil")
    args = ap.parse_args()

    print(f"生成 {args.n} 个内存样本... (zero_cmd={args.zero_cmd}, center={args.center})")
    X, C, Y = gen_dataset(args.n, zero_cmd=args.zero_cmd)
    ymean = Y.mean(axis=0) if args.center else None
    if ymean is not None:
        Y = Y - ymean
        print("标签均值:", ymean)
    print("X", X.shape, "C", C.shape, "Y", Y.shape)

    ds = (
        tf.data.Dataset.from_tensor_slices(((X, C), Y))
        .shuffle(2000)
        .batch(args.batch)
        .prefetch(2)
    )

    model = build_model(args.arch)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse", metrics=["mae"])
    model.fit(ds, epochs=args.epochs, verbose=2)

    sweep(model, f"内存数据/{args.arch}", ymean)


if __name__ == "__main__":
    main()
