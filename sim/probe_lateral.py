"""探针实验：图像到底能不能"读"出横向偏移？

上一个实验证明：模型只学 E[动作|cmd]，忽略图像。
现在直接问：给定一张渲染图，能不能预测出车的横向偏移？？

  - 若能（误差很小）→ 图像里有信息、CNN 也学得到。
                        那么"忽略图像"是任务/优化层面的捷径问题。
  - 若不能        → 图像或网络本身无法提取这个信息。

用法：
    cd OpenBot/sim && ../sim/gpu_python.sh probe_lateral.py --n 4000 --epochs 20
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "policy")))
from openbot import models as ob_models  # noqa: E402

from camera import Camera  # noqa: E402
from state import State  # noqa: E402
from world import World  # noqa: E402


def gen(n, seed=0):
    rng = np.random.default_rng(seed)
    world = World(0.9)
    cam = Camera((96, 256))
    X = np.zeros((n, 96, 256, 3), np.float32)
    Y = np.zeros((n, 2), np.float32)  # [lateral, heading]
    for i in range(n):
        y = rng.uniform(-0.72, 0.72)
        th = rng.uniform(-0.4, 0.4)
        X[i] = cam.render(State(0.0, y, th), world)
        Y[i] = [y, th]
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    X, Y = gen(args.n)
    ntr = int(0.8 * args.n)
    Xtr, Ytr, Xva, Yva = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    ds = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(2000).batch(32).prefetch(2)

    # 复用 cil_mobile 的 CNN 主干，输出改成 2 个数 [lateral, heading]
    base = ob_models.create_cnn(256, 96, 3)
    inp = tf.keras.Input((96, 256, 3))
    out = tf.keras.layers.Dense(2, activation="linear")(base(inp))
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse", metrics=["mae"])
    model.fit(ds, epochs=args.epochs, verbose=2)

    pred = model.predict(Xva, verbose=0)
    mae_lat = np.mean(np.abs(pred[:, 0] - Yva[:, 0]))
    mae_head = np.mean(np.abs(pred[:, 1] - Yva[:, 1]))
    # 基线：预测均值
    base_lat = np.mean(np.abs(Ytr[:, 0].mean() - Yva[:, 0]))
    base_head = np.mean(np.abs(Ytr[:, 1].mean() - Yva[:, 1]))
    print(f"\n验证集（{len(Xva)} 样本）：")
    print(f"  横向: 网络 MAE {mae_lat:.4f}  |  猜均值基线 {base_lat:.4f}  |  提升 {base_lat/max(mae_lat,1e-9):.1f}x")
    print(f"  航向: 网络 MAE {mae_head:.4f}  |  猜均值基线 {base_head:.4f}  |  提升 {base_head/max(mae_head,1e-9):.1f}x")
    print("\n抽查（真实 -> 预测）：")
    for i in range(6):
        print(f"  lateral {Yva[i,0]:+.3f} -> {pred[i,0]:+.3f} | heading {Yva[i,1]:+.3f} -> {pred[i,1]:+.3f}")


if __name__ == "__main__":
    main()
