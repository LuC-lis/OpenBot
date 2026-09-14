"""dropout 扫描：正则化强度 vs "看图转向"能力。

固定 cil_mobile 的其它部件（bn=True、cmd 拼接保留），只改 dropout，
从 0.0 扫到 0.5，看模型能否学会"看图转向"。

判据：
  val_MAE —— 在没见过的样本上的预测误差
  span    —— 横向扫描时转向的变化幅度；老师 ≈ -0.467，≈0 表示忽略图像

输出：
  dropout_sweep.png  曲线图

用法：
    cd OpenBot/sim && ./gpu_python.sh probe_dropout.py
"""

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

K = tf.keras
L = tf.keras.layers
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def build_cil(drop):
    mlp = ob_models.create_mlp(1, 16, 16, dropout=drop, name="cmd")
    cnn = ob_models.create_cnn(
        256, 96, 3,
        cnn_filters=(32, 64, 96, 128, 256),
        kernel_sz=(5, 3, 3, 3, 3),
        stride=(2, 2, 2, 2, 2),
        padding="same", activation="relu",
        conv_dropout=drop, mlp_filters=(128, 64), mlp_dropout=drop, bn=True,
    )
    x = L.Concatenate()([mlp.output, cnn.output])
    x = L.Dense(64, activation="relu")(x)
    if drop > 0:
        x = L.Dropout(drop)(x)
    x = L.Concatenate()([mlp.input, x])
    x = L.Dense(16, activation="relu")(x)
    if drop > 0:
        x = L.Dropout(drop)(x)
    x = L.Concatenate()([mlp.input, x])
    out = L.Dense(2, activation="linear")(x)
    return K.Model([cnn.input, mlp.input], out)


def gen(n, seed=0):
    rng = np.random.default_rng(seed)
    world, cam, teacher = World(0.9), Camera((96, 256)), TeacherPolicy()
    X = np.zeros((n, 96, 256, 3), np.float32)
    C = np.zeros((n, 1), np.float32)
    Y = np.zeros((n, 2), np.float32)
    for i in range(n):
        y = rng.uniform(-0.72, 0.72)
        th = rng.uniform(-0.4, 0.4)
        img = cam.render(State(0.0, y, th), world)
        info = {"state": (0.0, y, th), "lateral": y, "heading": th, "target_lateral": 0.0}
        X[i], C[i, 0], Y[i] = img, 0.0, teacher.act({"image": img, "cmd": np.array([0.0], np.float32)}, info)
    return X, C, Y


def span(model):
    world, cam = World(0.9), Camera((96, 256))
    turns = []
    for y in (-0.6, -0.3, 0.0, 0.3, 0.6):
        img = cam.render(State(0.0, y, 0.0), world)
        p = model([img[None, ...], np.array([[0.0]], np.float32)]).numpy()[0]
        turns.append(float(p[1] - p[0]))
    return turns[-1] - turns[0]


def main():
    n = 5000
    X, C, Y = gen(n)
    ntr = int(0.8 * n)
    Xtr, Ctr, Ytr = X[:ntr], C[:ntr], Y[:ntr]
    Xva, Cva, Yva = X[ntr:], C[ntr:], Y[ntr:]
    ds = tf.data.Dataset.from_tensor_slices(((Xtr, Ctr), Ytr)).shuffle(2000).batch(32).prefetch(2)

    results = []
    print(f"{'dropout':>8}{'train_MAE':>11}{'val_MAE':>10}{'span':>9}")
    for d in DROPOUTS:
        tf.keras.utils.set_random_seed(0)
        model = build_cil(d)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        h = model.fit(ds, epochs=25, verbose=0)
        vmae = float(np.mean(np.abs(model.predict([Xva, Cva], verbose=0) - Yva)))
        s = span(model)
        results.append((d, h.history["mae"][-1], vmae, s))
        print(f"{d:>8.1f}{h.history['mae'][-1]:>11.4f}{vmae:>10.4f}{s:>+9.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ds_ = [r[0] for r in results]
    vmae = [r[2] for r in results]
    sp = [r[3] for r in results]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ds_, vmae, "o-", color="tab:red", label="验证误差 val_MAE")
    ax1.set_xlabel("dropout")
    ax1.set_ylabel("val_MAE（越低越好）", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.axhline(-0.467, ls="--", color="gray", lw=1)
    ax2.plot(ds_, sp, "s-", color="tab:blue", label="看图转向幅度 span")
    ax2.set_ylabel("span（越接近 -0.467 越好）", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("dropout 强度 vs 模型能否学会『看图转向』")
    fig.tight_layout()
    fig.savefig("dropout_sweep.png", dpi=120)
    print("\n已保存 dropout_sweep.png")


if __name__ == "__main__":
    main()
