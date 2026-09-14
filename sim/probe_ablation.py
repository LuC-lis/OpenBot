"""消融实验：cil_mobile 里到底是哪个部件害它"不看图"？

对照组：同一个内存数据集、同一 seed、同样训练轮数，只改一个部件。

变体：
    cil                     原始 cil_mobile
    cil_nodropout           去掉所有 dropout
    cil_nobn                去掉 batch norm
    cil_nocmdconcat         去掉输出前那两次 cmd 拼接
    cil_nodropout_nobn      去 dropout + 去 bn
    cil_all_off             三者全关（应该≈普通 CNN+MLP）
    simple                  参照：简单架构

判据（cmd=0，横向扫描）：
    span = 转向(右) - 转向(左)
    老师 ≈ -0.466（偏右就左转，差 0.466）
    span 越接近 -0.466，说明越会"看图转向"；≈ 0 表示忽略图像。

用法：
    cd OpenBot/sim && ./gpu_python.sh probe_ablation.py --n 4000 --epochs 20 --zero-cmd
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

K = tf.keras
L = tf.keras.layers


def build(arch):
    if arch == "simple":
        base = ob_models.create_cnn(256, 96, 3)
        ii = K.Input((96, 256, 3))
        ci = K.Input((1,))
        f = base(ii)
        x = L.Concatenate()([f, ci])
        x = L.Dense(64, activation="relu")(x)
        return K.Model([ii, ci], L.Dense(2, activation="linear")(x))

    no_dropout = "nodropout" in arch or "all_off" in arch
    no_bn = "nobn" in arch or "all_off" in arch
    no_concat = "nocmdconcat" in arch or "all_off" in arch

    drop = 0.0 if no_dropout else 0.5
    mlp = ob_models.create_mlp(1, 16, 16, dropout=drop, name="cmd")
    cnn = ob_models.create_cnn(
        256, 96, 3,
        cnn_filters=(32, 64, 96, 128, 256),
        kernel_sz=(5, 3, 3, 3, 3),
        stride=(2, 2, 2, 2, 2),
        padding="same",
        activation="relu",
        conv_dropout=(0.0 if no_dropout else 0.2),
        mlp_filters=(128, 64),
        mlp_dropout=drop,
        bn=not no_bn,
    )
    x = L.Concatenate()([mlp.output, cnn.output])
    x = L.Dense(64, activation="relu")(x)
    if not no_dropout:
        x = L.Dropout(0.5)(x)
    if not no_concat:
        x = L.Concatenate()([mlp.input, x])
    x = L.Dense(16, activation="relu")(x)
    if not no_dropout:
        x = L.Dropout(0.5)(x)
    if not no_concat:
        x = L.Concatenate()([mlp.input, x])
    out = L.Dense(2, activation="linear")(x)
    return K.Model([cnn.input, mlp.input], out)


def gen(n, seed=0, zero_cmd=True):
    rng = np.random.default_rng(seed)
    world, cam, teacher = World(0.9), Camera((96, 256)), TeacherPolicy()
    X = np.zeros((n, 96, 256, 3), np.float32)
    C = np.zeros((n, 1), np.float32)
    Y = np.zeros((n, 2), np.float32)
    for i in range(n):
        y = rng.uniform(-0.72, 0.72)
        th = rng.uniform(-0.4, 0.4)
        cmd = 0.0 if zero_cmd else float(rng.choice([-1.0, 0.0, 1.0]))
        img = cam.render(State(0.0, y, th), world)
        info = {"state": (0.0, y, th), "lateral": y, "heading": th, "target_lateral": -cmd * 0.6}
        X[i], C[i, 0], Y[i] = img, cmd, teacher.act({"image": img, "cmd": np.array([cmd], np.float32)}, info)
    return X, C, Y


def span(model):
    world, cam = World(0.9), Camera((96, 256))
    turns = []
    for y in (-0.6, -0.3, 0.0, 0.3, 0.6):
        img = cam.render(State(0.0, y, 0.0), world)
        p = model([img[None, ...], np.array([[0.0]], np.float32)]).numpy()[0]
        turns.append(float(p[1] - p[0]))
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zero-cmd", action="store_true")
    args = ap.parse_args()

    print(f"生成数据集 n={args.n} zero_cmd={args.zero_cmd} ...")
    X, C, Y = gen(args.n, zero_cmd=args.zero_cmd)

    # 老师参照
    world, cam, teacher = World(0.9), Camera((96, 256)), TeacherPolicy()
    tt = []
    for y in (-0.6, -0.3, 0.0, 0.3, 0.6):
        img = cam.render(State(0.0, y, 0.0), world)
        a = teacher.act({"image": img, "cmd": np.array([0.0], np.float32)},
                        {"state": (0, y, 0), "lateral": y, "heading": 0, "target_lateral": 0})
        tt.append(float(a[1] - a[0]))
    print(f"老师转向序列(左→右): {[round(t,3) for t in tt]}  span={tt[-1]-tt[0]:+.3f}\n")

    variants = ["cil", "cil_nodropout", "cil_nobn", "cil_nocmdconcat",
                "cil_nodropout_nobn", "cil_all_off", "simple"]

    print(f"{'变体':<24}{'train_MAE':>10}{'span':>9}  转向序列(左→右)")
    for arch in variants:
        tf.keras.utils.set_random_seed(0)
        ds = tf.data.Dataset.from_tensor_slices(((X, C), Y)).shuffle(2000).batch(32).prefetch(2)
        model = build(arch)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse", metrics=["mae"])
        h = model.fit(ds, epochs=args.epochs, verbose=0)
        tr = span(model)
        print(f"{arch:<24}{h.history['mae'][-1]:>10.4f}{tr[-1]-tr[0]:>+9.3f}  {[round(t,3) for t in tr]}")


if __name__ == "__main__":
    main()
