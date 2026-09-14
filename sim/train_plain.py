"""对照实验：用【普通 MSE】而不是 OpenBot 的加权损失来训练。

目的：验证"模型忽略图像"是不是损失函数造成的。
    OpenBot 的 sq_weighted_mse_angle 用 (|左右差|+0.05)^2 加权，
    直行样本权重只有转弯样本的 ~1/49，可能导致网络学不到"看图纠错"。

用法:
    cd OpenBot/sim && conda run -n openbot python train_plain.py --epochs 15
"""

import argparse
import glob
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "policy")))
from openbot import models as ob_models  # noqa: E402


def load_index(data_dir):
    rows = []
    for f in glob.glob(os.path.join(data_dir, "*", "*", "sensor_data",
                                   "matched_frame_ctrl_cmd_processed.txt")):
        for line in list(open(f))[1:]:
            p = line.strip().split(",")
            if len(p) == 5:
                rows.append((p[1], int(p[2]), int(p[3]), int(p[4])))
    return rows


def make_ds(rows, batch, shuffle=2000, zero_cmd=False):
    paths = [r[0] for r in rows]
    left = [r[1] / 255.0 for r in rows]
    right = [r[2] / 255.0 for r in rows]
    cmd = [0.0 if zero_cmd else float(r[3]) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, left, right, cmd))

    def fn(path, l, r, c):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, dtype=tf.float32)
        img.set_shape([96, 256, 3])
        return (img, tf.reshape(tf.cast(c, tf.float32), (1,))), tf.stack([l, r])

    return ds.map(fn, num_parallel_calls=4).shuffle(shuffle).batch(batch).prefetch(2)


def build_model(arch, dropout=0.5):
    if arch == "simple":
        base = ob_models.create_cnn(256, 96, 3)
        ii = tf.keras.Input((96, 256, 3))
        ci = tf.keras.Input((1,))
        f = base(ii)
        x = tf.keras.layers.Concatenate()([f, ci])
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        o = tf.keras.layers.Dense(2, activation="linear")(x)
        return tf.keras.Model([ii, ci], o)
    if arch == "cil":
        L = tf.keras.layers
        mlp = ob_models.create_mlp(1, 16, 16, dropout=dropout, name="cmd")
        cnn = ob_models.create_cnn(
            256, 96, 3,
            cnn_filters=(32, 64, 96, 128, 256), kernel_sz=(5, 3, 3, 3, 3),
            stride=(2, 2, 2, 2, 2), padding="same", activation="relu",
            conv_dropout=dropout, mlp_filters=(128, 64), mlp_dropout=dropout, bn=True,
        )
        x = L.Concatenate()([mlp.output, cnn.output])
        x = L.Dense(64, activation="relu")(x)
        if dropout > 0:
            x = L.Dropout(dropout)(x)
        x = L.Concatenate()([mlp.input, x])
        x = L.Dense(16, activation="relu")(x)
        if dropout > 0:
            x = L.Dropout(dropout)(x)
        x = L.Concatenate()([mlp.input, x])
        out = L.Dense(2, activation="linear")(x)
        return tf.keras.Model([cnn.input, mlp.input], out)
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../policy/dataset/train_data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="../policy/models/plain_mse.tflite")
    ap.add_argument("--zero-cmd", action="store_true", help="把所有 cmd 置 0（强制用图像）")
    ap.add_argument("--arch", choices=["cil", "simple"], default="simple")
    ap.add_argument("--dropout", type=float, default=0.5)
    args = ap.parse_args()

    rows = load_index(args.data)
    print(f"样本数: {len(rows)}")
    ds = make_ds(rows, args.batch, zero_cmd=args.zero_cmd)

    model = build_model(args.arch, args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse", metrics=["mae"])
    model.fit(ds, epochs=args.epochs, verbose=2)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    open(args.out, "wb").write(converter.convert())
    print("已保存", args.out)


if __name__ == "__main__":
    main()
