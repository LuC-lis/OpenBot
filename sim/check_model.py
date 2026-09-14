"""检查一个 .tflite 模型能否直接放进本仿真环境使用。

仿真对模型的要求（接口契约）：
    img: (1, 96, 256, 3)  float32   # 相机图像，取值 [0,1]
    cmd: (1, 1)           float32   # 指令 -1/0/+1
    out: (1, 2)           float32   # [left, right] ∈ [-1,1]

用法：
    cd OpenBot/sim && ./gpu_python.sh check_model.py --model ../policy/models/dg_s3.tflite
"""

import argparse

import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    print(f"模型: {args.model}\n")
    it = tf.lite.Interpreter(model_path=args.model)
    it.allocate_tensors()

    ins = it.get_input_details()
    outs = it.get_output_details()

    print("输入:")
    img = cmd = None
    for d in ins:
        print(f"  {d['name']:<40} shape={list(d['shape'])} dtype={np.dtype(d['dtype']).name}")
        if len(d["shape"]) == 4:
            img = d
        elif len(d["shape"]) in (1, 2):
            cmd = d
    print("输出:")
    for d in outs:
        print(f"  {d['name']:<40} shape={list(d['shape'])} dtype={np.dtype(d['dtype']).name}")

    # --- 逐项判定 ---
    print("\n判定:")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK ' if cond else 'FAIL'}] {name} {detail}")

    check("有 4D 图像输入", img is not None)
    if img is not None:
        s = list(img["shape"])
        check(f"图像形状 = [1,{args.height},{args.width},3]", s == [1, args.height, args.width, 3], f"(实际 {s})")
        check("图像 dtype = float32", img["dtype"] == np.float32)

    check("有 cmd 输入（1 个标量）", cmd is not None,
          f"(shape {list(cmd['shape']) if cmd else None})")
    check("输出形状 = [1,2]", list(outs[0]["shape"]) == [1, 2], f"(实际 {list(outs[0]['shape'])})")

    # --- 真的跑一次（用真实渲染画面，而不是全黑图）---
    if ok:
        try:
            from camera import Camera
            from state import State
            from world import World

            frame = Camera((args.height, args.width)).render(State(0.0, 0.0, 0.0), World(0.9))
            it.set_tensor(img["index"], frame[None, ...].astype(np.float32))
        except Exception:
            it.set_tensor(img["index"], np.zeros((1, args.height, args.width, 3), np.float32))
        it.set_tensor(cmd["index"], np.zeros((1, 1), np.float32))
        it.invoke()
        y = it.get_tensor(outs[0]["index"])[0]
        print(f"\n  居中直行时的输出: [{y[0]:.3f}, {y[1]:.3f}]  （期望接近 [0.6, 0.6]）")
        check("输出数值合理（在 [-1,1] 内）", bool(np.all(np.abs(y) <= 1.0)), f"(实际 {y})")

    print("\n" + ("✅ 可以直接放进仿真：" if ok else "❌ 不兼容，需调整："))
    if ok:
        print(f"   ./gpu_python.sh m5_demo.py --model {args.model}")
        print(f"   ./gpu_python.sh eval_student.py --model {args.model} --episodes 20 --disturb 0.02")
    else:
        print("   - 图像尺寸不符：训练时用 --height/--width 匹配，或改 Camera(image_size=...)")
        print("   - 缺少 cmd 输入 / 输出不是 2：不是本项目的驾驶模型")


if __name__ == "__main__":
    main()
