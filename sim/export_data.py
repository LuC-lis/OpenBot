"""M6：数据工厂 —— 让老师开车，导出 OpenBot 训练数据。

产物目录结构（与 OpenBot 的 policy/ 训练管线兼容）：

    dataset/
      train_data/
        sim/
          session_000/
            images/
              <timestamp>_crop.jpeg      # 相机画面
            sensor_data/
              rgbFrames.txt              # <timestamp> <frame_id>
              ctrlLog.txt                # <timestamp> <left> <right>
              indicatorLog.txt           # <timestamp> <cmd>
              matched_frame_ctrl_cmd_processed.txt   # 便于人工检查
      test_data/
        ...

关键格式约定（来自 policy/openbot/associate_frames.py）：
    - 图像文件名 = <frame_id>_crop.jpeg，frame_id 通常就是时间戳
    - ctrlLog 的 left/right 是 [-255,255] 的整数（训练时再 /255 归一化）
    - indicatorLog 的 cmd 是 {-1,0,1} 的整数
    - 时间戳间隔远大于匹配阈值(1000)，保证一帧精确匹配一条控制/指令

运行：
    cd OpenBot/sim && python3 export_data.py --out dataset
"""

import argparse
import os

import numpy as np

from env import OpenBotEnv
from policies import TeacherPolicy

# 时间戳间隔（纳秒）。取 1e8 = 0.1 秒，正好对应 10Hz 控制频率。
# 远大于 associate_frames 的 max_offset=1e3，保证一一匹配。
TS_STEP = 100_000_000


def export_session(env, policy, session_dir, steps, seed, frame_id_start=0):
    """跑一局，把每一帧存成 OpenBot 格式。返回本局帧数。"""
    images_dir = os.path.join(session_dir, "images")
    sensor_dir = os.path.join(session_dir, "sensor_data")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sensor_dir, exist_ok=True)

    from PIL import Image

    rgb_lines, ctrl_lines, cmd_lines, proc_lines = [], [], [], []

    obs, info = env.reset(seed=seed)
    frame_id = frame_id_start
    for _ in range(steps):
        action = policy.act(obs, info)                       # 老师给动作
        left_i = int(round(float(action[0]) * 255))          # 归一化 -> PWM 整数
        right_i = int(round(float(action[1]) * 255))
        cmd_i = int(round(float(obs["cmd"][0])))
        ts = frame_id * TS_STEP

        # 存图：名字 = 时间戳，扩展名前加 _crop
        name = str(ts)
        Image.fromarray((obs["image"] * 255).astype(np.uint8)).save(
            os.path.join(images_dir, name + "_crop.jpeg"), quality=95
        )

        rgb_lines.append(f"{ts} {name}")
        ctrl_lines.append(f"{ts} {left_i} {right_i}")
        cmd_lines.append(f"{ts} {cmd_i}")
        proc_lines.append(
            f"{ts},{os.path.abspath(os.path.join(images_dir, name + '_crop.jpeg'))},"
            f"{left_i},{right_i},{cmd_i}"
        )

        obs, reward, terminated, truncated, info = env.step(action)
        frame_id += 1
        if terminated or truncated:
            break

    with open(os.path.join(sensor_dir, "rgbFrames.txt"), "w") as f:
        f.write("timestamp,frame\n")
        f.write("\n".join(rgb_lines) + "\n")
    with open(os.path.join(sensor_dir, "ctrlLog.txt"), "w") as f:
        f.write("timestamp,left,right\n")
        f.write("\n".join(ctrl_lines) + "\n")
    with open(os.path.join(sensor_dir, "indicatorLog.txt"), "w") as f:
        f.write("timestamp,cmd\n")
        f.write("\n".join(cmd_lines) + "\n")
    # 顺带写一份 processed（OpenBot 也会自己生成，留着方便检查）
    with open(os.path.join(sensor_dir, "matched_frame_ctrl_cmd_processed.txt"), "w") as f:
        f.write("timestamp,frame,left,right,cmd\n")
        f.write("\n".join(proc_lines) + "\n")

    return len(rgb_lines)


def export_split(split, root, episodes, steps, base_seed):
    env = OpenBotEnv(seed=base_seed)
    policy = TeacherPolicy()
    total = 0
    for ep in range(episodes):
        session_dir = os.path.join(root, split + "_data", "sim", f"session_{ep:03d}")
        n = export_session(env, policy, session_dir, steps, seed=base_seed + ep)
        total += n
        print(f"  {split} 第 {ep} 局: {n} 帧 -> {session_dir}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--train-episodes", type=int, default=8)
    ap.add_argument("--test-episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"导出到 {args.out}/ ...")
    n_train = export_split("train", args.out, args.train_episodes, args.steps, args.seed)
    n_test = export_split("test", args.out, args.test_episodes, args.steps, args.seed + 1000)
    print(f"\n完成：训练 {n_train} 帧，测试 {n_test} 帧")

    print("\n目录结构：")
    for dirpath, dirnames, filenames in os.walk(args.out):
        dirnames.sort()
        depth = dirpath.count(os.sep) - args.out.count(os.sep)
        print("  " * depth + os.path.basename(dirpath) + "/")
        if os.path.basename(dirpath) == "sensor_data":
            for fn in sorted(filenames):
                print("  " * (depth + 1) + fn)


if __name__ == "__main__":
    main()
