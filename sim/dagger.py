"""DAgger：数据集聚合（Dataset Aggregation）。

流程（每一轮）：
    1. 让【学生】(tflite) 在仿真里开，但以概率 beta 改用【老师】的动作，
       避免它一开始就翻车、轨迹太短（这叫"混合策略"）。
    2. 在它经过的每一个状态，记录 (学生看到的画面, 老师的正确动作, cmd)。
    3. 追加到训练数据集里。
    4. 之后重新训练 —— 学生会学会"犯错后怎么补救"。

关键：训练用的标签始终是【老师】的动作，但状态是【学生自己开出来的】。
这正是 DAgger 相对普通行为克隆的区别。

运行：
    cd OpenBot/sim && conda run -n openbot python dagger.py \
        --model <学生.tflite> --beta 0.5 --episodes 8 --tag d1
输出：
    追加到 ../policy/dataset/train_data/sim/dagger_<tag>_<i>/ ...
"""

import argparse
import os

import numpy as np
from PIL import Image

from env import OpenBotEnv
from policies import TeacherPolicy, TFLitePolicy

TS_STEP = 100_000_000


def collect_session(env, student, teacher, session_dir, steps, seed, beta, rng, disturb=0.0):
    images_dir = os.path.join(session_dir, "images")
    sensor_dir = os.path.join(session_dir, "sensor_data")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sensor_dir, exist_ok=True)

    obs, info = env.reset(seed=seed)
    frame_id = 0
    n_teacher_used = 0
    rgb_lines, ctrl_lines, cmd_lines, proc_lines = [], [], [], []

    for _ in range(steps):
        teacher_action = teacher.act(obs, info)   # 正确动作 -> 作为标签
        student_action = student.act(obs, info)   # 学生自己想做的

        use_teacher = rng.random() < beta
        applied = teacher_action if use_teacher else student_action
        n_teacher_used += int(use_teacher)

        left_i = int(round(float(teacher_action[0]) * 255))   # 标签 = 老师动作
        right_i = int(round(float(teacher_action[1]) * 255))
        cmd_i = int(round(float(obs["cmd"][0])))
        ts = frame_id * TS_STEP
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

        obs, reward, terminated, truncated, info = env.step(applied)
        frame_id += 1

        # 注入扰动，逼出"偏离→纠错"的样本，然后重新观测
        if disturb > 0 and rng.random() < disturb and not (terminated or truncated):
            env.perturb()
            obs, info = env.observe()

        if terminated or truncated:
            break

    with open(os.path.join(sensor_dir, "rgbFrames.txt"), "w") as f:
        f.write("timestamp,frame\n"); f.write("\n".join(rgb_lines) + "\n")
    with open(os.path.join(sensor_dir, "ctrlLog.txt"), "w") as f:
        f.write("timestamp,left,right\n"); f.write("\n".join(ctrl_lines) + "\n")
    with open(os.path.join(sensor_dir, "indicatorLog.txt"), "w") as f:
        f.write("timestamp,cmd\n"); f.write("\n".join(cmd_lines) + "\n")
    with open(os.path.join(sensor_dir, "matched_frame_ctrl_cmd_processed.txt"), "w") as f:
        f.write("timestamp,frame,left,right,cmd\n")
        f.write("\n".join(proc_lines) + "\n")

    return frame_id, n_teacher_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="学生 tflite 模型")
    ap.add_argument("--beta", type=float, default=0.5, help="使用老师动作的概率")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--tag", default="d1")
    ap.add_argument("--disturb", type=float, default=0.0, help="每步注入扰动的概率")
    ap.add_argument("--out", default="../policy/dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = OpenBotEnv(seed=args.seed)
    student = TFLitePolicy(os.path.abspath(args.model))
    teacher = TeacherPolicy()
    rng = np.random.default_rng(args.seed)

    total = 0
    for i in range(args.episodes):
        session_dir = os.path.join(
            args.out, "train_data", "sim", f"dagger_{args.tag}_{i:02d}"
        )
        n, used = collect_session(
            env, student, teacher, session_dir, args.steps, args.seed + i,
            args.beta, rng, args.disturb
        )
        total += n
        print(f"  第 {i} 局: {n} 帧 (其中 {used} 步用了老师动作) -> {session_dir}")
    print(f"\nDAgger 采集完成: 共 {total} 帧，已追加到 {args.out}/train_data")


if __name__ == "__main__":
    main()
