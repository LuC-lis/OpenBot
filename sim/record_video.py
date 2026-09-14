"""录制驾驶视频（GIF）：左边=学生相机视角，右边=俯视轨迹。

用法：
    cd OpenBot/sim
    ./gpu_python.sh record_video.py --model ../policy/models/dg_s3.tflite --out drive_dg_s3
    ./gpu_python.sh record_video.py --teacher --out drive_teacher      # 看老师开
输出：
    <out>.gif
"""

import argparse

import numpy as np

from env import OpenBotEnv
from policies import TeacherPolicy, TFLitePolicy


class StraightPolicy:
    def act(self, obs, info):
        return np.array([0.6, 0.6], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--straight", action="store_true")
    ap.add_argument("--out", default="drive")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="每隔几帧取一帧，用于减小文件")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    env = OpenBotEnv(seed=args.seed)
    if args.teacher:
        policy = TeacherPolicy()
        title = "老师（上帝视角）"
    elif args.straight:
        policy = StraightPolicy()
        title = "直线策略"
    else:
        policy = TFLitePolicy(args.model)
        title = f"学生：{args.model.split('/')[-1]}（只看图像）"

    obs, info = env.reset(seed=args.seed)

    cam_frames, xs, ys, targets, cmds = [], [], [], [], []
    for _ in range(args.steps):
        action = policy.act(obs, info)
        obs, rew, term, trunc, info = env.step(action)
        cam_frames.append((obs["image"] * 255).astype(np.uint8))
        xs.append(info["state"][0])
        ys.append(info["state"][1])
        targets.append(info["target_lateral"])
        cmds.append(info["cmd"])
        if term or trunc:
            break

    n = len(cam_frames)
    print(f"录制 {n} 帧（{title}）")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    half = env.world.road_half_width
    XMAX = max(35.0, max(xs) + 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
    fig.suptitle(title, fontsize=11)

    im = ax1.imshow(cam_frames[0])
    ax1.axis("off")
    ax1.set_title("相机视角（学生看到的）", fontsize=9)

    ax2.axvspan(-half, half, color="0.88")
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(0, XMAX)
    ax2.set_xlabel("y 横向 (m)  +左 / -右", fontsize=8)
    ax2.set_ylabel("x 前进 (m)", fontsize=8)
    ax2.set_title("俯视轨迹", fontsize=9)
    (traj_line,) = ax2.plot([], [], "-b", lw=1.5, label="轨迹")
    (target_scat,) = ax2.plot([], [], "k--", lw=0.8, label="目标位置")
    (car_dot,) = ax2.plot([], [], "o", color="blue", ms=6)
    ax2.legend(fontsize=7, loc="upper right")
    txt = ax2.text(0.02, 0.02, "", transform=ax2.transAxes, fontsize=8, va="bottom")

    idx = list(range(0, n, args.stride))

    def update(k):
        i = idx[k]
        im.set_data(cam_frames[i])
        traj_line.set_data(ys[: i + 1], xs[: i + 1])
        target_scat.set_data(targets[: i + 1], xs[: i + 1])
        car_dot.set_data([ys[i]], [xs[i]])
        txt.set_text(f"step {i:3d}   cmd={cmds[i]:+.0f}   横向={ys[i]:+.2f}")
        return im, traj_line, target_scat, car_dot, txt

    ani = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / args.fps)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{args.out}.gif"
    ani.save(out, writer="pillow", fps=args.fps)
    plt.close(fig)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
