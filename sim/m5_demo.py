"""M5：把训练好的 .tflite 装回仿真，让学生（只看图像）开车。

对比：
    老师  —— 用上帝视角，完美驾驶
    学生  —— 只吃图像 + cmd（训练出来的神经网络）
    直线  —— 死板基准

运行（需要 tensorflow 环境）：
    cd OpenBot/sim && conda run -n openbot python m5_demo.py
输出：
    m5_trajectory.png
"""

import argparse
import os

import numpy as np

from env import OpenBotEnv
from policies import TeacherPolicy, TFLitePolicy


class StraightPolicy:
    def act(self, obs, info):
        return np.array([0.6, 0.6], dtype=np.float32)


def run_episode(env, policy, seed=0, max_steps=400):
    obs, info = env.reset(seed=seed)
    xs, ys, targets, rewards = [], [], [], []
    for _ in range(max_steps):
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        xs.append(info["state"][0])
        ys.append(info["state"][1])
        targets.append(info["target_lateral"])
        rewards.append(reward)
        if terminated or truncated:
            break
    return {
        "xs": xs, "ys": ys, "targets": targets, "rewards": rewards,
        "terminated": terminated, "reason": info["components"]["reason"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="../policy/models/openbot_cil_mobile_lr0.002_bz32_bn_flip/checkpoints/best.tflite",
    )
    args = ap.parse_args()
    model_path = os.path.abspath(args.model)
    print("加载模型:", model_path)

    env = OpenBotEnv(seed=0)
    policies = {
        "老师": TeacherPolicy(),
        "学生": TFLitePolicy(model_path),
        "直线": StraightPolicy(),
    }

    results = {}
    print(f"\n{'策略':<5}{'步数':>5}{'回报':>9}{'平均|横向|':>11}{'跟踪误差':>10}  结束")
    for name, policy in policies.items():
        r = run_episode(env, policy, seed=0)
        results[name] = r
        ys = np.array(r["ys"]); tg = np.array(r["targets"])
        reason = r["reason"] if r["terminated"] else "timeout"
        print(
            f"{name:<5}{len(ys):>5}{sum(r['rewards']):>9.1f}"
            f"{np.mean(np.abs(ys)):>11.3f}{np.mean(np.abs(ys - tg)):>10.3f}  {reason}"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    half = env.world.road_half_width
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.axvspan(-half, half, color="0.88", label="道路")
    colors = {"老师": "g", "学生": "b", "直线": "orange"}
    for name, r in results.items():
        ax.plot(r["ys"], r["xs"], "-", lw=1.5, color=colors[name], label=name)
    ax.plot(results["老师"]["targets"], results["老师"]["xs"], "k--", lw=1.0, label="目标位置")
    ax.set_xlabel("y（横向，米，+左 / -右）")
    ax.set_ylabel("x（前进，米）")
    ax.set_title("老师 vs 学生（只看图像）")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig("m5_trajectory.png", dpi=120)
    print("\n已保存 m5_trajectory.png")


if __name__ == "__main__":
    main()
