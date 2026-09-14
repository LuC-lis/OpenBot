"""M4 组合演示：引入"老师策略"。

对比三位司机：
    随机策略  —— 乱动
    直线策略  —— 死板往前
    老师策略  —— 用上帝视角，看路修正方向，并按 cmd 移动到目标位置

运行：
    cd OpenBot/sim && python3 m4_demo.py
输出：
    m4_trajectory.png
"""

import numpy as np

from env import OpenBotEnv
from policies import RandomPolicy, TeacherPolicy


class StraightPolicy:
    """永远两轮同速的直线策略。"""

    def act(self, obs, info):
        return np.array([0.6, 0.6], dtype=np.float32)


def run_episode(env, policy, seed=0, max_steps=600):
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
        "xs": xs,
        "ys": ys,
        "targets": targets,
        "rewards": rewards,
        "terminated": terminated,
        "reason": info["components"]["reason"],
    }


def main():
    env = OpenBotEnv(seed=0)

    policies = {
        "随机": RandomPolicy(env.rng),
        "直线": StraightPolicy(),
        "老师": TeacherPolicy(),
    }

    results = {}
    print(f"{'策略':<6}{'步数':>5}{'回报':>10}{'平均|横向|':>12}{'平均跟踪误差':>14}  结束原因")
    for name, policy in policies.items():
        r = run_episode(env, policy, seed=0)
        results[name] = r
        ys = np.array(r["ys"])
        targets = np.array(r["targets"])
        track_err = np.mean(np.abs(ys - targets))
        reason = r["reason"] if r["terminated"] else "timeout"
        print(
            f"{name:<6}{len(ys):>5}{sum(r['rewards']):>10.1f}"
            f"{np.mean(np.abs(ys)):>12.3f}{track_err:>14.3f}  {reason}"
        )

    # 画轨迹：老师应该贴着虚线（目标位置）走
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    half = env.world.road_half_width
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.axvspan(-half, half, color="0.88", label="道路")
    for name, r in results.items():
        ax.plot(r["ys"], r["xs"], "-", lw=1.4, label=name)
    ax.plot(results["老师"]["targets"], results["老师"]["xs"], "k--", lw=1.0, label="老师的目标")
    ax.set_xlabel("y（横向，米，+左 / -右）")
    ax.set_ylabel("x（前进，米）")
    ax.set_title("三位司机的轨迹（俯视）")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig("m4_trajectory.png", dpi=120)
    print("已保存 m4_trajectory.png")


if __name__ == "__main__":
    main()
