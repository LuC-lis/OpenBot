"""M3 组合演示：完整闭环 env.reset() / env.step()。

跑两个 episode：
  1. 随机策略   -> 很快就会冲出道路
  2. 直线策略   -> 一直往前，能坚持更久
并把轨迹画出来。

运行：
    cd OpenBot/sim && python3 m3_demo.py
输出：
    m3_trajectory.png
"""

import numpy as np

from env import OpenBotEnv


def run_episode(env, policy, seed=0, max_steps=600):
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), "观测不符合契约！"
    assert obs["image"].shape == (env.H, env.W, 3)
    assert obs["image"].dtype == np.float32

    xs, ys, rewards = [], [], []
    for _ in range(max_steps):
        action = policy(obs, info)
        assert env.action_space.contains(action), "动作不符合契约！"

        obs, reward, terminated, truncated, info = env.step(action)
        xs.append(info["state"][0])
        ys.append(info["state"][1])
        rewards.append(reward)

        if terminated or truncated:
            return xs, ys, rewards, terminated, info

    return xs, ys, rewards, False, info


def main():
    env = OpenBotEnv(seed=0)

    policies = {
        "随机策略": lambda obs, info: env.rng.uniform(-1.0, 1.0, size=2).astype(np.float32),
        "直线策略": lambda obs, info: np.array([0.6, 0.6], dtype=np.float32),
    }

    results = {}
    for name, policy in policies.items():
        xs, ys, rewards, terminated, info = run_episode(env, policy, seed=0)
        results[name] = (xs, ys)
        reason = info["components"]["reason"] if terminated else "timeout"
        print(
            f"{name:6s}: 步数={len(xs):3d}  回报={sum(rewards):8.2f}  "
            f"结束原因={reason:9s}  末横向偏移={ys[-1]:+.2f}"
        )

    # 画轨迹
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    half = env.world.road_half_width
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.axvspan(-half, half, color="0.85", label="道路")
    for name, (xs, ys) in results.items():
        ax.plot(ys, xs, "-o", ms=2, lw=1.2, label=name)
    ax.set_xlabel("y（横向，米，+左 / -右）")
    ax.set_ylabel("x（前进，米）")
    ax.set_title("两个策略的轨迹（俯视）")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig("m3_trajectory.png", dpi=120)
    print("已保存 m3_trajectory.png")


if __name__ == "__main__":
    main()
