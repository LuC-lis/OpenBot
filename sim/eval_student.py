"""评估学生策略的"纠错鲁棒性"。

和 m5_demo 不同：这里跑多个 episode，且故意用【困难起点】
（车停在偏边、车头大角度偏），甚至中途施加扰动，看策略能不能救回来。

指标：
    success     —— 跑满 max_steps 的比例（没翻车）
    steps       —— 平均存活步数
    return      —— 平均累计回报
    track_err   —— 平均跟踪误差 |lateral - target|

用法：
    cd OpenBot/sim && ./gpu_python.sh eval_student.py --model ../policy/models/cil_d01.tflite
"""

import argparse

import numpy as np

from env import OpenBotEnv
from policies import TFLitePolicy


def evaluate(model_path, episodes=20, max_steps=400, hard=True, disturb=0.0, seed0=0):
    env = OpenBotEnv(seed=0)
    policy = TFLitePolicy(model_path)

    steps_list, ret_list, track_list, reasons = [], [], [], []
    for i in range(episodes):
        rng = np.random.default_rng(seed0 + i)
        obs, info = env.reset(seed=seed0 + i)
        if hard:
            y = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 0.85) * env.world.road_half_width
            th = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.5)
            env.state.reset(0.0, float(y), float(th))
            obs, info = env.observe()

        total_r, track = 0.0, []
        t = 0
        for t in range(max_steps):
            action = policy.act(obs, info)
            obs, rew, term, trunc, info = env.step(action)
            total_r += rew
            track.append(abs(info["lateral"] - info["target_lateral"]))
            if disturb > 0 and env.rng.random() < disturb and not (term or trunc):
                env.perturb()
                obs, info = env.observe()
            if term or trunc:
                break

        steps_list.append(t + 1)
        ret_list.append(total_r)
        track_list.append(float(np.mean(track)))
        reasons.append(info["components"]["reason"] if term else "timeout")

    n = len(steps_list)
    success = sum(1 for r in reasons if r == "timeout") / n
    return {
        "success": success,
        "steps": float(np.mean(steps_list)),
        "return": float(np.mean(ret_list)),
        "track_err": float(np.mean(track_list)),
        "reasons": {r: reasons.count(r) for r in set(reasons)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hard", action="store_true", default=True)
    ap.add_argument("--easy", dest="hard", action="store_false")
    ap.add_argument("--disturb", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = evaluate(args.model, args.episodes, hard=args.hard, disturb=args.disturb, seed0=args.seed)
    tag = "困难起点" if args.hard else "普通起点"
    print(f"{args.model}")
    print(f"  [{tag}, 扰动={args.disturb}] 成功率 {r['success']*100:5.1f}%  "
          f"平均步数 {r['steps']:6.1f}  回报 {r['return']:7.1f}  跟踪误差 {r['track_err']:.3f}")
    print(f"  结束原因: {r['reasons']}")


if __name__ == "__main__":
    main()
