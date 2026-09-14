"""模块 5：任务 / 奖励（Task）

定义两件事：
    1. "开得好不好"   -> reward
    2. "什么时候结束" -> terminated

它依赖 world（要知道路在哪），但不依赖相机，也不依赖闭环。

任务：指令条件下的车道保持。
    cmd ∈ {-1, 0, +1} 表示 左 / 直行 / 右 指令。
    我们把 cmd 解释为"目标横向位置"：
        cmd =  0  -> 目标在道路正中央
        cmd = +1  -> 目标在道路右侧（lateral = -lateral_target）
        cmd = -1  -> 目标在道路左侧（lateral = +lateral_target）

    车要：
      (1) 横向位置尽量靠近目标；
      (2) 车头尽量朝前（theta -> 0）；
      (3) 保持前进；
      (4) 动作别猛打猛收。

奖励 = 目标横向 + 车头朝前 + 前进 - 动作抖动；冲出道路则终止并重罚。
"""

import math


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LaneKeepingTask:
    def __init__(
        self,
        world,
        lateral_target=0.6,       # cmd=±1 时，目标横向偏移的大小（米）
        off_road_margin=0.15,     # 冲出道路边线多少米算失败
        w_lane=1.0,
        w_head=0.3,
        w_progress=0.2,
        w_smooth=0.02,
        terminate_penalty=5.0,
    ):
        self.world = world
        self.lateral_target = lateral_target
        self.off_road_margin = off_road_margin
        self.w_lane = w_lane
        self.w_head = w_head
        self.w_progress = w_progress
        self.w_smooth = w_smooth
        self.terminate_penalty = terminate_penalty

    def target_lateral(self, cmd):
        """cmd 对应的目标横向位置。

        约定：+y 为左，-y 为右。
            cmd=+1（右） -> 目标横向为负
            cmd=-1（左） -> 目标横向为正
        """
        return -float(cmd) * self.lateral_target

    def evaluate(self, state, action, prev_action, v_norm, cmd):
        """计算一步的奖励与是否终止。

        参数：
            state:       State
            action:      (2,) 本步动作（已夹到 [-1,1]）
            prev_action: (2,) 上一步动作
            v_norm:      车体线速度 / 最大速度，范围约 [-1, 1]
            cmd:         本步指令
        返回：
            (reward, terminated, components)
        """
        lateral = self.world.lateral_offset(state.x, state.y)
        half = self.world.road_half_width
        target = self.target_lateral(cmd)

        r_lane = -self.w_lane * abs(lateral - target) / half
        r_head = -self.w_head * abs(_wrap(state.theta))
        r_progress = self.w_progress * v_norm
        r_smooth = -self.w_smooth * float(((action - prev_action) ** 2).sum())

        reward = r_lane + r_head + r_progress + r_smooth

        off_road = abs(lateral) > half + self.off_road_margin
        turned = abs(_wrap(state.theta)) > math.pi / 2.0
        terminated = off_road or turned
        if terminated:
            reward -= self.terminate_penalty

        components = {
            "lane": r_lane,
            "head": r_head,
            "progress": r_progress,
            "smooth": r_smooth,
            "target_lateral": target,
            "reason": "off_road" if off_road else ("turned" if turned else None),
        }
        return reward, terminated, components
