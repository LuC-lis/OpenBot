"""模块 7：闭环（Env）—— 把前面所有模块串成一个可运行的仿真环境。

它把第 2 节那个循环真正实现出来：

    reset()  -> 随机放车、给定初始指令，返回第一条观测
    step(a)  -> 用动力学更新状态，算奖励，判断是否结束，返回新观测

依赖（全部来自前几步）：
    state / dynamics / world / camera / task / observation / spaces

对外接口（Gymnasium 风格）：
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
    obs = {"image": (H,W,3) float32 in [0,1], "cmd": (1,) float32}
    action = (2,) float32 in [-1, 1]
"""

import numpy as np

from camera import Camera
from dynamics import DT, MAX_WHEEL_SPEED, DiffDrive
from observation import make_observation
from spaces import Box, Dict
from state import State, wrap_angle
from task import LaneKeepingTask
from world import World


class OpenBotEnv:
    def __init__(
        self,
        image_size=(96, 256),
        dt=DT,
        max_steps=600,
        road_half_width=0.9,
        command_mode="turn",     # "turn": cmd∈{-1,0,1}；"speed": cmd∈[-1,1]
        seed=None,
    ):
        self.H, self.W = image_size

        # --- 组装各个模块 ---
        self.state = State()
        self.vehicle = DiffDrive(dt=dt)
        self.world = World(road_half_width=road_half_width)
        self.camera = Camera(image_size=image_size)
        self.task = LaneKeepingTask(self.world)

        self.command_mode = command_mode

        # --- 用"空间"把接口契约写成可检查的代码 ---
        self.action_space = Box(-1.0, 1.0, (2,))
        self.observation_space = Dict(
            {
                "image": Box(0.0, 1.0, (self.H, self.W, 3)),
                "cmd": Box(-1.0, 1.0, (1,)),
            }
        )

        self.rng = np.random.default_rng(seed)
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.cmd = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 随机起点：横向偏移在道路中间 50% 范围，航向小角度偏差
        half = self.world.road_half_width
        y0 = self.rng.uniform(-0.5 * half, 0.5 * half)
        theta0 = self.rng.uniform(-0.15, 0.15)
        self.state.reset(0.0, y0, theta0)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.cmd = self._sample_cmd()

        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)

        # 1) 动力学更新状态
        v, omega = self.vehicle.step(self.state, action[0], action[1])

        # 2) 指令变化更频繁（模拟驾驶者不断拨转向灯），让数据里出现更多转向
        if self.rng.random() < 0.10:
            self.cmd = self._sample_cmd()

        # 3) 任务：奖励 + 是否终止
        reward, terminated, components = self.task.evaluate(
            self.state, action, self.prev_action, v / MAX_WHEEL_SPEED, self.cmd
        )

        self.prev_action = action
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info(
            v, omega, components
        )

    # ------------------------------------------------------------------
    def _info(self, v=0.0, omega=0.0, components=None):
        """组装 info。reset 和 step 共用，保证第一步也能读到完整状态。"""
        return {
            "state": self.state.pose(),
            "lateral": self.world.lateral_offset(self.state.x, self.state.y),
            "heading": self.state.theta,
            "v": v,
            "omega": omega,
            "cmd": self.cmd,
            "target_lateral": self.task.target_lateral(self.cmd),
            "components": components,
        }

    # ------------------------------------------------------------------
    def perturb(self, lateral_std=0.25, heading_std=0.10):
        """人为扰动车辆状态（模拟被推了一下），逼老师从偏离状态纠错。

        这是让数据覆盖"偏离中心"状态的关键：老师开得太好，自然数据里
        几乎没有"偏了之后怎么救"的样本，模型就学不会纠错。
        """
        self.state.y += float(self.rng.normal(0.0, lateral_std))
        self.state.theta = wrap_angle(
            self.state.theta + float(self.rng.normal(0.0, heading_std))
        )
        limit = self.world.road_half_width * 0.9
        self.state.y = float(np.clip(self.state.y, -limit, limit))

    def observe(self):
        """返回当前状态的 (obs, info)（扰动后重新观测用）。"""
        return self._obs(), self._info()

    # ------------------------------------------------------------------
    def _sample_cmd(self):
        if self.command_mode == "speed":
            return float(
                self.rng.choice([1.0, 0.8, 0.6, 0.0, -0.6], p=[0.4, 0.2, 0.2, 0.1, 0.1])
            )
        return float(self.rng.choice([-1.0, 0.0, 1.0], p=[0.25, 0.5, 0.25]))

    def _obs(self):
        image = self.camera.render(self.state, self.world)
        return make_observation(image, self.cmd)

    def render(self):
        return (self._obs()["image"] * 255).astype(np.uint8)

    def close(self):
        pass
