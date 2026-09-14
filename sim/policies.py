"""M4：策略（Policies）

一个"策略"就是一个函数：给它观测（和 info），它返回动作 [left, right]。
本文件放两个手写策略：

    RandomPolicy  —— 随机乱动，用作 baseline
    TeacherPolicy —— "老师"：用上帝视角的状态精确控制，开得很好

TeacherPolicy 是行为克隆里"标准答案"的来源：
它在仿真里能读到精确的 (x, y, theta)，因此能算出完美的方向盘。
学生（NN）将来只能看画面，却要学会模仿老师。
"""

import math

import numpy as np

from dynamics import MAX_WHEEL_SPEED, WHEEL_BASE


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RandomPolicy:
    """每次随机给一对 [-1, 1] 的动作。"""

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, info):
        return self.rng.uniform(-1.0, 1.0, size=2).astype("float32")


class TeacherPolicy:
    """"老师"：用特权状态做串级比例控制。

    思路（两层）：
        1) 横向控制：根据"与目标横向位置的差"决定应该朝哪个方向（期望航向 theta_des）。
        2) 航向控制：根据"与期望航向的差"决定角速度 omega。
        3) 再把 (v_des, omega) 反解成左右轮指令。

    只用 info 里的特权信息（lateral / heading / target_lateral），
    这正是学生看不到、而老师可以"作弊"的部分。
    """

    def __init__(
        self,
        k_lat=1.5,            # 横向误差 -> 期望航向（rad per m）
        theta_max=0.5,        # 期望航向限幅（rad）
        k_head=4.0,           # 航向误差 -> 角速度
        forward_speed=0.6,    # 前进速度比例 [0,1]
        wheel_base=WHEEL_BASE,
        max_speed=MAX_WHEEL_SPEED,
    ):
        self.k_lat = k_lat
        self.theta_max = theta_max
        self.k_head = k_head
        self.forward_speed = forward_speed
        self.wheel_base = wheel_base
        self.max_speed = max_speed

    def act(self, obs, info):
        lateral = info["lateral"]
        theta = info["heading"]
        target_lat = info["target_lateral"]

        # 1) 横向误差 -> 期望航向
        e_lat = target_lat - lateral
        theta_des = max(-self.theta_max, min(self.theta_max, self.k_lat * e_lat))

        # 2) 航向误差 -> 角速度
        omega = self.k_head * _wrap(theta_des - theta)

        # 3) (v_des, omega) -> 左右轮指令
        v_des = self.forward_speed * self.max_speed
        v_l = v_des - 0.5 * omega * self.wheel_base
        v_r = v_des + 0.5 * omega * self.wheel_base

        u = [v_l / self.max_speed, v_r / self.max_speed]
        u = np.clip(np.array(u, dtype=np.float32), -1.0, 1.0)
        return u


class TFLitePolicy:
    """加载训练好的 OpenBot .tflite 模型，只看图像 + cmd 输出动作（M5）。

    模型输入（来自 policy/openbot/models.py 的 cil_mobile）：
        img: (1, 96, 256, 3) float32 in [0,1]
        cmd: (1, 1)          float32
    输出：(1, 2) = [left, right]
    """

    def __init__(self, model_path):
        import tensorflow as tf  # 训练环境里已有

        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.img_i = self.cmd_i = None
        for d in self.interp.get_input_details():
            if len(d["shape"]) == 4:
                self.img_i = d["index"]
            else:
                self.cmd_i = d["index"]
        self.out_i = self.interp.get_output_details()[0]["index"]

    def act(self, obs, info):
        img = np.asarray(obs["image"], dtype=np.float32)[None, ...]
        cmd = np.asarray(obs["cmd"], dtype=np.float32).reshape(1, 1)
        self.interp.set_tensor(self.img_i, img)
        self.interp.set_tensor(self.cmd_i, cmd)
        self.interp.invoke()
        return np.clip(self.interp.get_tensor(self.out_i)[0], -1.0, 1.0).astype(np.float32)
