"""模块 4：相机（Camera）

把状态 (x, y, theta) + 世界 渲染成一张图像 (H, W, 3)，取值 [0, 1]。

接口对齐真实 OpenBot（android/.../assets/config.json）：
    AUTOPILOT 模型 inputSize = "256x96"  ->  高 96，宽 256
    Autopilot.java 里像素除以 255 归一化 -> [0, 1]
    形状 [1, H, W, 3]，通道在最后（RGB）

原理（前视针孔相机 + 地平面求交）：
  1. 相机装在车上，离地高度 h，向下俯仰 phi。
  2. 图像每个像素 (u, v) 对应一条射线
         d = d0 + x_c * r0 + y_c * u0
     d0=光轴, r0=图像"右", u0=图像"上"。
  3. 射线与地面 z=0 求交：t = -h / d_z，得到地面落点 (Px, Py)。
  4. 用 world.lateral_offset 判断落点是否在路上，据此上色。

注意：几何由 world 提供，相机只负责"外观"（颜色、光照）。
      所以世界换成弯道时，相机代码不用改。
"""

import numpy as np


# 外观参数（几何参数在 World 里，这里只放"长什么样"）
PALETTE = {
    "sky": (0.55, 0.72, 0.90),
    "grass": (0.35, 0.55, 0.28),
    "road": (0.30, 0.30, 0.33),
    "lane": (0.92, 0.92, 0.88),
}


class Camera:
    def __init__(
        self,
        image_size=(96, 256),   # (高, 宽)
        fov_deg=70.0,           # 水平视场角
        height=0.12,            # 相机离地高度（米）
        pitch_deg=12.0,         # 相机向下俯仰
        forward_offset=0.02,    # 相机相对车心的前移，避免看到自身
        lane_width=0.04,        # 车道线宽度（米）
        fog_distance=12.0,      # 多远开始雾化
    ):
        self.H, self.W = image_size
        self.fov = np.deg2rad(fov_deg)
        self.height = height
        self.pitch = np.deg2rad(pitch_deg)
        self.forward_offset = forward_offset
        self.lane_width = lane_width
        self.fog_distance = fog_distance
        self._precompute()

    def _precompute(self):
        """预计算每个像素在相机坐标系下的 (x_c, y_c)，每帧只需做旋转。"""
        W, H = self.W, self.H
        f = (W / 2.0) / np.tan(self.fov / 2.0)     # 焦距（像素）
        us = np.arange(W)
        vs = np.arange(H)
        self.xc = (us - W / 2.0) / f               # (W,)
        self.yc = (H / 2.0 - vs) / f               # (H,)  行号向下增大

    def render(self, state, world):
        """渲染一帧。返回 float32 图像 (H, W, 3)，取值 [0, 1]。"""
        H, W = self.H, self.W
        x, y, theta = state.pose()

        # --- 车体基向量 ---
        F = np.array([np.cos(theta), np.sin(theta), 0.0])   # 前
        U = np.array([0.0, 0.0, 1.0])                        # 上
        R = np.array([np.sin(theta), -np.cos(theta), 0.0])   # 右

        c, s = np.cos(self.pitch), np.sin(self.pitch)
        d0 = c * F - s * U      # 光轴：朝前并向下
        u0 = s * F + c * U      # 图像"上"
        r0 = R                  # 图像"右"

        # --- 每个像素的射线方向 (H, W, 3) ---
        dirs = (
            d0[None, None, :]
            + self.xc[None, :, None] * r0[None, None, :]
            + self.yc[:, None, None] * u0[None, None, :]
        )
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # --- 相机位置，与地面求交 ---
        Cx = x + self.forward_offset * np.cos(theta)
        Cy = y + self.forward_offset * np.sin(theta)
        h = self.height

        dz = dirs[..., 2]
        ground = dz < -1e-6                      # 只有向下的射线能打到地面
        # 非地面像素的 t 设为 0（而不是 inf），避免 inf*0 = nan 的告警
        t = np.where(ground, -h / np.where(ground, dz, -1.0), 0.0)
        Px = Cx + t * dirs[..., 0]
        Py = Cy + t * dirs[..., 1]

        # --- 上色 ---
        img = np.empty((H, W, 3), dtype=np.float32)
        img[:] = PALETTE["sky"]
        img[ground] = PALETTE["grass"]

        # 落点相对道路中心的横向偏移（用 world 判断，保证与地图一致）
        lateral = Py - world.centerline_y(Px)
        half = world.road_half_width

        on_road = ground & (np.abs(lateral) <= half)
        img[on_road] = PALETTE["road"]

        # 两侧实线
        lane = ground & (np.abs(np.abs(lateral) - half) < self.lane_width)
        img[lane] = PALETTE["lane"]

        # 中心虚线（每 0.5 m 一段，1 m 周期）
        Px_safe = np.where(ground, Px, 0.0)
        dash = np.floor(Px_safe / 0.5).astype(np.int64) % 2 == 0
        center = ground & (np.abs(lateral) < self.lane_width) & dash
        img[center] = PALETTE["lane"]

        # 远处雾化：让网络能从画面里感知远近
        fog = np.clip(t / self.fog_distance, 0.0, 1.0)
        fog = np.where(ground, fog, 0.0)[..., None]
        sky = np.array(PALETTE["sky"], dtype=np.float32)
        img = img * (1.0 - 0.55 * fog) + sky * (0.55 * fog)

        return np.clip(img, 0.0, 1.0).astype(np.float32)
