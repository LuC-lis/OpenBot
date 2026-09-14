"""迷你神经网络（纯 numpy，从零实现）

目的：不借助任何深度学习框架，手写一个神经网络，看清 NN 到底是什么。

它要做的事和 OpenBot 一模一样 —— **模仿老师**：
    输入：车当前的状态（横向偏移 lateral、航向 heading）和指令 cmd
    输出：左右轮动作 [left, right]
    答案：老师 TeacherPolicy 给的 [left, right]

也就是说，这是一个"从零写的、微缩版的行为克隆"。

只依赖 numpy 和我们的 policies.py。
运行：
    cd OpenBot/sim && python3 tiny_nn.py
"""

import numpy as np

from policies import TeacherPolicy

# ======================================================================
# 第 1 部分：网络本体
# ======================================================================
class TinyMLP:
    """一个极简的多层感知机（MLP）。

    结构：若干个"全连接层 + tanh 激活"，最后一层线性输出。
    sizes = [3, 16, 16, 2] 表示 输入3维 -> 隐藏16 -> 隐藏16 -> 输出2维
    """

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            # 初始化：小随机数。太大或太小都学不好。
            self.W.append(rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    # ------------------------------------------------------------------
    def forward(self, X):
        """前向传播：把输入一路算到输出。

        同时把中间结果存起来（self.a / self.z），反向传播要用。
        """
        self.a = [X]          # a[0] = 输入
        self.z = []           # 每层的"加权和"（激活之前）
        A = X
        # 隐藏层：线性 + tanh
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            self.z.append(Z)
            A = np.tanh(Z)
            self.a.append(A)
        # 输出层：线性（回归任务，不加激活）
        Z = A @ self.W[-1] + self.b[-1]
        self.z.append(Z)
        self.a.append(Z)
        return Z

    # ------------------------------------------------------------------
    def train_step(self, X, Y, lr):
        """一个批次的训练：前向 -> 算误差 -> 反向求梯度 -> 更新权重。

        这就是神经网络的全部核心。详见下面每行注释。
        """
        Yhat = self.forward(X)
        N, K = X.shape[0], Y.shape[1]

        # --- 损失：均方误差 MSE = mean((预测 - 答案)^2) ---
        loss = np.mean((Yhat - Y) ** 2)

        # --- 反向传播 ---
        # 损失对输出层的导数：dL/dZ_out = 2*(Yhat - Y) / (N*K)
        dZ = 2.0 * (Yhat - Y) / (N * K)

        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # 输出层梯度
        dW[-1] = self.a[-2].T @ dZ
        db[-1] = dZ.sum(axis=0)
        # 误差往回传，得到对上一层激活的导数
        dA = dZ @ self.W[-1].T

        # 逐层往回：隐藏层
        for i in range(len(self.W) - 2, -1, -1):
            # tanh 的导数 = 1 - tanh(z)^2，而 tanh(z) 正是 self.a[i+1]
            dZ = dA * (1.0 - self.a[i + 1] ** 2)
            dW[i] = self.a[i].T @ dZ
            db[i] = dZ.sum(axis=0)
            dA = dZ @ self.W[i].T

        # --- 梯度下降：往"误差变小"的方向挪一小步 ---
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

        return loss

    # ------------------------------------------------------------------
    def train(self, X, Y, epochs=3000, lr=0.05, batch_size=32, seed=0, log_every=300):
        rng = np.random.default_rng(seed)
        N = X.shape[0]
        for ep in range(1, epochs + 1):
            idx = rng.permutation(N)                  # 每轮打乱数据
            for s in range(0, N, batch_size):
                b = idx[s : s + batch_size]           # 取一个小批次
                self.train_step(X[b], Y[b], lr)
            if ep == 1 or ep % log_every == 0:
                full_loss = np.mean((self.forward(X) - Y) ** 2)
                print(f"  epoch {ep:5d}   训练集 loss = {full_loss:.5f}")
        return self

    def predict(self, X):
        return self.forward(X)


# ======================================================================
# 第 2 部分：造数据 —— 让老师当标准答案
# ======================================================================
def make_dataset(n=4000, seed=0):
    """随机采样各种车辆状态，问老师要动作，做成数据集。"""
    rng = np.random.default_rng(seed)
    teacher = TeacherPolicy()

    lateral = rng.uniform(-0.9, 0.9, size=n)      # 横向偏移
    heading = rng.uniform(-0.3, 0.3, size=n)      # 航向
    cmd = rng.choice([-1.0, 0.0, 1.0], size=n)    # 指令

    X = np.stack([lateral, heading, cmd], axis=1)  # 输入特征
    Y = np.zeros((n, 2), dtype=np.float64)         # 老师答案 [left, right]
    for i in range(n):
        info = {
            "lateral": lateral[i],
            "heading": heading[i],
            "target_lateral": -cmd[i] * 0.4,       # 与 task.target_lateral 一致
        }
        Y[i] = teacher.act(None, info)
    return X, Y


# ======================================================================
# 第 3 部分：训练 + 验证
# ======================================================================
def main():
    print("1) 造数据：采样车辆状态，问老师要'标准答案'...")
    X, Y = make_dataset(n=4000, seed=0)

    # 划分训练集 / 验证集
    n_train = 3200
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xva, Yva = X[n_train:], Y[n_train:]
    print(f"   训练样本 {len(Xtr)}，验证样本 {len(Xva)}，输入维度 {X.shape[1]}，输出维度 {Y.shape[1]}")

    print("\n2) 建网络：[3, 16, 16, 2]  即 3维输入 -> 两个16维隐藏层 -> 2维输出")
    net = TinyMLP([3, 16, 16, 2], seed=0)

    print("\n3) 训练（看着 loss 往下掉）...")
    net.train(Xtr, Ytr, epochs=3000, lr=0.05, batch_size=32, seed=0, log_every=300)

    # --- 验证：在没见过的数据上，网络预测得有多准 ---
    pred = net.predict(Xva)
    mae = np.mean(np.abs(pred - Yva))
    # 对照基线：如果网络什么都不会，只会猜平均值
    baseline = np.mean(np.abs(np.mean(Ytr, axis=0) - Yva))
    print(f"\n4) 验证结果：")
    print(f"   网络的平均绝对误差   = {mae:.4f}")
    print(f"   '只会猜平均'的基线误差 = {baseline:.4f}")
    print(f"   -> 网络比基线好了 {baseline / mae:.1f} 倍")

    print("\n5) 抽几个样本，对比 网络 vs 老师：")
    print(f"   {'lateral':>8}{'heading':>9}{'cmd':>5} | {'网络[L,R]':>18} | {'老师[L,R]':>18}")
    for i in range(5):
        p = pred[i]
        t = Yva[i]
        print(
            f"   {Xva[i,0]:>8.2f}{Xva[i,1]:>9.2f}{Xva[i,2]:>5.0f} | "
            f"[{p[0]:>7.2f},{p[1]:>7.2f}] | [{t[0]:>7.2f},{t[1]:>7.2f}]"
        )

    print("\n结论：网络只是靠'前向传播 + 反向传播 + 梯度下降'，")
    print("      就学会了模仿老师 —— 这正是 OpenBot 行为克隆在做的事。")
    print("      区别只是：OpenBot 的输入是图像(CNN)，这里压缩成了 3 个数字。")


if __name__ == "__main__":
    main()
