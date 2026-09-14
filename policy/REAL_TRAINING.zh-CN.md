# OpenBot 真实数据训练与部署指南（中文）

> 这份文档讲的是 **OpenBot 的"真车路线"**：用真车采集真实数据 → 在电脑上训练 → 把模型装回真车。
>
> 和 `sim/README.zh-CN.md`（仿真学习笔记）不同——那份是"在仿真里学原理"，这份是"**真的让车自己开**"。

---

## 0. 两条路线对照

| | 仿真路线（我们的笔记）| **真实路线（本文档）** |
|---|---|---|
| 数据来源 | 仿真老师自动生成 | **人开真车录制** |
| 图像 | 合成画面 | 真实手机摄像头 |
| 能否上真车 | ❌（域差距太大）| ✅ **可以** |
| 用途 | 学原理、调流程 | 真正部署 |

**为什么必须用真实数据？** 因为模型最终要面对真实世界的图像。训练和推理必须"看到同一个世界"，否则就会出现巨大的域差距（domain gap）。

---

## 1. 硬件准备

| 部件 | 要求 |
|------|------|
| 机器人本体 | OpenBot（DIY / Lite / RTR / RC Truck 均可），见 `body/` |
| 固件 | 已刷好 Arduino/ESP32 固件，见 `firmware/README.zh-CN.md` |
| 手机 | Android，已安装 OpenBot 机器人 App（`android/`）|
| 游戏手柄 | 蓝牙手柄（PS4 / Xbox / 第三方），用于驾驶与录制开关 |
| 连接 | 手机 ↔ 车身：USB 或蓝牙（BLE）|

---

## 2. 采集真实数据

### 2.1 连接与设置

1. 打开机器人 App，进入设置，确认与车身连接正常（USB 图标应为已连接）。
2. 用蓝牙配对手柄：
   - **PS4**：按住 `PS` + `Share` 直到指示灯快闪，进入配对。
   - **Xbox**：按配对键。
3. 在主菜单选择模型 **`CIL-Mobile-Cmd`**（这是指令条件化的行为克隆模型）。

### 2.2 录制

1. 用**手柄**驾驶机器人，走你想让它学会的路线。
2. **录制开关**：
   - PS4：按 **X** 按钮切换记录开/关。
   - 其他手柄：在 App 的日志界面可看到录制状态。
3. **`cmd`（转向指示灯）**：录制过程中用界面上的左/右转向按钮产生 `cmd` 信号（对应 `indicatorLog.txt`）。
   > 注意：`cmd` 是**转向指令**（-1/0/+1），不是速度。网络会学会"在什么指令下该怎么转"。

### 2.3 数据落在哪

录制文件保存在手机内部存储：

```
Documents/OpenBot/yyyymmdd_hhmmss.zip
```

每个 zip 是一次录制（一个会话），里面包含：

```
images/            # 每帧图像
sensor_data/
  rgbFrames.txt    # 帧-时间戳
  ctrlLog.txt      # 左/右电机指令
  indicatorLog.txt # cmd 指令
  ...              # 还有电压/声纳/轮速等日志
```

### 2.4 录多少、怎么录

| 建议 | 说明 |
|------|------|
| **时长** | 论文实验约 30 分钟。起步可以先录 10~15 分钟。 |
| **一致性** | **网络会模仿你**。你开得越稳、越一致，它学得越好。 |
| **多样性** | 覆盖不同光照、不同路线、不同指令。 |
| **多个会话** | 分多次录制，每次一个 zip，便于按 80/20 划分。 |
| **避免** | 大量急打方向、频繁停车（会被 `remove_zeros` 过滤掉静止帧）。|

---

## 3. 把数据放到电脑并整理

### 3.1 拷贝与解压

把手机里的 zip 拷到电脑，解压。每个 zip 解压后得到一个会话目录（含 `images/` 与 `sensor_data/`）。

### 3.2 目录结构（**关键**）

OpenBot 训练代码读取的目录是**两级**结构：

```
policy/dataset/
├── train_data/            # 训练集
│   └── <分组名>/           # 例如 my_dataset_1（可任意命名）
│       ├── <会话1>/        # 例如 session_1
│       │   ├── images/
│       │   └── sensor_data/
│       └── <会话2>/
│           ├── images/
│           └── sensor_data/
└── test_data/             # 测试集（结构同上）
    └── <分组名>/
        └── <会话>/
            ├── images/
            └── sensor_data/
```

> **为什么是两级？** 代码先列出 `train_data` 下的一级子目录（"分组"），
> 再列出每个分组下的二级子目录（"会话"），会话目录里才找 `sensor_data/`。
> 我们仿真的数据就是按 `train_data/sim/session_000/...` 组织的，已验证可用。

### 3.3 划分训练/测试

**80% 训练、20% 测试**（按**会话**划分，不要按帧划分）。

### 3.4 自查：结构对不对

运行训练时会打印 `Processing folder ...`。如果看到大量

```
Skipping <某目录>
```

说明**层级不对**——多半是缺少那一层"分组"，或者会话目录里没有 `sensor_data/`。

### 3.5 更省事的办法：WebApp 上传（推荐）

OpenBot 自带一个 Web 界面，可自动上传、解压、整理数据：

```bash
conda activate openbot
python -m openbot.server
```

浏览器打开：`http://localhost:8000/#/uploaded`

- 手机和电脑连**同一个 WiFi**；
- 录制文件会自动上传并解压到 `policy/dataset/uploaded/`；
- 然后在网页上把会话**移动到数据集**（train/test）。

> 若上传失败：重启服务器和 App；确认同一 WiFi；关闭同名 5GHz 网络；
> 在 Android Studio 的 Logcat 里过滤 `NSD` / `Upload` 看调试信息。

---

## 4. 训练

### 4.1 环境

用仓库自带的环境文件（CPU/GPU 均可）：

```bash
cd OpenBot/policy
conda env create -f environment_linux.yml     # 或 _mac.yml / _win.yml
conda activate openbot
```

> Windows 还需安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。
> Linux 若要 GPU，确保已装好 NVIDIA 驱动。

### 4.2 训练命令

**最简单（直接从目录读数据）**：

```bash
cd OpenBot/policy
python -m openbot.train --no_tf_record \
    --model cil_mobile \
    --batch_size 64 \
    --num_epochs 50 \
    --batch_norm
```

**推荐（先转 tfrecord，内存更省、更快）**：

```bash
python -m openbot.train --create_tf_record \
    --model cil_mobile --batch_size 64 --num_epochs 50 --batch_norm
```

> `--create_tf_record` 只需在**数据变化后**跑一次；之后去掉它会自动走 tfrecord 路径。
> 重新采集数据后，要删掉 `dataset/tfrecords/` 或重新加 `--create_tf_record`。

### 4.3 常用超参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `--model` | 网络架构 | `cil_mobile`（默认，手机可跑）/ `pilot_net`（更大）|
| `--batch_size` | 批大小 | 32~128（越大越吃内存）|
| `--num_epochs` | 训练轮数 | 30~100 |
| `--learning_rate` | 学习率 | 默认 0.0003；慢就调到 0.001~0.002 |
| `--batch_norm` | 用批归一化 | 建议开 |
| `--flip_aug` | 左右翻转增强 | 路线对称时建议开 |
| `--cmd_aug` | 指令增广 | 可选 |
| `--resume` | 从上次检查点继续 | 中断后用 |

### 4.4 输出

模型保存在 `policy/models/<自动生成的名字>/checkpoints/`：

```
best.tflite         ← 验证集最优（一般用这个）
best-train.tflite   ← 训练集最优
best-val.tflite
last.tflite         ← 最后一轮
cp-*.ckpt/          ← 检查点（可 --resume）
logs/               ← log.csv、loss.png、error.png、test_preview.png
```

**训练时盯这几个指标**：

| 指标 | 含义 |
|------|------|
| `loss` / `val_loss` | 应持续下降，`val_loss` 不再降就说明过拟合 |
| `mean_absolute_error` | 预测动作与你的动作的平均偏差 |
| `angle_metric` | 转向是否在 0.1 以内（越接近 1 越好）|
| `direction_metric` | 转向方向是否正确（越接近 1 越好）|

> ⚠️ **注意**：`angle_metric` 会被大量"直行"样本稀释。**务必单独检查"转弯样本"**，
> 否则会像我们仿真里那样——指标好看，实际不会转弯。

### 4.5 GPU 加速（可选）

如果电脑有 NVIDIA 显卡：

- `environment_linux.yml` 里含 `cudatoolkit=11.2 + cudnn=8.1 + tensorflow 2.9.3`；
- **注意**：CUDA 11.2 不支持 Ada 架构（RTX 40 系）。40 系显卡需改用 **TF ≥ 2.15 + CUDA ≥ 11.8**（详见 `sim/README.zh-CN.md` 第 7 节我们踩过的全部坑）。

---

## 5. 部署到真车

### 5.1 方法 A：替换 App 内置模型并重新编译

```bash
# 1) 选一个 tflite，改名
cp policy/models/<模型名>/checkpoints/best.tflite \
   android/robot/src/main/assets/networks/autopilot_float.tflite

# 2) 确认 config.json 中的 inputSize 与模型一致（CIL-Mobile-Cmd 为 "256x96"）
#    android/robot/src/main/assets/config.json

# 3) 用 Android Studio 打开 android/ 目录，编译安装
```

> 注意：`assets/networks/` 里的官方模型是**构建时从云端下载**的（见 `android/robot/download.gradle`）。
> 如果你替换了它，构建时可能被重新下载覆盖——可临时注释掉 `preBuild.dependsOn downloadNetworks`
> 或改用下面方法 B。

### 5.2 方法 B：用 App 的模型管理加载文件（不改代码，推荐）

1. 把 `.tflite` 传到手机存储；
2. 在 App 的「模型管理」里，把该模型的 `pathType` 设为 `FILE`、`path` 指向手机上的文件；
3. 选择该模型即可使用。

> `config.json` 里模型有三种来源：`ASSET`（打包进 App）、`URL`（运行时下载）、`FILE`（手机本地文件）。

### 5.3 首次测试（务必）

1. **拆掉轮胎**，确认所有功能正常；
2. 连接游戏手柄，熟悉**急停按键**；
3. 在空旷无障碍的场地；
4. 先用**手动模式**跑一跑，再切 **Autopilot**。

---

## 6. 提升效果的建议

| 问题 | 对策 |
|------|------|
| 转向不够、总是直行 | 检查训练数据里转弯样本是否太少；用 `--flip_aug`；降低 dropout（见下）|
| 一上真车就跑偏 | 域差距：增加数据多样性（光照、路线）；确保相机安装一致 |
| 指标好看但不会转弯 | 单独统计转弯样本的误差；别只看平均值 |
| **模型完全忽略图像、只看指令** | 参见 `sim/README.zh-CN.md` 第 8 节——我们定位到 **dropout=0.5 会导致捷径学习**；可尝试把模型里的 dropout 调小 |
| 过拟合（val_loss 上升）| 加数据、加 `--flip_aug`、减小模型（`cil_mobile` 而非 `pilot_net`）|

---

## 7. 安全注意事项

OpenBot 官方[免责声明](DISCLAIMER.zh-CN.md)强调：

- ❗ **碰撞可能损坏手机**；
- ❗ 自动驾驶策略**不完美**，可能撞车；
- ❗ **始终连接手柄**，随时接管；
- ❗ 在安全、空旷环境操作；
- ❗ 测试新模型时先**拆轮胎**；
- ❗ 风险自负。

---

## 8. 与仿真笔记的对照

| 概念 | 仿真（`sim/README.zh-CN.md`）| 真实（本文档）|
|------|------------------------------|--------------|
| 老师 | `TeacherPolicy`（上帝视角程序）| **人**（手柄驾驶）|
| 数据 | `export_data.py` 自动生成 | App 录制 zip |
| 图像 | 合成渲染 | 真手机摄像头 |
| 训练 | `train_plain.py` / `train.py` | `python -m openbot.train` |
| 部署 | 回灌仿真验证 | 装进 App 上真车 |
| 核心难点 | 捷径学习 / dropout | **域差距** |

**两者互补**：仿真帮你**低成本理解原理、快速调流程、定位 bug**；真实数据才是**能上车的唯一途径**。

---

## 附：一页速查

```bash
# 采集：手机 App + 手柄，Document/OpenBot/*.zip

# 整理：policy/dataset/{train_data,test_data}/<分组>/<会话>/{images,sensor_data}/
#       或：python -m openbot.server  → http://localhost:8000/#/uploaded

# 训练：
cd OpenBot/policy
conda activate openbot
python -m openbot.train --create_tf_record --model cil_mobile \
    --batch_size 64 --num_epochs 50 --batch_norm

# 产物：policy/models/<名字>/checkpoints/best.tflite

# 部署：拷到 android/robot/src/main/assets/networks/autopilot_float.tflite 后重新编译
#       或传给手机，用 App 的模型管理（pathType=FILE）加载

# 安全：拆轮胎测试 → 连手柄 → 空旷场地 → 随时接管
```
