# 复现操作日志

> 开始时间: 2026-06-03
> 环境: Linux 6.6.87.2-microsoft-standard-WSL2, Python 3.12.7, JAX 0.6.2 (CUDA)

---

## 初始环境状态

| 检查项 | 状态 |
|--------|------|
| Python 3.12.7 (uv venv) | ✅ |
| JAX 0.6.2 + CUDA | ✅ CudaDevice(id=0) |
| Flax 0.10.7 | ✅ |
| Optax 0.2.5 | ✅ |
| PyBaMM 25.6.0 | ✅ |
| 训练数据 (8.4G) | ✅ FNO(8) / FNO-Cape(8) / DeepONet(4) |
| models/ 目录 | ❌ 不存在，训练时自动创建 |
| logs/ 目录 | 🔨 新建 |
| REPRODUCTION_LOG.md | 🔨 新建（本操作日志） |

---

## Step 1-3: 启动训练 (2026-06-03)

### FNO 训练
- 命令: `.venv/bin/python train_model.py configs/training/FNO.yaml`
- 后台任务ID: `brtyr1hq5`
- 日志: `logs/train_FNO.log`
- 配置: Prada2013, CC+PLS+Triangle+GRF, 11000样本, 150 epochs, batch_size=20

### CAPE-FNO2 训练
- 命令: `.venv/bin/python train_model.py configs/training/CAPE_FNO2.yaml`
- 后台任务ID: `blsd5lwbj`
- 日志: `logs/train_CAPE_FNO2.log`
- 配置: Prada2013, CC+PLS+GRF+Triangle, 33011样本, 50 epochs, batch_size=40

### DeepONet 训练
- 命令: `.venv/bin/python train_model.py configs/training/DON.yaml`
- 后台任务ID: `bn9qvc1l5`
- 日志: `logs/train_DON.log`
- 配置: Prada2013, CC+PLS+Triangle+GRF, 2200样本, 100 epochs, batch_size=20

### ⚠️ 问题：三任务并行导致系统卡死
三模型同时训练导致内存/GPU过载，已重启电脑。改为顺序执行。

### 新建文件/目录
| 文件/目录 | 操作 |
|-----------|------|
| `logs/` | mkdir 新建 |
| `REPRODUCTION_LOG.md` | 新建（本操作日志） |

---

## Step 1: FNO 训练（顺序执行，第一个）

**开始时间**: 2026-06-03 23:22
**命令**: `.venv/bin/python train_model.py configs/training/FNO.yaml`
**PID**: 757

- 数据集: Prada2013, 4种电流 (CC/PLS/Triangle/GRF), 11000 样本
- 超参: 150 epochs, batch_size=20, peak_lr=1e-2
- 23:23 — JAX 编译完成，开始训练 Anode (Prada2013_CC), epoch 0/150
- 00:35 — ✅ FNO 训练完成！8个模型文件 (~38MB), 耗时约73分钟

## Step 2: CAPE-FNO2 训练（顺序执行，第二个）

**开始时间**: 2026-06-04 00:36
**命令**: `.venv/bin/python train_model.py configs/training/CAPE_FNO2.yaml`
**配置**: Prada2013, CC+PLS+GRF+Triangle, 33011样本, 50 epochs, batch_size=40
- 14:06 — ✅ CAPE-FNO2 完成！8个模型文件 (452MB), 耗时约5小时

## Step 3: DeepONet 训练（顺序执行，第三个）

**开始时间**: 2026-06-04 14:07
**命令**: `.venv/bin/python train_model.py configs/training/DON.yaml`
**配置**: Prada2013, CC+PLS+Triangle+GRF, 2200样本, 100 epochs, batch_size=20
- 14:26 — ✅ DeepONet 完成！8个模型文件 (232MB), 耗时约19分钟

---

## 训练完成汇总

| 模型 | 文件数 | 大小 | 耗时 |
|------|--------|------|------|
| FNO | 8 | 38 MB | ~73 分钟 |
| CAPE-FNO2 (PE-FNO) | 8 | 452 MB | ~5 小时 |
| DeepONet (DON) | 8 | 232 MB | ~19 分钟 |
| **合计** | **24** | **722 MB** | **~6.5 小时** |

---

## Step 4: 评估模型误差

### FNO 评估结果

| 指标 | CC | Triangle | PLS | GRF | **综合** |
|------|-----|----------|-----|-----|----------|
| 浓度 Rel L2 | 0.058% | 0.046% | 0.176% | 0.141% | **0.099%** |
| 浓度 Rel L∞ | 0.105% | 0.099% | 0.289% | 0.231% | **0.172%** |
| 电压 MAE | 0.373 mV | 0.459 mV | 1.120 mV | 1.419 mV | **0.777 mV** |
| 电压 Rel L2 | 0.025% | 0.031% | 0.085% | 0.123% | **0.060%** |

### CAPE-FNO2 评估结果

| 指标 | CC | Triangle | PLS | GRF | **综合** |
|------|-----|----------|-----|-----|----------|
| 浓度 Rel L2 | 0.136% | 0.114% | 0.283% | 0.286% | **0.185%** |
| 浓度 Rel L∞ | 0.273% | 0.227% | 0.573% | 0.545% | **0.369%** |
| 电压 MAE | 1.119 mV | 1.439 mV | 1.725 mV | 2.500 mV | **1.571 mV** |
| 电压 Rel L2 | 0.087% | 0.103% | 0.142% | 0.215% | **0.123%** |

### DeepONet 评估结果

| 指标 | CC | Triangle | PLS | GRF | **综合** |
|------|-----|----------|-----|-----|----------|
| 浓度 Rel L2 | 0.797% | 3.350% | 2.596% | 13.217% | **4.613%** |
| 浓度 Rel L∞ | 1.524% | 4.001% | 3.718% | 16.530% | **5.903%** |
| 电压 MAE | 8.157 mV | 22.534 mV | 11.609 mV | 40.601 mV | **20.807 mV** |
| 电压 Rel L2 | 0.450% | 1.259% | 0.666% | 2.091% | **1.132%** |

### 三模型综合对比

| 模型 | 浓度 Rel L2 | 浓度 Rel L∞ | 电压 MAE (mV) | 电压 Rel L2 |
|------|------------|------------|--------------|------------|
| **FNO** | **0.099%** | **0.172%** | **0.777** | **0.060%** |
| CAPE-FNO2 | 0.185% | 0.369% | 1.571 | 0.123% |
| DeepONet | 4.613% | 5.903% | 20.807 | 1.132% |

> FNO 表现最佳，CAPE-FNO2 次之，DeepONet 在少数据下误差较大（尤其 GRF 电流）。

---

## Step 5: 生成出版级对比图

**命令**: `.venv/bin/python plots_concentration_voltage.py`
**输出**:
- `plots/metrics_comparison/concentration_error_metrics_comparison.png` (246 KB)
- `plots/metrics_comparison/voltage_error_metrics_comparison.png` (285 KB)

---

## 全部完成 ✅

全部 5 步已执行完毕，详见上方各阶段记录。
