# VSCode 调试配置教程

本文档说明如何使用 VSCode 调试本项目（GRPO-MEC 对比算法项目）的所有功能。

## 📁 配置文件位置

- `.vscode/launch.json` - 调试启动配置
- `.vscode/settings.json` - 工作区设置

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐使用 uv）
uv venv .venv
uv pip install -r requirements.txt

# 或使用 pip
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 安装调试依赖

```bash
pip install debugpy
```

### 3. 在 VSCode 中打开项目

```bash
code .
```

### 4. 选择 Python 解释器

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter`
3. 选择 `.venv/Scripts/python.exe`

## 🎯 调试配置说明

### 🟢 训练模块 (Train)

| 配置名称 | 说明 | 适用场景 |
|---------|------|---------|
| 🟢 Train GRPO | 训练 GRPO 算法 | 调试 GRPO 算法核心逻辑 |
| 🟢 Train PPO | 训练 PPO 算法 | 调试 PPO 算法 |
| 🟢 Train SAC | 训练 SAC 算法 | 调试 SAC 算法 |

**使用方法：**
1. 打开 `scripts/train.py`
2. 在需要调试的代码行上设置断点
3. 按 `F5` 或点击调试按钮
4. 选择对应的训练配置

### 🟡 基准测试模块 (Benchmark)

| 配置名称 | 说明 | 适用场景 |
|---------|------|---------|
| 🟡 Benchmark All (Quick) | 快速测试 3 个算法 | 快速验证代码修改 |
| 🟡 Benchmark All (Full) | 完整测试 17 个算法 | 全面性能评测 |
| 🟡 Benchmark Single | 单算法测试 | 调试特定算法 |

**使用方法：**
1. 打开 `scripts/benchmark.py`
2. 设置断点
3. 选择调试配置

### 🔵 评估模块 (Evaluate)

| 配置名称 | 说明 | 适用场景 |
|---------|------|---------|
| 🔵 Evaluate Checkpoint | 评估训练好的模型 | 验证训练结果 |

**前置条件：** 需要有训练好的 checkpoint 文件

### 🟣 绘图模块 (Plot)

| 配置名称 | 说明 | 适用场景 |
|---------|------|---------|
| 🟣 Plot Results | 生成 PNG 图表 | 常规结果可视化 |
| 🟣 Plot Quick Results | 生成 PDF 图表 | 论文图表 |

### 🧪 测试模块 (Pytest)

| 配置名称 | 说明 | 适用场景 |
|---------|------|---------|
| 🧪 Pytest (Current File) | 运行当前文件测试 | 调试单个测试文件 |
| 🧪 Pytest (All Tests) | 运行所有测试 | 完整测试套件 |
| 🧪 Pytest (MEC Envs) | 运行环境测试 | 调试 MEC 环境 |

### 🔴 通用调试

| 配置名称 | 说明 | 适用场景 |
|---------|------|---------|
| 🔴 Python: Current File | 运行当前打开的 Python 文件 | 快速调试任意脚本 |

## 📋 常用命令

### 调试快捷键

| 快捷键 | 功能 |
|-------|------|
| `F5` | 开始调试 / 继续运行 |
| `F9` | 设置/取消断点 |
| `F10` | 单步跳过 |
| `F11` | 单步进入 |
| `Shift+F5` | 停止调试 |
| `Shift+F11` | 单步跳出 |
| `Ctrl+Shift+F5` | 重新开始调试 |
| `Ctrl+F5` | 不调试运行 |
| `F6` | 暂停程序 |

### 断点类型

1. **代码断点** - 在代码行左侧点击
2. **条件断点** - 右键断点 → 设置条件
3. **日志点** - 右键断点 → 选择 "Log Message"

### ⏹️ 如何中断训练

**在调试中停止训练的方法：**

1. **点击停止按钮** - 调试工具栏中的红色方块按钮
2. **使用快捷键** - `Shift+F5` 立即停止
3. **在终端中** - 按 `Ctrl+C`

**暂停/恢复运行：**
- `F6` - 暂停程序（训练会停在当前断点）
- `F5` - 继续运行

**设置条件断点提前终止：**
- 在 trainer.train() 循环内设置条件断点
- 右键断点 → 选择 "Edit Condition"
- 输入条件如 `self.total_steps > 1000` 在训练 1000 步后自动中断

## 🔧 调试训练流程

### 示例：调试 GRPO 算法训练

1. **设置断点**
   ```python
   # scripts/train.py 第 84 行附近
   agent = create_agent(algo, env, cfg, device)
   ```

2. **启动调试**
   - 选择配置：`🟢 Train GRPO`
   - 按 `F5` 开始

3. **观察变量**
   - `algo` - 算法名称
   - `env` - 环境实例
   - `cfg` - 配置字典
   - `device` - 计算设备

### 示例：调试 Benchmark 单算法

1. **设置断点**
   ```python
   # scripts/benchmark.py 第 259 行附近
   trainer.train()
   ```

2. **启动调试**
   - 选择配置：`🟡 Benchmark Single Algorithm`
   - 按 `F5` 开始

## 🎨 自定义调试配置

如果需要添加新的训练算法配置，例如添加 DDPG：

```json
{
    "name": "🟢 Train DDPG",
    "type": "debugpy",
    "request": "launch",
    "module": "scripts.train",
    "cwd": "${workspaceFolder}",
    "console": "integratedTerminal",
    "justMyCode": false,
    "env": {
        "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    },
    "args": [
        "--config", "configs/algorithms/ddpg.yaml",
        "--algorithm", "ddpg",
        "--timesteps", "5000",
        "--seed", "42"
    ]
}
```

## 🐛 常见问题

### 1. 找不到模块 `scripts.train`

**解决方案：**
- 确认 PYTHONPATH 包含项目根目录
- 检查 `settings.json` 中的 `python.analysis.extraPaths`

### 2. 断点不生效

**解决方案：**
- 确保 `justMyCode` 设置为 `false`
- 确认使用的是 debugpy 类型

### 3. 调试速度慢

**解决方案：**
- 减少 `timesteps` 参数
- 使用 `--quiet` 减少日志输出

### 4. CUDA 相关问题

**解决方案：**
- 将 `device` 参数改为 `"cpu"` 进行调试
- 避免在 GPU 相关代码处设置过多断点

## 📊 调试技巧

### 1. 观察环境交互

在 `env.step()` 附近设置断点，观察：
- 状态 (observation)
- 奖励 (reward)
- 完成标志 (done)

### 2. 观察智能体决策

在 `agent.select_action()` 或 `agent.get_action()` 处设置断点

### 3. 观察梯度更新

在 trainer 的 `update()` 方法中设置断点

### 4. 使用调试控制台

在调试控制台中可以：
- 修改变量值
- 调用函数
- 计算表达式

## 🎯 推荐调试流程

1. **单元测试** - 先用 Pytest 验证单个模块
2. **单算法训练** - 调试一个算法的完整流程
3. **Benchmark** - 验证算法对比功能
4. **评估** - 验证训练结果

## 📝 脚本参数参考

### train.py 参数

```
--config       配置文件路径 (默认: configs/algorithms/grpo.yaml)
--algorithm    算法名称 (如: GRPO, PPO, SAC)
--env          环境名称 (默认: auto)
--timesteps    训练步数 (覆盖配置)
--seed         随机种子 (默认: 42)
--device       计算设备 (auto/cuda/cpu)
--save-dir     保存目录
--eval-episodes 评估 episode 数
```

### benchmark.py 参数

```
--algorithms   算法列表 (如: grpo ppo sac)
--all          运行所有 17 个算法
--env          环境名称 (auto/离散/连续)
--episodes     评估 episode 数
--timesteps    训练步数
--seeds        随机种子列表
--device       计算设备
--output       输出 JSON 路径
--quiet        静默模式
```

### evaluate.py 参数

```
--checkpoint   模型检查点路径 (必需)
--config        配置文件
--algorithm     算法名称
--env           环境名称
--num-episodes  评估 episode 数
--seed          随机种子
--device        计算设备
--save-results  保存结果路径
```

### plot_results.py 参数

```
--input    输入 JSON 路径
--output   输出目录
--format   输出格式 (png/pdf/svg)
```

## ✅ 验证配置

运行以下命令验证环境：

```bash
python -c "import torch; import gymnasium; print('Dependencies OK')"
```

## 📞 获取帮助

如遇问题，请检查：
1. `.vscode` 目录是否存在且配置正确
2. 虚拟环境是否激活
3. 所有依赖是否已安装
4. Python 版本是否 >= 3.10
