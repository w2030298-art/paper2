# RL-MEC Benchmark 项目文件访问链接

## ⚠️ 重要说明

GitHub网页端可能被限制访问，请使用以下`raw.githubusercontent.com`链接访问文件内容。

## 项目基本信息

- **仓库**：`w2030298-art/paper2`
- **分支**：`main`
- **访问格式**：`https://raw.githubusercontent.com/w2030298-art/paper2/main/文件路径`

---

## 核心文件链接

### 📚 文档
- [README.md](https://raw.githubusercontent.com/w2030298-art/paper2/main/README.md)
- [项目方案](https://raw.githubusercontent.com/w2030298-art/paper2/main/docs/RL-MEC%20%E7%B3%BB%E7%BB%9F%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%96%B9%E6%A1%88.md)
- [GameTheory环境文档](https://raw.githubusercontent.com/w2030298-art/paper2/main/docs/mec_v3_game_theory.md)
- [改进文档](https://raw.githubusercontent.com/w2030298-art/paper2/main/docs/mec_improvements.md)

### ⚙️ 配置文件
- [pyproject.toml](https://raw.githubusercontent.com/w2030298-art/paper2/main/pyproject.toml)
- [requirements.txt](https://raw.githubusercontent.com/w2030298-art/paper2/main/requirements.txt)
- [default.yaml](https://raw.githubusercontent.com/w2030298-art/paper2/main/configs/default.yaml)

### 🤖 强化学习算法实现
- [base_agent.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/base_agent.py)
- [game_theory_utils.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/game_theory_utils.py)
- [grpo.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/grpo.py)
- [ppo.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/ppo.py)
- [sac.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/sac.py)
- [ddqn.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/ddqn.py)
- [ddpg.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/ddpg.py)
- [td3.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/td3.py)
- [a3c.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/a3c.py)
- [trpo.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/trpo.py)
- [simpo.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/simpo.py)
- [mappo.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/mappo.py)
- [qmix.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/qmix.py)
- [coma.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/coma.py)
- [ippo.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/ippo.py)
- [vdn.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/vdn.py)
- [maddpg.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/maddpg.py)
- [iql.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/iql.py)
- [matd3.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/matd3.py)

### 🧠 神经网络
- [networks.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/utils/networks.py)
- [buffers.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/rl_algorithms/utils/buffers.py)

### 🌍 环境
- [game_theory_env.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/environments/mec_v3/game_theory_env.py)
- [game_theory_adapters.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/environments/mec_v3/game_theory_adapters.py)
- [base_env.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/environments/mec_v2/base_env.py)

### 📊 训练器
- [base_trainer.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/trainer/base_trainer.py)
- [on_policy_trainer.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/trainer/on_policy_trainer.py)
- [off_policy_trainer.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/trainer/off_policy_trainer.py)
- [benchmark.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/trainer/benchmark.py)

### 🔧 工具函数
- [helpers.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/utils/helpers.py)
- [logger.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/utils/logger.py)
- [buffer.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/utils/buffer.py)
- [config.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/src/utils/config.py)

### 📝 脚本
- [train.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/scripts/train.py)
- [benchmark.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/scripts/benchmark.py)
- [evaluate.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/scripts/evaluate.py)
- [plot_results.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/scripts/plot_results.py)

### 🧪 测试
- [test_mec_envs.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/tests/test_mec_envs.py)
- [test_trainers.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/tests/test_trainers.py)
- [test_algorithms_on_envs.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/tests/test_algorithms_on_envs.py)
- [test_game_theory_fusion.py](https://raw.githubusercontent.com/w2030298-art/paper2/main/tests/test_game_theory_fusion.py)

### 📋 汇报文档
- [implementation_matrix.md](https://raw.githubusercontent.com/w2030298-art/paper2/main/docs/reporting/implementation_matrix.md)
- [status_snapshot.md](https://raw.githubusercontent.com/w2030298-art/paper2/main/docs/reporting/status_snapshot.md)
- [mentor_report_template.md](https://raw.githubusercontent.com/w2030298-art/paper2/main/docs/reporting/mentor_report_template.md)

---

## 使用方法

直接访问上述链接即可查看文件内容，或将链接复制给Claude进行代码分析和问答。

## 生成时间

2026-04-24
