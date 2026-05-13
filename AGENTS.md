# AGENTS.md - Codex 执行代理规范

> 本文件放置于项目根目录，Codex CLI / Claude Code 启动时自动读取。
> 本文件定义了 AI 编码代理在本项目中的行为规范、工作流程和质量标准。

---

## 项目信息

- **项目名称**：grpo-mec
- **技术栈**：Python 3.10 + PyTorch + Gymnasium + NumPy/SciPy + Matplotlib/TensorBoard
- **项目状态**：`.ai/ledger.json`（单一状态源）
- **开发计划**：`.ai/inbox/plan.md`
- **参考资料**：`ref/`

---

## 第一部分：启动协议

### 每次新对话开始时，你必须按以下顺序执行

```text
步骤 1 -> 读取 .ai/ledger.json
          了解当前项目状态、进度和待办事项。

步骤 2 -> 读取 .ai/inbox/plan.md 中对应模块段落
          了解接下来要做什么。

步骤 3 -> 如 ref/ 下有相关参考文件，读取它们
          了解实现时需要参考的技术要点。

步骤 4 -> 使用 /graphify（如已安装）扫描项目当前状态
          了解代码结构和模块关系。

步骤 5 -> 向用户报告当前状态：
          "当前进度：[模块 X] Step [N]。上次完成了 [摘要]。
           接下来将执行 [下一步描述]。是否继续？"
```

**绝对不能跳过启动协议。** 即使用户直接给出指令，也必须先完成状态确认。

---

## 第二部分：执行规范

### 2.1 严格遵循计划

- `.ai/inbox/plan.md` 是你的唯一执行依据。
- 按照 `.ai/inbox/plan.md` 中的 Step 编号逐步执行，不跳步。
- 每个 Step 中标注的文件名、类名、方法名、逻辑必须严格遵循。
- **你不做技术决策**：如果遇到 `.ai/inbox/plan.md` 未覆盖的情况，停下来向用户报告，并写入 `.ai/ledger.json` 的 `open_items`。

### 2.2 Step 执行循环

对每一个 Step，执行以下循环：

1. 读取 Step 要求。
2. 执行编码。
3. 运行 Step 中指定的验证命令。
4. 验证通过：更新 `.ai/ledger.json` 的 `progress` / `validation`，向用户报告，然后进入下一个 Step。
5. 验证失败：分析原因并修复后重试，最多 3 次。
6. 若 3 次后仍失败：停止并向用户报告。

### 2.3 进度更新

每完成一个 Step，立即更新 `.ai/ledger.json` 中的 `progress` 字段：

```json
{
  "progress": {
    "done": 12,
    "total": 17,
    "label": "12 / 17 steps (module 24: 4/4, module 25: 8/8, module 26: 0/5)"
  }
}
```

### 2.4 遇到问题时

当遇到 `.ai/inbox/plan.md` 未覆盖的情况：

1. **不要自行决策**。
2. 将问题记录到 `.ai/ledger.json` 的 `open_items` 数组：

```json
{
  "open_items": [
    "Issue #1: [标题] - [问题描述]"
  ]
}
```

3. 向用户报告并等待指令。

### 2.5 禁止行为

- 不得偏离 `.ai/inbox/plan.md` 自行添加功能或重构。
- 不得删除或修改 `.ai/inbox/plan.md` 中未要求改动的文件。
- 不得在代码中引入 `.ai/inbox/plan.md` 未指定的依赖库。
- 不得跳过测试验证。
- 不得“优化” `.ai/inbox/plan.md` 中已确定的方案（除非发现明确的 bug）。

---

## 第三部分：代码质量标准

### 3.1 基本规范

- 所有代码必须有类型注解（Python）或 TypeScript 类型定义。
- 函数/方法必须有 `docstring` 或 JSDoc 注释。
- 不允许硬编码：配置值使用环境变量或配置文件。
- 不允许 magic number：使用命名常量。
- 错误处理：不允许裸 `except` / `catch`，必须捕获具体异常。

### 3.2 Git 提交规范

每完成一个 Step 提交一次，提交信息格式：

```text
[模块名] Step N: 简短描述

- 实现了 xxx
- 添加了 xxx 测试
- 验证状态：通过/部分通过
```

### 3.3 测试要求

- 每个 Step 中指定的测试必须编写并通过。
- 测试使用项目约定的测试框架。
- 测试文件命名：`test_[模块名].py` / `[模块名].test.ts`。
- 优先使用内存数据库或 mock 进行单元测试。

---

## 第四部分：Harness 工作流集成

### 4.1 上下文管理

本项目采用分层上下文策略：

```text
层级 1 - AGENTS.md（你正在读的文件）
  -> 项目级规范，每次对话都加载。

层级 2 - .ai/ledger.json
  -> 项目状态、进度、待办事项。

层级 3 - .ai/inbox/plan.md
  -> 当前开发计划，按需加载。

层级 4 - ref/
  -> 参考资料，按需加载。
```

### 4.2 上下文刷新策略

当对话上下文过长（输出质量下降）时：

1. 主动告知用户“建议开启新对话继续”。
2. 确保 `.ai/ledger.json` 已更新到最新状态。
3. 新对话启动后，按启动协议重新加载上下文。

### 4.3 多模块并行（高级）

如果用户要求并行开发多个无依赖模块：

1. 每个模块在独立的 worktree 分支中开发。
2. 分支命名：`feat/[module-name]`。
3. 每个分支的 `.ai/ledger.json` 独立更新。
4. 完成后由用户手动合并并解决冲突。

---

## 第五部分：Graphify 知识图谱指引

### 5.1 何时使用 Graphify

在以下场景调用 `/graphify`：

- **新对话开始**：快速了解项目全貌。
- **跨模块开发**：需要理解模块间依赖关系。
- **调试复杂 bug**：追踪调用链和数据流。
- **代码审查**：识别架构偏离。

### 5.2 Graphify 查询模式

```text
/graphify
/graphify query "模块 A 的入口点是什么"
/graphify query "哪些文件依赖 UserService"
/graphify query "数据从 API 入口到数据库的完整路径"
```

### 5.3 知识图谱与 plan.md 的配合

- `.ai/inbox/plan.md` 告诉你“应该怎么做”。
- Graphify 告诉你“现在是什么样”。
- 两者结合用于验证已实现代码是否符合计划。

---

## 第六部分：自检与报告

### 每次任务结束时的报告模板

```markdown
## 本次执行报告

### 完成内容
- [模块 X] Step N: [描述]
- [模块 X] Step N+1: [描述]

### 测试结果
- test_xxx: ✅ 通过
- test_yyy: ✅ 通过
- test_zzz: ❌ 失败（原因：...，已修复 / 需要用户确认）

### 与计划的偏差
- 无偏差 / [描述偏差及原因]

### 下一步
- 下一个待执行：[模块 X] Step N+2
- 预计工作内容：[描述]

### 问题与建议
- [如有]
```

---

## 第七部分：自定义规范区（用户填写）

### 代码风格
```text
- Python 使用 ruff 统一 lint/import 排序，行长 100
- 类型注解遵循现有 mypy 配置，命名保持 snake_case/PascalCase
- Git commit convention: [模块/范围] 做了什么
- Test framework: pytest
```

### 项目特定规则
```text
- 严格按 `.ai/inbox/plan.md` 执行，未覆盖项先记录到 `.ai/ledger.json` 的 `open_items` 并等待确认
- 每完成一个 Step 更新 `.ai/ledger.json`，并运行对应测试
- 不做未请求的重构或依赖引入
```
