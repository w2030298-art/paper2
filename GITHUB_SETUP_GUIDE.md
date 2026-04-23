# GitHub 配置指南

## 当前状态
✅ Git 已初始化  
✅ SSH 密钥已存在  
⏳ 需要配置远程仓库

---

## 步骤 1: 将 SSH 密钥添加到 GitHub

### 1.1 复制你的 SSH 公钥
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFuoOF1x5MwYpyhNNZNYe5vQYCuOm/bBBYdgfgAa07MT your_email@example.com
```

### 1.2 添加到 GitHub
1. 打开浏览器访问: https://github.com/settings/keys
2. 点击 **"New SSH key"**
3. 填写信息:
   - **Title**: `我的Windows电脑` (或其他描述性名称)
   - **Key type**: `Authentication Key`
   - **Key**: 粘贴上方的 SSH 公钥
4. 点击 **"Add SSH key"**

---

## 步骤 2: 创建 GitHub 仓库

### 2.1 在线创建
1. 访问: https://github.com/new
2. 填写:
   - **Repository name**: `paper2`
   - **Description**: `RL-MEC 系统模型框架`
   - **Private/Public**: 根据需要选择
   - ⚠️ **不要**勾选 "Add a README file" (因为本地已有)
3. 点击 **"Create repository"**

### 2.2 复制仓库 URL
创建后,GitHub会显示仓库地址,例如:
- SSH: `git@github.com:你的用户名/paper2.git`
- HTTPS: `https://github.com/你的用户名/paper2.git`

---

## 步骤 3: 连接本地仓库到 GitHub

**方法 A - 如果有现有仓库 URL:**
```bash
git remote add origin git@github.com:你的用户名/paper2.git
```

**方法 B - 如果是新创建的仓库:**
```bash
git remote add origin https://github.com/你的用户名/paper2.git
```

---

## 步骤 4: 创建初始提交并推送

```bash
# 1. 添加所有文件到暂存区
git add .

# 2. 创建提交
git commit -m "Initial commit: RL-MEC 系统框架"

# 3. 推送到 GitHub
git push -u origin main
```

---

## 步骤 5: VSCode 中验证

1. 在 VSCode 中打开项目
2. 点击左侧 **源代码管理** (Git图标)
3. 应该能看到:
   - ✅ 远程仓库已连接
   - ✅ 分支显示为 `main`
   - ✅ 可以看到提交历史

---

## 常见问题

### Q: SSH 密钥添加失败?
确保 GitHub 账户已验证邮箱。

### Q: Push 被拒绝?
可能原因:
- 仓库不是空的 (选择 "Initialize" 时创建了文件)
- 权限不足 (需要先在 GitHub 创建仓库)

### Q: 如何查看远程仓库配置?
```bash
git remote -v
```

---

## 下一步

完成配置后,你可以在 VSCode 中:
1. 📝 编辑代码
2. 🔄 使用 Git 面板提交更改 (Ctrl+Shift+G)
3. ⬆️ 推送到 GitHub
4. 📥 从 GitHub 拉取更新
