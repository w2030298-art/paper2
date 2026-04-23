# VSCode GitHub 配置完整指南

## 📋 当前状态
✅ Git 已初始化  
✅ SSH 密钥已存在  
⏳ 需要配置 VSCode GitHub 扩展和仓库连接  

---

## 步骤 1: 安装 VSCode GitHub 扩展

### 1.1 打开 VSCode
1. 启动 VSCode
2. 按 `Ctrl + Shift + X` 打开扩展视图

### 1.2 搜索并安装
在扩展商店搜索框中输入: `github`

安装以下扩展(通常已预装):
- **GitHub Pull Requests and Issues**
- **GitHub Authentication Provider**

### 1.3 验证安装
安装完成后:
- VSCode 左侧活动栏会出现 GitHub 图标
- 底部状态栏会显示 GitHub 登录状态

---

## 步骤 2: 使用 Token 登录 GitHub

### 2.1 在 VSCode 中登录
1. 按 `Ctrl + Shift + P` 打开命令面板
2. 输入 `GitHub: Sign In`
3. 选择 **"Sign in with GitHub Authentication Provider"**
4. 浏览器会打开 GitHub 授权页面

### 2.2 输入你的 Token
1. 当 VSCode 提示输入 Token 时,粘贴你的 Personal Access Token
2. 按 Enter 确认

### 2.3 Token 格式
确保你的 Token 格式正确:
- 应该类似于: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- 或者: `github_pat_xxxxxxxxxxxxxx`

---

## 步骤 3: 在 GitHub 创建仓库

### 3.1 在线创建仓库
1. 打开浏览器访问: https://github.com/new
2. 填写仓库信息:
   - **Repository name**: `paper2`
   - **Description**: `RL-MEC 系统模型框架`
   - **Visibility**: 选择 Private (私有) 或 Public (公开)
   - ⚠️ **不要勾选** "Add a README file" (因为本地已有)

### 3.2 复制仓库 URL
创建完成后,复制仓库的 HTTPS URL,例如:
```
https://github.com/你的用户名/paper2.git
```

---

## 步骤 4: 配置 Git 远程仓库

### 4.1 设置远程仓库
打开终端(在 VSCode 中按 `` Ctrl + ` ``),执行:

```bash
git remote add origin https://github.com/你的用户名/paper2.git
```

### 4.2 验证远程配置
```bash
git remote -v
```

应该看到:
```
origin  https://github.com/你的用户名/paper2.git (fetch)
origin  https://github.com/你的用户名/paper2.git (push)
```

### 4.3 配置 GitHub Token 认证
为了让 Git 使用你的 Token,需要配置凭据存储:

```bash
git config --global credential.helper store
```

---

## 步骤 5: 首次提交并推送

### 5.1 在 VSCode 中提交
1. 按 `Ctrl + Shift + G` 打开 Git 视图
2. 点击 **"+"** 暂存所有更改
3. 输入提交信息: `Initial commit: RL-MEC 系统框架`
4. 点击 **"✓"** 提交

### 5.2 或者使用终端命令
```bash
git add .
git commit -m "Initial commit: RL-MEC 系统框架"
git push -u origin main
```

### 5.3 首次推送时
系统会提示输入用户名和密码:
- **用户名**: 你的 GitHub 用户名或邮箱
- **密码**: 使用你的 Personal Access Token

---

## 步骤 6: 验证配置成功

### 6.1 检查 VSCode 状态
- 左下角应该显示 `main` 分支
- GitHub 图标应该显示已登录
- 可以看到提交历史

### 6.2 测试功能
尝试在 VSCode 中:
1. 修改一个文件
2. 在 Git 视图中查看更改
3. 创建一个新分支
4. 推送到 GitHub

---

## 🔧 高级配置

### 配置用户信息(如果还没有)
```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"
```

### 设置默认分支为 main
```bash
git config --global init.defaultBranch main
```

### 启用 GitHub 集成
```bash
# 在 VSCode 命令面板中
# Ctrl + Shift + P
# 输入: "GitHub: Enable GitHub"
```

---

## ❓ 常见问题

### Q: Token 在哪里找?
访问: https://github.com/settings/tokens

如果需要创建新 Token:
1. 点击 **"Generate new token (classic)"**
2. 设置名称和过期时间
3. 勾选 `repo` 权限
4. 生成并复制 Token

### Q: Push 被拒绝?
可能原因:
- Token 权限不足(需要 repo 权限)
- 用户名或密码错误(密码应该是 Token)
- 仓库不存在

### Q: 如何查看配置?
```bash
git config --list
git remote -v
```

---

## ✅ 完成后你可以

1. 📝 在 VSCode 中编辑代码
2. 🔄 使用 Git 视图提交更改
3. ⏭️ 推送和拉取代码
4. 🌿 创建和管理分支
5. 🔀 发起 Pull Requests
6. 📋 查看 GitHub Issues

---

## 📞 下一步

配置完成后,你可以:
1. 打开 VSCode 的 GitHub 视图 (`Ctrl + Shift + G`)
2. 尝试提交一个小的更改
3. 推送到 GitHub
4. 在 GitHub 上查看你的代码
