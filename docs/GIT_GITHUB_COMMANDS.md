# Git 连接 GitHub 仓库命令手册

这份文档面向当前总仓库：

- 本地仓库：`D:\Code\all`
- 当前分支：`main`
- 当前远程：`origin`
- 当前 GitHub 仓库：`https://github.com/yezi56/LiuZheCode.git`

适用目标：

- 初始化本地代码仓库
- 连接 GitHub 远程仓库
- 首次推送
- 日常提交与同步
- 更换远程仓库
- 新分支开发
- 处理认证与常见报错

## 1. 基础检查

在 `PowerShell` 或 `cmd` 都可以执行，推荐用 VS Code 终端里的 `PowerShell`。

```powershell
git --version
git config --global --list
```

如果没有配置身份，先执行：

```powershell
git config --global user.name "yezi56"
git config --global user.email "2421564211@qq.com"
```

## 2. 查看当前仓库状态

进入总仓库：

```powershell
cd D:\Code\all
```

查看状态、分支、远程：

```powershell
git status
git branch --show-current
git remote -v
```

当前这份仓库正常情况下应显示：

```text
分支: main
远程: origin -> https://github.com/yezi56/LiuZheCode.git
```

## 3. 当前仓库最常用命令

这是你以后最常用的一组：

```powershell
cd D:\Code\all
git add -A
git commit -m "写你的提交说明"
git push
```

如果只是先看看有哪些改动：

```powershell
git status --short
```

如果想先拉一下远程最新代码再继续：

```powershell
git pull origin main
```

## 4. 场景一：本地代码还不是 Git 仓库

适用于你以后新建一个目录，想推到 GitHub。

```powershell
cd D:\你的代码目录
git init
git add -A
git commit -m "init project"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

## 5. 场景二：本地已经是 Git 仓库，但还没有远程

先检查：

```powershell
git remote -v
```

如果没有任何输出，说明还没绑远程，执行：

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
git branch -M main
git push -u origin main
```

## 6. 场景三：远程仓库地址绑错了，需要更换

查看当前远程：

```powershell
git remote -v
```

修改远程地址：

```powershell
git remote set-url origin https://github.com/你的用户名/新的仓库名.git
```

再次检查：

```powershell
git remote -v
```

然后推送：

```powershell
git push -u origin main
```

## 7. 场景四：已经连接好远程，只想正常提交代码

```powershell
cd D:\Code\all
git status
git add -A
git commit -m "feat: 更新实验代码"
git push
```

## 8. 场景五：新开一个分支做实验

适合做论文实验，不想把所有改动都堆在 `main` 上。

新建并切换分支：

```powershell
git checkout -b exp/cbam
```

提交并推送这个新分支：

```powershell
git add -A
git commit -m "exp: add cbam experiment"
git push -u origin exp/cbam
```

查看所有分支：

```powershell
git branch
git branch -r
```

切回主分支：

```powershell
git checkout main
```

## 9. 场景六：把远程最新代码拉下来

当前分支更新：

```powershell
git pull
```

明确指定主分支：

```powershell
git pull origin main
```

如果只是想先看远程更新，不合并：

```powershell
git fetch origin
git log --oneline --all --graph --decorate -20
```

## 10. 场景七：远程仓库已经有内容，本地也有内容

这时首次推送可能会冲突。标准处理顺序：

先拉远程：

```powershell
git pull origin main --allow-unrelated-histories
```

解决冲突后再提交：

```powershell
git add -A
git commit -m "merge remote and local history"
git push origin main
```

如果 GitHub 上是空仓库，就不需要这一步。

## 11. HTTPS 登录与 Token

GitHub 现在通常不用账户密码直接推送，而是用 `Personal Access Token`。

### 生成 Token

打开：

- [GitHub Tokens](https://github.com/settings/tokens)

建议至少授予 `repo` 权限。

### 推送时怎么填

- 用户名：`yezi56`
- 密码：填 `Personal Access Token`

不是填 GitHub 登录密码。

## 12. SSH 方式连接 GitHub

如果你不想每次走 HTTPS 认证，可以改用 SSH。

### 生成 SSH key

```powershell
ssh-keygen -t ed25519 -C "2421564211@qq.com"
```

一路回车即可。

### 查看公钥

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

把输出内容复制到 GitHub：

- Settings
- SSH and GPG keys
- New SSH key

### 测试 SSH

```powershell
ssh -T git@github.com
```

### 把远程地址切到 SSH

```powershell
git remote set-url origin git@github.com:yezi56/LiuZheCode.git
git remote -v
git push
```

## 13. 当前总仓库的真实命令

### 查看当前绑定情况

```powershell
cd D:\Code\all
git branch --show-current
git remote -v
```

### 当前总仓库日常提交

```powershell
cd D:\Code\all
git add -A
git commit -m "docs: update workspace notes"
git push
```

### 当前总仓库新建实验分支

```powershell
cd D:\Code\all
git checkout -b exp/new-paper-round
git add -A
git commit -m "exp: reorganize segmentation experiments"
git push -u origin exp/new-paper-round
```

## 14. 常见报错与处理

### 14.1 `fatal: not a git repository`

说明你当前目录不是 Git 仓库。

先切到正确目录：

```powershell
cd D:\Code\all
```

再执行：

```powershell
git status
```

### 14.2 `remote origin already exists`

说明你已经绑过远程了。

看一下当前远程：

```powershell
git remote -v
```

如果只是想改地址：

```powershell
git remote set-url origin https://github.com/你的用户名/你的仓库名.git
```

### 14.3 `failed to push some refs`

通常是远程比你本地更新。

先拉再推：

```powershell
git pull origin main
git push origin main
```

### 14.4 `Permission denied`

可能是：

- 终端权限问题
- 文件被 VS Code、资源管理器或其他程序占用
- SSH key / Token 没配好

优先检查：

```powershell
git remote -v
git status
```

如果是 GitHub 认证问题，优先换用 Token 或 SSH。

### 14.5 `src refspec main does not match any`

说明还没有提交，或者当前分支不叫 `main`。

先提交一次：

```powershell
git add -A
git commit -m "init"
```

再把分支改成 `main`：

```powershell
git branch -M main
git push -u origin main
```

## 15. 推荐习惯

### 总仓库级别

- 所有模型统一放在 `D:\Code\all`
- 用一个总仓库统一管理论文实验
- 用分支管理不同实验路线

### 提交习惯

推荐提交信息风格：

```text
feat: 新增模块
fix: 修复训练脚本
docs: 更新实验说明
exp: 添加注意力实验
refactor: 调整目录结构
```

## 16. 一套最短可复制命令

### 当前仓库直接提交推送

```powershell
cd D:\Code\all
git add -A
git commit -m "docs: update git manual"
git push
```

### 当前仓库换远程地址

```powershell
cd D:\Code\all
git remote set-url origin https://github.com/yezi56/LiuZheCode.git
git remote -v
git push
```

### 新项目首次推送

```powershell
cd D:\你的项目目录
git init
git add -A
git commit -m "init project"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```
