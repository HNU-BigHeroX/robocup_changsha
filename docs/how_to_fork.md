# 获取仓库并开始开发

本次比赛直接在官方比赛仓库中使用个人分支，不需要 Fork。先向组织方确认仓库地址、访问权限和自己的参赛编号。

下面用 `P017` 举例。请将它替换为自己的编号，并依次完成：确认编号 → 检查并准备分支 → 创建本人目录 → 开发。

## 克隆仓库

将下面的地址换成组织方提供的地址：

```sh
git clone <仓库地址> robocup-mpe2
cd robocup-mpe2
```

本文后面的命令都在这个目录中执行。如果仓库已经在本地，直接进入仓库，不必再克隆一份。

## 先准备编号分支

先查看当前改动和已有分支：

```sh
git status --short
git fetch origin
git branch --all
```

工作区有尚未保存的改动时，先确认它们属于谁、应该保留在哪条分支，不要直接覆盖。首次参赛且本地、远程都没有 `P017` 分支时，从官方主线创建：

```sh
git switch -c P017 origin/main
```

如果本地已有本人分支，使用 `git switch P017`；如果只有远程存在，使用 `git switch --track origin/P017`。不要重复创建，也不要借用别人的编号。

用 `git branch --show-current` 确认当前是 `P017`，再进行下一步。后续需要功能分支时，先按 [Git 工作流程](git.md)准备好分支，再开始修改。

## 创建本人目录

确认 `participant/P017/` 尚不存在，然后复制模板。已有本人目录时继续使用，不要用模板覆盖。

Windows PowerShell：

```powershell
Copy-Item -LiteralPath participant/_template -Destination participant/P017 -Recurse
```

Linux / macOS：

```sh
cp -R participant/_template participant/P017
```

打开 `participant/P017/submission.yaml`，将 `participant_id` 改为自己的编号：

```yaml
participant_id: "P017"
```

模板已经带有示例模型和摘要。只修改编号时，不需要重算模型摘要；替换模型后则需要同步更新清单。

检查改动，再提交初始化结果：

```sh
git add participant/P017/
git diff --cached --stat
git commit -m "chore(P017): 初始化参赛目录"
git push -u origin P017
```

## 接下来

按[使用指南](how_to_use.md)配置评测环境，先跑通模板。自己的策略、训练脚本、模型和报告都写在 `participant/P017/` 中。

开发期间可以随时推送本人分支。版本标签是可选的，最终提交需要在统一提交窗口内创建 PR，要求见[参与指南](../CONTRIBUTING.md)。窗口开放前，不得向比赛仓库之外的渠道发布比赛实现。
