# 提交消息

提交消息写清楚这次改了什么，方便自己回看实验，也方便评审把代码与实验记录对应起来。仓库采用 Conventional Commits 格式，可以手写，也可以用 Commitizen 辅助填写。

## 格式

```text
<type>(Pxxx): <subject>
```

`Pxxx` 填本人的参赛编号；主题用一句话说明具体改动，不超过 50 字，结尾不加句号。需要解释原因时，空一行再写正文。

| 类型 | 适用的改动 |
| --- | --- |
| `feat` | 增加功能或策略行为 |
| `fix` | 修复错误 |
| `docs` | 更新文档、实验记录或报告 |
| `refactor` | 调整代码结构，不改变行为 |
| `test` | 增加或修改测试 |
| `perf` | 优化运行性能 |
| `chore` | 调整配置、依赖或其他维护事项 |

## 示例

```text
feat(P017): 增加目标速度的历史估计窗口
fix(P017): 修正归一化统计导出的均值符号
docs(P017): 补充公开测试的复现命令
test(P017): 检查 reset 是否清空上一回合的记忆
chore(P017): 初始化参赛目录
```

避免只写 `update`、`修改` 或 `111`，这些消息看不出改动内容。一次提交尽量围绕一件事；互不相关的修改分开提交。

## 手工提交

在正确的编号分支或功能分支上，暂存本人文件并检查后提交：

```sh
git add participant/P017/
git diff --cached
git commit -m "fix(P017): 修正归一化统计导出的均值符号"
```

组织方维护公共文件时可以省略编号。参赛提交请使用自己的编号，不要填写他人的编号。

## 使用 Commitizen（可选）

如果希望交互式填写消息，可以在单独的本地工具环境中安装 [Commitizen](https://commitizen-tools.github.io/commitizen/)。它不属于训练或推理依赖，不需要写入提交的依赖文件。

```sh
python -m pip install commitizen
cz commit
```

按提示选择类型，在 scope 中填本人编号，再填写主题。手工提交与 Commitizen 生成的提交使用相同的格式；不必为使用这个工具修改仓库的公共配置。

## 回看实验

按编号筛选提交历史：

```sh
git log --oneline --fixed-strings --grep="(P017)"
```

提交消息说明改了什么，`LOG.md` 记录尝试的原因和结论，`experiments.csv` 保存具体配置与成绩。把实验对应的提交标识记下来，之后才能找到产生那份结果的代码。
