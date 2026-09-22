# Python 多版本管理

本文说明如何在同一台机器上管理多个 Python 版本，并为本仓库准备 3.12 环境。整体环境配置、安装命令与提交预检见[使用指南](how_to_use.md)。

同一台机器上通常同时存在系统 Python、conda 环境或其他项目的解释器。下面的做法是让它们共存，而不改动系统默认解释器。

## 版本要求

本仓库声明的解释器范围是 `pyproject.toml` 中的 `requires-python = ">=3.12,<3.13"`，`.python-version` 记录的是 `3.12.9`。**主、次版本必须是 3.12**：3.10/3.11 装的依赖与协议实现不匹配，3.13 也不在声明范围内。补丁版本（如 3.12.14）可以不同。

评测环境和训练环境可以各自独立，但**两者都必须是 3.12**，否则训练产物与推理结果可能对不上。

## 先确认现有版本

Windows PowerShell：

```powershell
py -0
py -0p
```

`py -0` 列出已安装的版本，`py -0p` 额外列出各自的路径。没有 `py` 命令时，说明安装 Python 时没有勾选 py launcher（见下文）。

Linux / macOS：

```sh
python3 --version
which -a python3 python3.12 python3.13
ls /usr/bin/python3* /usr/local/bin/python3* 2>/dev/null
```

期望能找到 `python3.12`（或 Windows 上 `py -3.12` 可用）。如果只有 3.13 或其他版本，先按下面安装一个 3.12，**不要去改系统默认的 `python3`**。

## 安装 3.12

按下表选择一种方式即可。

| 平台 | 方式 | 命令 |
| --- | --- | --- |
| Windows | python.org 安装器 | 下载 3.12.x 安装包，安装时勾选 **py launcher** 与 **Add python.exe to PATH**，完成后用 `py -3.12 --version` 验证 |
| Ubuntu / Debian | 发行版包（较新版本） | `sudo apt update && sudo apt install python3.12 python3.12-venv` |
| Ubuntu / Debian | deadsnakes PPA（旧发行版） | `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install python3.12 python3.12-venv` |
| macOS | Homebrew | `brew install python@3.12`，然后用 `python3.12` 显式调用 |
| 通用 | pyenv | `pyenv install 3.12.9 && pyenv local 3.12.9` |
| 通用 | conda / miniconda | `conda create -n coverage312 python=3.12 && conda activate coverage312` |
| 通用 | uv（本仓库的锁文件由 uv 导出，可选） | `uv python install 3.12 && uv venv --python 3.12 .venv` |

不要用 `sudo pip install` 往系统解释器里装本仓库依赖。所有依赖都装在虚拟环境里。使用 uv 时请显式指定 `--python 3.12`，不要用 `uv python pin` 改写仓库自带的 `.python-version`（那是官方文件）。

## 用显式解释器创建虚拟环境

虚拟环境会继承**创建它的那个解释器**的版本，所以关键是不要用含糊的 `python`：

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
```

Linux / macOS：

```sh
python3.12 -m venv .venv
```

如果 `python3.12` 不在 PATH 里（例如 conda 环境内、或 Homebrew 未链接），就写完整路径，或激活 conda 环境后直接用 `python -m venv .venv`。使用 uv 时它已经按上面的命令建好 `.venv`，[使用指南](how_to_use.md)中的主流程保留 `.venv` 这个目录名即可。

创建后立即确认这个环境真的是 3.12：

```sh
.venv/bin/python -c "import sys; print(sys.version); print(sys.executable)"
```

Windows 把第一段换成 `.venv/Scripts/python.exe`。输出中的版本应以 `3.12.` 开头，路径应指向本仓库内的 `.venv`。

## 各版本共存时的注意点

- **一律用 `python -m pip` 而不是裸 `pip`。** 多版本环境下 `pip` 可能属于另一个解释器，装到错误的位置。
- **优先写绝对路径。** 使用指南中出现的 `python` 都是简写；不确定当前指向时，评测用 `.venv/bin/python`（Windows 为 `.venv/Scripts/python.exe`），训练用 `.venv-train` 下的同名文件。
- **不要修改系统 Python 的默认指向。** 无需使用 `update-alternatives`、`brew unlink`、取消勾选 PATH 之类的手段；用 `py -3.12`、`python3.12` 或虚拟环境内的绝对路径即可。
- **不要在同一个目录里复用版本不同的 `.venv`。** 如果它已经是 3.13 或 3.10，直接删掉重建，比升级环境更省事。
- **conda 环境内不要依赖 `python3.12`。** conda 只提供 `python`，此时用 `python -m venv .venv` 或直接在该环境里工作。
- **Windows 上 `python` 可能指向 Microsoft Store 的转发器。** 它的行为与常规安装不同，建议改用 `py -3.12`。
- 训练环境 `.venv-train` 与评测环境 `.venv` 必须来自同一主、次版本，否则导出与推理的数值可能不一致。

## 自检命令

搭建完成后，下面几条能快速定位版本问题：

| 命令 | 期望结果 |
| --- | --- |
| `py -0`（Windows） | 列表中出现 `3.12` |
| `python3.12 --version`（Linux / macOS） | `Python 3.12.x` |
| `.venv/bin/python -c "import sys; print(sys.version)"` | 以 `3.12.` 开头 |
| `.venv/bin/python -c "import sys; print(sys.executable)"` | 指向本仓库的 `.venv` |
| `.venv/bin/python -c "import sys; print(sys.prefix != sys.base_prefix)"` | 输出 `True`，说明当前在虚拟环境内 |

## 常见问题

### 找不到 `python3.12`，或虚拟环境版本不是 3.12

`python --version` 显示的版本、`python -m venv` 建出的环境版本，都可能来自另一套解释器。先确认虚拟环境内的实际版本：

```sh
.venv/bin/python -c "import sys; print(sys.version); print(sys.executable)"
```

版本不是 `3.12.` 时，删除 `.venv` 并用 `py -3.12 -m venv .venv`（Windows）或 `python3.12 -m venv .venv`（Linux / macOS）重建；`python3.12` 本身不存在时，按本文[安装 3.12](#安装-312) 处理。注意不要用裸 `pip`，也不要改系统默认的 `python3`。

### `py` 命令不存在

Windows 下安装 Python 时未勾选 py launcher。重新运行 3.12 安装包并勾选该选项，或改用安装目录下的 `python.exe`。

### 训练环境和评测环境版本不一致

分别确认两个环境的实际版本：

```sh
.venv/bin/python -c "import sys; print(sys.version)"
.venv-train/bin/python -c "import sys; print(sys.version)"
```

两行都应以 `3.12.` 开头。不一致时，把版本不对的那个环境删除后用 3.12 重建。

## 接下来

回到[使用指南](how_to_use.md)安装锁定依赖并跑通模板。
