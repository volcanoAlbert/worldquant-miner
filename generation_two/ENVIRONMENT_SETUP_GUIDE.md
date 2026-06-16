# Generation Two Environment Setup Guide

这份文档记录我们在当前机器上配置、启动、排查 `generation_two` 的过程，并整理成一份新设备配置指南。重点覆盖 macOS、`uv` 管理 Python 依赖、LLM API 配置、GUI 启动、Dock 快捷入口，以及之前遇到的问题和解决方案。

本文不会记录任何真实账号、密码或 API key。

## 当前约定

- 仓库路径：`/Users/volcano/code/worldquant-miner`
- 项目目录：`/Users/volcano/code/worldquant-miner/generation_two`
- Python 版本：`3.11`
- 虚拟环境：`generation_two/.venv`
- Python 依赖管理：`uv`
- WorldQuant/LLM 配置文件：仓库根目录的 `credential.txt`
- GUI 日志：`generation_two/generation_two_gui.log`
- GUI PID 文件：`generation_two/generation_two_gui.pid`
- 本地结果库：`generation_two/generation_two_backtests.db`
- macOS App 快捷入口：`/Users/volcano/code/worldquant-miner/Generation Two.app`

## 新设备快速配置

### 1. 安装系统工具

在一台新的 macOS 上，先安装 Command Line Tools：

```bash
xcode-select --install
```

安装 Homebrew 后，安装 `uv`：

```bash
brew install uv
```

如果你希望系统里也有一个全局可用的 `python3.11`，可以额外安装：

```bash
brew install python@3.11
brew link python@3.11
```

项目本身不依赖全局 Python 运行，推荐仍然用 `uv` 给项目创建独立 `.venv`。

### 2. 获取仓库

```bash
mkdir -p /Users/volcano/code
cd /Users/volcano/code
git clone <your-repo-url> worldquant-miner
cd /Users/volcano/code/worldquant-miner/generation_two
```

如果路径不是 `/Users/volcano/code/worldquant-miner`，后面的 `.app` 启动脚本需要同步修改路径。

### 3. 安装 Python 3.11

项目里有 `.python-version`，当前值是 `3.11`。使用 `uv` 安装对应 Python：

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv python install 3.11
```

如果你不需要把 uv 缓存放在仓库目录，也可以简单执行：

```bash
uv python install 3.11
```

### 4. 创建虚拟环境并安装依赖

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv venv --python 3.11 .venv
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv pip install -e .
```

如果使用默认 uv 缓存路径，可以简化为：

```bash
uv pip install -e .
```

安装完成后确认：

```bash
.venv/bin/python --version
.venv/bin/python -c "import requests, numpy, yaml; import tkinter; print('ok')"
```

期望 Python 是 `3.11.x`，并输出 `ok`。

## 配置 credential.txt

在仓库根目录创建：

```bash
/Users/volcano/code/worldquant-miner/credential.txt
```

推荐格式是第一行 WorldQuant Brain 账号，后面用 `key=value` 写 LLM API 配置：

```text
["your.worldquant@email.com", "your_worldquant_password"]
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_API_KEY=your_api_key
LLM_MODEL=your_model_name
LLM_TIMEOUT=120
```

也支持这些别名：

```text
OPENAI_BASE_URL=...
OPENAI_API_KEY=...
OPENAI_MODEL=...
OPENAI_TIMEOUT=...
```

注意：

- 不要把真实 `credential.txt` 提交到 git。
- 根目录已有 `credential.example.txt`，只作为格式参考。
- 程序会从命令行参数、`generation_two/`、仓库根目录、当前工作目录等位置寻找凭证。

## 启动项目

### 推荐：从项目目录启动

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
.venv/bin/python -u gui/run_gui.py ../credential.txt
```

如果想写入日志并后台运行：

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
nohup .venv/bin/python -u gui/run_gui.py ../credential.txt > generation_two_gui.log 2>&1 &
echo $! > generation_two_gui.pid
```

查看日志：

```bash
tail -n 200 /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.log
```

查看 PID：

```bash
cat /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.pid
ps -p "$(cat /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.pid)" -o pid,etime,command
```

### 从 macOS App 启动

当前 App 位于：

```text
/Users/volcano/code/worldquant-miner/Generation Two.app
```

它的启动脚本是：

```text
Generation Two.app/Contents/MacOS/generation-two-launcher
```

当前脚本逻辑：

```bash
PROJECT_DIR="/Users/volcano/code/worldquant-miner/generation_two"
LOG_FILE="$PROJECT_DIR/generation_two_gui.log"

cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" -u gui/run_gui.py ../credential.txt >> "$LOG_FILE" 2>&1
```

所以 `.app` 和项目目录启动的核心区别只是：

- `.app` 固定使用脚本里的 `PROJECT_DIR`
- `.app` 默认追加写入 `generation_two_gui.log`
- `.app` 依赖 `generation_two/.venv` 已经存在
- `.app` 使用 `../credential.txt`

如果新机器路径不同，编辑 `generation-two-launcher` 里的 `PROJECT_DIR`。

## 创建 Dock 快捷入口

如果仓库路径保持不变，可以直接把这个 App 放进 Dock：

```bash
open /Users/volcano/code/worldquant-miner
```

然后把 `Generation Two.app` 拖到 Dock。

也可以复制到 Applications：

```bash
cp -R "/Users/volcano/code/worldquant-miner/Generation Two.app" /Applications/
```

如果复制到 `/Applications`，仍然要保证 App 内部 launcher 指向真实项目目录，否则双击会找不到 `.venv` 或代码。

## 常用维护命令

### 重新安装依赖

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv pip install -e .
```

### 重建虚拟环境

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
rm -rf .venv generation_two.egg-info
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv venv --python 3.11 .venv
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv pip install -e .
```

### 关闭 GUI 实例

```bash
PID="$(cat /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.pid)"
ps -p "$PID" -o pid,etime,command
kill "$PID"
```

如果 PID 文件是 `0` 或进程不存在，说明程序已经退出。

### 检查最近 mining 结果

```bash
sqlite3 -header -column -cmd ".timeout 10000" \
  /Users/volcano/code/worldquant-miner/generation_two/generation_two_backtests.db \
  "select count(*) total,
          coalesce(sum(success),0) success_count,
          round(max(sharpe),3) best_sharpe,
          round(max(fitness),3) best_fitness
     from backtest_results;
   select datetime(timestamp,'unixepoch','localtime') time,
          alpha_id, region,
          round(sharpe,3) sharpe,
          round(fitness,3) fitness,
          round(turnover,3) turnover,
          success,
          substr(tags,1,160) tags,
          substr(error_message,1,180) error
     from backtest_results
    order by timestamp desc
    limit 20;"
```

## 我们之前遇到的问题和解决方案

### 1. 新机器没有 Python 环境

症状：

- `python` 或 `python3` 不存在，或版本不对
- GUI 不能启动
- 依赖安装混乱

解决：

- 安装 `uv`
- 用 `uv python install 3.11` 安装项目 Python
- 用 `uv venv --python 3.11 .venv` 为项目创建独立环境
- 用 `uv pip install -e .` 安装 editable package

这样避免污染系统 Python，也避免 conda/base 环境和项目依赖互相影响。

### 2. pyproject 里错误声明 tkinter 依赖

症状：

- `uv pip install -e .` 可能尝试解析不存在的 PyPI 包 `tkinter`
- 或者安装依赖时出现和 GUI 相关的依赖错误

原因：

- `tkinter` 是 Python 标准库模块，不应该写在 `pyproject.toml` 的 `dependencies` 里。

解决：

- 删除 `tkinter` 依赖声明。
- 使用带 Tk 支持的 Python。当前 uv 管理的 Python 可以正常 `import tkinter`。

验证：

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
.venv/bin/python -c "import tkinter; print('tk ok')"
```

### 3. editable install 包路径不完整

症状：

- 运行时出现 `ModuleNotFoundError`
- 安装后 console script 或 import 找不到子包

原因：

- `generation_two` 是仓库根目录下的包，但 `pyproject.toml` 位于 `generation_two/` 内部。
- 原始 setuptools 配置只声明了顶层包，没有完整包含 `core/gui/storage/ollama/...` 子包。

解决：

- 在 `pyproject.toml` 中设置：

```toml
[tool.setuptools]
package-dir = {"" = ".."}
```

- 并显式列出 `generation_two` 的子包。
- `setup.py` 简化为读取 `pyproject.toml` 元数据。

### 4. 从 .app 启动和从项目目录启动表现不同

症状：

- 命令行能启动，双击 `.app` 失败
- `.app` 找不到 `.venv`
- `.app` 找不到 `credential.txt`

原因：

- `.app` 的工作目录和 shell 环境不同。
- Dock 启动时不会继承你终端里的 PATH、venv 激活状态或当前目录。

解决：

- 在 `Generation Two.app/Contents/MacOS/generation-two-launcher` 里写绝对路径。
- 固定 `PROJECT_DIR`。
- 固定使用 `$PROJECT_DIR/.venv/bin/python`。
- 固定把日志写到 `$PROJECT_DIR/generation_two_gui.log`。

### 5. PID 文件不准确，无法判断 GUI 是否还在运行

症状：

- `generation_two_gui.pid` 不可靠
- 无法确认 GUI 是否还活着
- 关闭实例时可能找错进程

解决：

- `gui/run_gui.py` 启动时写入真实 `os.getpid()`。
- 进程正常退出时，如果 PID 文件仍指向自己，就清空为 `0`。

检查：

```bash
cat /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.pid
ps -p "$(cat /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.pid)" -o pid,etime,command
```

### 6. conda 上没有遇到的问题，uv/venv 上遇到了

现象解释：

- conda 环境通常自带较完整的 Python、Tk、动态库和二进制依赖。
- uv 创建的是更干净的项目虚拟环境，缺什么会更直接暴露出来。
- 这不是坏事：项目可复现性更好，但需要把依赖和 Python 版本声明清楚。

解决思路：

- 用 `.python-version` 固定 Python 版本。
- 用 `pyproject.toml` 固定 Python 包依赖。
- 不依赖 conda base 环境里的隐式包。

### 7. Data fields 加载看起来卡住

症状：

- GUI 显示黄色 loading
- 日志里反复看到某些 region 无字段

原因：

- USA 字段通常能加载。
- EUR/CHN/ASI/GLB/IND 有时会因为 region/universe/delay 组合拿不到字段，日志里出现 `No fields found` 或 `Failed to fetch fields for any universe`。
- UI 状态曾经可能没有及时从 loading 更新到完成。

排查：

```bash
tail -n 200 /Users/volcano/code/worldquant-miner/generation_two/generation_two_gui.log
```

建议：

- 初次配置时先只用 USA/TOP3000 验证主流程。
- 其他 region 后续再逐个调 universe 和字段源。

### 8. Simulation 卡在 RUNNING 或出现 429

症状：

- 日志里持续：

```text
Progress-only simulation response: progress=0.35, treating as RUNNING
```

- 或 POST `/simulations` 返回 `429`。

解释：

- 持续 `RUNNING` 不一定是卡死，平台有时只返回 `progress`。
- `429 CONCURRENT_SIMULATION_LIMIT_EXCEEDED` 表示平台并发 simulation 限制。

解决：

- 程序现在有并发 semaphore 和 429 retry。
- 不要手动开太多 GUI 实例。
- 如果长期没有结果，检查日志尾部和 DB 最新 timestamp。

### 9. IS check 解析曾经误判

症状：

- 明明 WorldQuant check 失败，本地却可能当作 success。
- PASS/PENDING/FAIL 解析不稳定。

解决：

- 现在优先解析 check 的 `result/status/severity/type`。
- `FAIL/FAILED/ERROR/RED/REJECTED/INVALID` 会阻断 success。
- `PASS` 不再被误加入错误 tag。
- `PENDING` 会记录 pending 类型 tag。

### 10. LOW_SHARPE/LOW_FITNESS 反复触发 LLM refeed

症状：

- 因子表达式语法没坏，只是表现不好，却触发 LLM 修语法。
- 消耗 LLM 调用和 simulation 预算。

解决：

- refeed 现在只处理结构性错误，例如 syntax/parse/compiler、unknown field/operator、placeholder、输入数量错误。
- `LOW_SHARPE`、`LOW_FITNESS`、`CONCENTRATED_WEIGHT`、`LOW_SUB_UNIVERSE_SHARPE` 这类表现问题不会触发语法 refeed。
- 有潜力的表现问题交给 improve 层处理。

### 11. 默认 truncation 过高

症状：

- 默认 `truncation=0.08` 可能过于宽松，容易掩盖集中度问题。

解决：

- 默认改为 `0.05`。
- robust sweep 会在 promising alpha 上测试一小组 truncation/decay/neutralization 组合。

### 12. 本地无法复刻 WorldQuant 回测

结论：

- 当前程序能拉取 fields/operators 元数据、提交 simulations、读取 alpha 指标和 checks。
- API 没有把完整 in-sample 原始数据矩阵下载到本地。
- 本地可以做静态验证、去重、字段/算子质量筛选、失败模式预测。
- 最终 Sharpe/Fitness/Checks 仍然要通过 WorldQuant 平台 simulation 确认。

## 新设备验收清单

按顺序确认：

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
.venv/bin/python --version
.venv/bin/python -c "import requests, numpy, yaml; import tkinter; print('imports ok')"
.venv/bin/python -m py_compile gui/run_gui.py core/simulator_tester.py storage/backtest_storage.py
```

启动 GUI：

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
.venv/bin/python -u gui/run_gui.py ../credential.txt
```

看到以下信息基本说明环境是通的：

- `Authentication successful`
- `Session verified`
- `tkinter` import 成功
- GUI 窗口出现
- Step 1 能加载 operators
- USA data fields 能 fetch 或从缓存加载

## 推荐的新机器配置顺序

1. 安装 Xcode Command Line Tools。
2. 安装 Homebrew。
3. 安装 `uv`。
4. clone 仓库。
5. 在 `generation_two` 下安装 Python 3.11。
6. 创建 `.venv`。
7. `uv pip install -e .`。
8. 创建根目录 `credential.txt`，配置 WorldQuant 和 LLM API。
9. 用命令行启动一次 GUI，确认日志无异常。
10. 再配置或复制 `Generation Two.app` 到 Dock。
11. 先用 USA/TOP3000 跑通 fields、generation、simulation。
12. 再开启 mining/evolution/improve 等长流程。

## 最小可用命令汇总

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e .
.venv/bin/python -c "import requests, numpy, yaml; import tkinter; print('ok')"
.venv/bin/python -u gui/run_gui.py ../credential.txt
```

如果使用仓库内 uv 缓存：

```bash
cd /Users/volcano/code/worldquant-miner/generation_two
UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv python install 3.11

UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv venv --python 3.11 .venv

UV_CACHE_DIR=/Users/volcano/code/worldquant-miner/.uv-cache \
UV_PYTHON_INSTALL_DIR=/Users/volcano/code/worldquant-miner/.uv-python \
uv pip install -e .
```
