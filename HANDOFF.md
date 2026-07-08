# 项目移交指南 / Handoff Guide

> 本文档面向**接手本 fork 的新同学**。上游 KernelAgent 假设"本机就有 GPU + PyTorch"，
> 而本 fork 做了大量改造，使其能在**一台没有 GPU/PyTorch 的开发机**上运行：代码生成走
> LLM，所有需要 GPU 的验证 / benchmark / NCU 都通过 SSH 提交到**远程 H100 devspace** 执行。
> 读完本文你应该能在 ~30 分钟内从零跑通一个 conv 融合算子的生成 + 优化，并在网页上看到迭代轨迹。

---

## 1. 这个 fork 相比上游改了什么（先有个整体印象）

| 能力 | 说明 | 关键文件 |
| --- | --- | --- |
| **远程 GPU 执行** | 本机无需 GPU/torch；验证、benchmark、NCU profiling 全部 rsync 到远程 H100 通过 SSH 跑，产物回传 | `utils/remote_config.py`, `utils/remote_exec.py` |
| **GLM-5.2 模型接入** | 通过 llm-center（OpenAI 协议）调用 GLM-5.2 推理模型，捕获 reasoning/finish_reason | `utils/providers/llm_center_provider.py`, `utils/providers/openai_base.py` |
| **流式请求（关键）** | llm-center 的 GLM 端点非流式极慢（同一请求非流式 90min vs 流式 2min），已强制 `stream=True` | `utils/providers/openai_base.py` (`_aggregate_stream`) |
| **参数绑定 SSOT** | conv/linear 等带权重算子，验证与 benchmark 用同一套"按位置绑定"逻辑，避免签名错位 | `.../benchmarking/kernel_binding.py`, `timing.py` |
| **轨迹收集 + 网页面板** | 每轮优化的时间/瓶颈/思考(reasoning)/失败都落盘，用 FastAPI 面板观测 | `.../searching/trajectory.py`, `scripts/optimization_dashboard.py` |
| **续跑 resume** | 从已有 run 的 `program_db.json` 接着往下跑更多轮，显示为同一版本的延续 | `triton_kernel_agent/opt_manager.py`, `examples/run_opt_manager.py --resume-from` |
| **端到端 pipeline 脚本** | agent1 重试直到生成正确 kernel → agent2 N 轮优化 | `run_gemm_pipeline.py`, `run_conv_pipeline.py`, `run_conv_l1_pipeline.py` |
| **心跳保活** | 防止远程 devspace 空闲被回收；掉线会告警 | `scripts/devspace_heartbeat.py`（cctl 版）或 SSH 直连版（见下） |

**两个 agent 的分工**（理解流程的关键）：
- **agent1（生成，KernelFalcon）**：把问题描述 → 一个通过正确性验证的 Triton kernel。
- **agent2（优化，KernelAgent）**：拿 agent1 的 kernel → 用 NCU roofline 分析指导，N 轮迭代提速。

---

## 2. 一次性准备

### 2.1 本机安装

```bash
cd KernelAgent
pip install -e .
# 本机不需要 GPU/torch/triton —— 那些只在远程用
```

### 2.2 申请一台远程 GPU devspace

所有 GPU 工作在远程 H100 上跑。你需要一台 **Cybertron devspace**：

1. 在 Cybertron 平台申请一台 **H100 devspace**。**优先选 `loopharness` 镜像**——它自带
   `rsync` 和 `ncu`（Nsight Compute），开箱即用。若拿到的镜像缺这两个工具，见 §5 排查。
2. 记下 devspace 的编号（例如 `528797`）。它的完整 Teleport 节点名形如
   `devspace-lizhen-loopharness-528797`。
3. 确认你本机装了 `tsh`（Teleport CLI）且已 `tsh login`（有效登录态可用
   `tsh status` 查看）。本 fork 里 tsh 路径示例：
   `/Users/<you>/Library/Application Support/Cursor/User/globalStorage/yangsuiyun.cybertron/bin/tsh`

### 2.3 配置 SSH（让 `ssh <节点名>` 直接能连）

在 `~/.ssh/config` 追加一段（把 `528797` 换成你的编号，`loopharness` 换成实际镜像名，
`lizhen` / tsh 路径换成你自己的）：

```
# 🏷️  POD Node: devspace-lizhen-loopharness-528797
Host devspace-lizhen-loopharness-528797
    HostName devspace-lizhen-loopharness-528797
    User root
    Port 3022
    ProxyCommand "/path/to/tsh" proxy ssh --cluster=teleport.cybertron.modelbest.co --proxy=teleport.cybertron.modelbest.co:443 %r@%h:%p
    IdentityFile "/Users/<you>/.tsh/keys/teleport.cybertron.modelbest.co/lizhen"
    CertificateFile "/Users/<you>/.tsh/keys/teleport.cybertron.modelbest.co/lizhen-ssh/teleport.cybertron.modelbest.co-cert.pub"
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
```

验证连通 + 环境齐全（应看到 GPU 名、torch/triton 版本、rsync、ncu 都在）：

```bash
ssh devspace-lizhen-loopharness-528797 \
  'nvidia-smi --query-gpu=name --format=csv,noheader | head -1; \
   python3 -c "import torch,triton;print(torch.__version__,triton.__version__,torch.cuda.is_available())"; \
   which rsync ncu'
```

### 2.4 告诉 KernelAgent 用哪台机器：`remote.toml`

仓库根目录的 `remote.toml`（**已 gitignore，不会提交**）：

```toml
[remote]
kind = "ssh"                                    # local = 本地跑；ssh = 远程跑
hostname = "devspace-lizhen-loopharness-528797" # 换成你的节点名
workspace = ""                                  # 留空即用远程 $HOME/.kernelagent_remote
```

> 也可用环境变量覆盖：`KERNEL_REMOTE_KIND=ssh KERNEL_REMOTE_HOST=<节点名>`。

### 2.5 配置 LLM（GLM-5.2）

本 fork 默认用 llm-center 上的 GLM-5.2。设置 key（放进 gitignore 的 `.env.local` 或直接 export）：

```bash
export LLM_CENTER_API_KEY=sk-xxxxxxxx     # 向团队索取；不要提交到仓库
```

自检模型可用（应在几秒内返回 `OK`）：

```bash
python3 -c "from utils.providers import get_model_provider as g; \
p=g('glm-5.2'); print(p.get_response('glm-5.2',[{'role':'user','content':'Reply with exactly: OK'}],max_tokens=2000).content)"
```

---

## 3. 开启心跳（强烈建议）

远程 devspace **空闲会被回收**，且跑一个任务可能要一两个小时，中途掉线整个 run 就废了。
跑任务前先开心跳保活。最稳的是 **SSH 直连心跳**（不依赖 cctl 的可见范围）：

```bash
H=devspace-lizhen-loopharness-528797
nohup bash -c '
while true; do
  out=$(timeout 45 ssh -o ConnectTimeout=35 -o BatchMode=yes "'"$H"'" \
        "echo HB_OK_$(date +%s); : > \$HOME/.devspace_keepalive" 2>/dev/null)
  if [ $? -eq 0 ] && echo "$out" | grep -q HB_OK_; then
    echo "[$(date +%FT%T)] '"$H"' OK ($out)"
  else
    echo "[$(date +%FT%T)] '"$H"' FAIL"     # 掉线会在这里如实报 FAIL
  fi
  sleep 240
done' > /tmp/hb.log 2>&1 &
tail -f /tmp/hb.log     # 观察，Ctrl-C 退出观察（后台继续跑）
```

> 若你的 cctl token 能看到自己的 devspace，也可用 `python3 scripts/devspace_heartbeat.py start`
> （cctl API 版）。注意：cctl 只能看到 token 可见范围内的 devspace，看不到就用上面的 SSH 版。

---

## 4. 跑任务

### 4.1 一键 pipeline（agent1 生成 → agent2 优化 10 轮）

```bash
# 融合 conv（Conv2d + ReLU + per-channel bias，KernelBench level2 风格）
LLM_CENTER_API_KEY=$LLM_CENTER_API_KEY \
KERNEL_WORKER_TIMEOUT_S=3600 LLM_TIMEOUT_S=3600 \
python3 -u run_conv_pipeline.py 2>&1 | tee /tmp/conv_run.log
```

其它现成入口：`run_gemm_pipeline.py`（方阵 GEMM）、`run_conv_l1_pipeline.py`（level1 纯 Conv2d）。

**跑起来后会发生什么**（对照日志看）：
1. `Generating N kernel seeds` → agent1 让 GLM 生成 N 个候选（GLM 有长推理，conv 约 2–7 分钟）。
2. `Verification passed` → 远程验证通过，写 `examples/optimize_conv/input.py`。
3. `PyTorch baseline: X ms` → 远程测出 PyTorch 参考时间。
4. `Round k: ...` → agent2 逐轮优化；成功轮会更新最优 kernel。
5. `GREEDY_GLM OPTIMIZATION SUCCESSFUL! Speedup vs initial ...` → 收尾。

产物落在 `examples/optimize_conv/runs/v1_<时间戳>/`（该目录 gitignore）。

### 4.2 直接跑 agent2（已有 kernel 时）

```bash
python3 examples/run_opt_manager.py \
  --kernel-dir examples/optimize_conv \
  --strategy greedy_glm \
  --max-rounds 10
```
（`examples/optimize_conv/` 需含 `problem.py` + `test.py`；`input.py` 由 agent1 生成。）

### 4.3 续跑 resume（从已有 run 接着优化更多轮）

```bash
python3 examples/run_opt_manager.py \
  --kernel-dir examples/optimize_conv \
  --strategy greedy_glm \
  --max-rounds 20 \
  --resume-from examples/optimize_conv/runs/v1_<时间戳>
```
它会读旧 run 的 `program_db.json`，从历史最优 kernel 继续，**轮次编号接着往下**，
显示为同一版本的延续（不是新建 baseline）。

---

## 5. 观测：优化轨迹网页面板

```bash
python3 scripts/optimization_dashboard.py --root . --port 8086
# 浏览器打开 http://localhost:8086
```

面板会自动发现所有 run，展示每一轮：本轮耗时、历史最优、瓶颈类别、**模型的思考(reasoning)**、
以及失败轮的失败原因——即"完整轨迹，包含失败尝试"。

---

## 6. 常见问题排查

| 现象 | 原因 & 处理 |
| --- | --- |
| `failed connecting to host ...:3022` / `not found in inventory` | **devspace 掉线/被回收**。用 `tsh ls` 确认节点是否还在；重新申请一台，更新 `~/.ssh/config` 和 `remote.toml`。心跳日志出现 `FAIL` 就是这个。 |
| `which rsync` 为空 | 镜像缺 rsync。可从静态构建推一个上去：本机 `curl -L -o /tmp/rsync https://github.com/jbruechert/rsync-static/releases/download/continuous/rsync-x86` 然后 `cat /tmp/rsync \| ssh <节点> 'cat >/usr/local/bin/rsync && chmod +x /usr/local/bin/rsync'`。 |
| `which ncu` 为空 | 镜像缺 Nsight Compute。agent2 每轮需要 NCU 分析，缺了会 `No analysis available, skipping round`。**优先换 `loopharness` 镜像**（自带 ncu）。 |
| agent1 一个请求跑了 1 小时还不返回 | 确认走的是**流式**路径（`_create_with_hard_timeout` 里 `stream=True`）。非流式在 llm-center 上会病态慢。 |
| `No available provider for model 'glm-5.2'` | `LLM_CENTER_API_KEY` 没设或没传进子进程。用 `export` 或 `.env.local`，别只写在命令行局部。 |
| 前端看不到新任务 | agent1 还没生成完，agent2 未启动就没有 trajectory；等验证通过后再看。确认 dashboard `--root` 指向仓库根。 |
| 验证"通过"了但 benchmark 崩 | 曾因 `setsid` 吞远程退出码导致假通过，已修（`setsid --wait`）。若自定义远程命令，务必保留 `--wait`。 |

---

## 7. 关键代码地图（要改东西时看这里）

- **远程执行**：`utils/remote_config.py`（配置）、`utils/remote_exec.py`（rsync 推送 / ssh 运行 / 产物回传）
- **LLM provider**：`utils/providers/openai_base.py`（流式 + 硬超时看门狗）、`utils/providers/llm_center_provider.py`
- **agent1 生成**：`triton_kernel_agent/agent.py`、`triton_kernel_agent/worker.py`
- **agent2 优化**：`triton_kernel_agent/opt_manager.py`、`.../orchestrator/optimization_orchestrator.py`
- **参数绑定 SSOT**：`.../benchmarking/kernel_binding.py`、`.../benchmarking/timing.py`（`bind_kernel_function`）
- **NCU / roofline**：`kernel_perf_agent/kernel_opt/profiler/`、`.../roofline/`
- **轨迹 / 面板 / 续跑**：`.../searching/trajectory.py`、`scripts/optimization_dashboard.py`、`examples/run_opt_manager.py`
- **测试**：`tests/`（`pytest tests/ -o addopts=""` 跑全套；本机无 torch 也能跑大部分）

---

## 8. 快速自检清单（移交后第一天照着走）

- [ ] `pip install -e .` 成功
- [ ] 有一台 H100 devspace，`ssh <节点>` 能连，`rsync`/`ncu` 都在
- [ ] `remote.toml` 指向该节点
- [ ] `LLM_CENTER_API_KEY` 已设，GLM 自检返回 `OK`
- [ ] 心跳已开，`/tmp/hb.log` 持续 `OK`
- [ ] `run_conv_pipeline.py` 跑出 `OPTIMIZATION SUCCESSFUL` 和 speedup
- [ ] 面板 `http://localhost:8086` 能看到该 run 的逐轮轨迹
