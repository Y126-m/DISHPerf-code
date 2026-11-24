# Paper-aligned DISHPerf — 2026-11-17

独立目录实现，依据paper (11-16).tex。请先读《论文实现对应说明.md》。保留原TF1方法和原始记录协议，修正数据泄漏、评分公式及完整输入连接。未运行正式120候选实验。

## 本地验证

PowerShell中使用已存在的原生环境；PATH仅对当前终端生效：

```powershell
$env:PATH = 'D:\anaconda\envs\moxiangev;D:\anaconda\envs\moxiangev\Library\bin;D:\anaconda\envs\moxiangev\Scripts;' + $env:PATH
$env:PYTHONIOENCODING = 'utf-8'
Set-Location '.\experiments\hinnperf\dishperf'
& 'D:\anaconda\envs\moxiangev\python.exe' verify.py --data-root '..\hinnperf\datasets'
```

verify只做极短训练和泄漏/图结构检查，不产生论文可用正式结果。

## 查看计划

```powershell
& 'D:\anaconda\envs\moxiangev\python.exe' run.py --stage plan --data-root '..\hinnperf\datasets' --system x264 --start-rep 0 --repeats 10
```

plan只验证源数据/代码身份并打印任务，不训练。固定种子公式中的注册重复数始终为30。10次子集仍与作者30次方案的前10次对应。

## 正式执行入口

先查看计划并确认本版本与论文的对应说明，再运行：

```powershell
& 'D:\anaconda\envs\moxiangev\python.exe' run.py --stage run --data-root '..\hinnperf\datasets' --system x264 --start-rep 0 --repeats 10
& 'D:\anaconda\envs\moxiangev\python.exe' summarize.py
```

该入口按rep轮转预算，先r0的4档再r1；不是每档全部30次后才换档。正式任务每个120候选；已完成任务校验后跳过；中断后的已完成候选缓存可以复用，但当前候选和最终重训可能重算。任务锁防止重复进程；硬终止遗留锁时应核对PID和实际进程后再处理，不能一律删除活跃锁。仅在同一source_manifest身份下继续缓存。

没有内置共享队列或自动资源调整；此目录不覆盖旧服务器调度器，也不会自动启动大批训练。服务器使用对应TF1.10环境和原始CSV路径运行相同入口，需先核对实际部署路径。

## 输出

- `results/formal/<system>_b<budget>_r<rep>/`：原始划分、fit-only筛选审计、候选记录、最终配置、重训记录、逐行预测、metrics、SHA256完成标记。
- `results/verification_only/`：2步singleton检查；summarize严格排除，不可混入论文。
- `results/summary_all_completed.csv`：全部正式完成重复，均值、SD、Margin、实际n；无截尾筛选。

数据不随包重复复制；data_manifest锁定10份原始CSV哈希。实现校验只锁源代码和数据清单，不锁README、执行结果或说明文件；代码改动须另立版本并重新评估，不能手工刷新哈希伪装原版本。
