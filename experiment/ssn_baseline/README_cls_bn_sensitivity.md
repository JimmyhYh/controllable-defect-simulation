# 分类卷积BN统计来源敏感性诊断：运行说明

本目录新增三个文件：`cls_bn_sensitivity.py`（唯一新计算模块）、`cls_bn_sensitivity_cli.py`（薄CLI）、本说明。
科学边界以 `计划/SSN分类BN统计来源敏感性验证方案.md` 为准；此处只列**实际实现并核验过**的入口。

## 这是什么

把 `model.discriminator.dec_head[1]`（`nn.BatchNorm2d(128)`）的归一化统计来源，由保存的 running 统计
临时换为该图当前统计，在固定21张训练图上做一次反事实前向，观察分类 logit 与 `sigmoid>0.5` 计数的变化。

- 不是修复，不是新配置，不让原 `gate=false` 转绿。
- 不改 `seg` BN、骨干 BN、损失、LR、步数、标签、阈值、输入预处理。
- 不写任何缓冲：`state_dict` 前后 hash 必须等于 `1513a3df…c3337`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `results/ssn_baseline/ksdd2_r5_v1/pilot_cls_bn_sensitivity_preparation_v1/frozen_v2/` | 冻结协议 + 静态核验，已seal |
| `results/ssn_baseline/ksdd2_r5_v1/pilot_cls_bn_sensitivity_preparation_v1/stage1b/` | 第1b语义核验记录，已单独seal |
| `results/ssn_baseline/ksdd2_r5_v1/pilot_cls_bn_sensitivity_v1/` | 运行产物：transaction／scores／control／identity／complete |
| `results/ssn_baseline/ksdd2_r5_v1/pilot_cls_bn_sensitivity_summary_v1/` | 21行汇总，已seal |

`frozen_v1/` 保留为历史：它把1b标记规划进同一个seal目录，而 `r.write` 为独占写、seal不可追加，
该布局在1b时无法落盘。因此在任何运行与结果产生之前重建为 `frozen_v2/`，并把1b标记移出冻结seal范围。

运行目录故意与准备目录分开：方案第六节规定"目标目录已存在且无 complete.json 即停止"，草案不能占用运行目录。

## 命令（PowerShell）

准备与只读核验用旧CPU环境 `.venv`（不导入Torch）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m ssn_baseline.cls_bn_sensitivity_cli prepare
& '.\.venv\Scripts\python.exe' -m ssn_baseline.cls_bn_sensitivity_cli verify-preparation
```

运行必须用隔离环境 `.venv-ssn`（`resource_admission` 强制校验解释器路径），且**需要第1b授权标记**：

```powershell
& '.\.venv-ssn\Scripts\python.exe' -m ssn_baseline.cls_bn_sensitivity_cli run
```

运行后只用CPU核验与汇总：

```powershell
& '.\.venv\Scripts\python.exe' -m ssn_baseline.cls_bn_sensitivity_cli verify
& '.\.venv\Scripts\python.exe' -m ssn_baseline.cls_bn_sensitivity_cli summarize
```

`prepare` 若目录已冻结，只做seal核验并返回，不重写；写入用独占模式（`x`），不存在静默覆盖。

## `run` 的前置与拒绝条件

1. 冻结协议 seal 与模块源码 hash 一致（`verify-preparation` 通过）。
2. `stage1b/` 必须存在且seal有效、`verification.json` 的 `passed=true`；否则**在导入Torch之前**直接拒绝，不建运行目录。
3. 资源：起点可用RAM≥6GiB、空闲显存≥6GiB、磁盘≥80GiB；每图前采样可用RAM≥3GiB。
4. 21张输入各一次，顺序取自基线 `scores.json`，不重排、不重抽、不增删。
5. 逐图四项断言（方案3.4）：生效计数递增至21、μ与σ²相对running统计的差异**有限且>1e-6**、`training`仍为False且走原生路径、`Δ=0`单列。任一失败立即停止并写 `failure.json`。
6. 分割控制：新 `feature_probability` 必须与基线npz逐图 `allclose(atol=1e-6, rtol=1e-5)`；否则该次运行作废。

## 输出

| 文件 | 内容 |
| --- | --- |
| `transaction.json` | 导入Torch前写入，含协议hash与资源快照 |
| `scores/{id}.npz` | 新 logit、新特征概率、序号、两项统计差异 |
| `scores.json` | 21行：新旧 logit／score、Δ、决策标记 |
| `control.json` | 逐图分割一致性（`max_abs_diff`、`allclose`） |
| `identity.json` | 权重／状态hash、四类计数（探测1、输入前向21、生效21、恢复1）、最小RAM |
| `complete.json` | seal，运行结束才生成 |
| `failure.json` | 仅失败时出现；保留部分前向，不自动重试 |

## 不做什么

不训练、不反向、不生成、不校准、不搜索阈值、不算AP/AUROC、不选中间权重、不读Q/E/U或训练GT。
即使本诊断把FP降到≤1/16，也只作为训练同池现象记录，原 `gate=false` 不变。
