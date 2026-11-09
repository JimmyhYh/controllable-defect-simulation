# 固定12张技术probe：独立资源复跑

当前状态：本轮一周任务已完成，图件及[独立研究报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练研究报告.md)齐备。平均AP增益为正但不稳定，全部冻结成果及历史失败保留；下一项仅审阅独立曝光匹配提案，以一周计划第六节为准。
本轮只改变输出位置及启动可用RAM余量；同12张、seed2026101906、GN模型、损失、初始化、学习率日程、1000步与最终AP0.80要求不变。原100步RAM失败的run／active／failure全部保留；没有保存点可以恢复，独立尝试从头初始化。

启动时可用RAM至少4.5GiB、空闲显存至少6GiB，运行中每100步仍要求RAM≥3GiB。4.5GiB是资源候选，不能保证全程足够；若准备／验证期间可用RAM下降，入口会在初始化前停止，不自动降低条件。助手不启动训练。

## 用户手动运行

只有复跑准备完整核验后才运行以下已实现入口：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli train-probe-retry --seed 2026101906
```

这条命令会更新参数。1000步，1—800步lr1e-4、801—900步2e-5、901—1000步4e-6；每200步自有完整检查点。具体耗时未知，不把原100步失败或旧模型耗时当预算结论。Ctrl+C不额外保存；只在自有完整检查点和事务明确时允许同一人工命令恢复。错误、未知锁、未完成权重保存或最终预测事务保留并查因，不自动重试。新尝试仍适用原停止与有限性检查，不加载旧probe权重。

新模型目录：`D:/华兴源创/实验/弱缺陷Demo/models/fewshot_synthetic/ksdd2_v1/retry_ram_v1/probe_2026101906/`。

新结果目录：`D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/retry_ram_v1/probe_2026101906/`。

独立协议：`D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/probe_retry_preparation_v1/`。结果父目录binding.json绑定该协议、适配器与旧核心实现，不向旧准备目录或原失败目录追加文件。

## 只读入口

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli probe-retry-status
```

完成后可核对完整步数、检查点、LR与最终输出身份：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-probe-retry
```

状态与核验均不导入Torch或进行模型前向；模型文件只读hash，不反序列化。`prepare-probe-retry`只做CPU元信息准备，完整缓存复用不生成或读图。

复跑结束不自动计算AP或开始正式组。原`probe-metrics`指向原失败目录，不能用于本次新结果；本次CPU指标路由待完成后另行准备／授权，不能用旧通过标记替代。最终12张预测的704256有效像素／26414阳性分母不变。GPU完整恢复、峰值和1000步拟合仍待实测，3对正式训练当前0/6。

适配器只在当前人工进程临时路由输出、提高启动RAM检查、核对初始化hash，退出后恢复原函数引用；原common／sampling／model／training／metrics／inference六份源码与已冻结输入不改。原失败目录及其PID状态不明确、身份不同、或新目录出现未知产物时停止查因。
