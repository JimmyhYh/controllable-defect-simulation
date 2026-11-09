# 少样本真实工业缺陷：运行说明

当前状态：本轮一周实验／报告已完成并保持冻结；后续SSN核验与计划已完成但未运行，当前下一项以[SSN新计划](D:/华兴源创/计划/SuperSimpleNet严格R5基线执行计划.md)第六节为准。

第1天CPU输入与第2天配对准备已完成并核验：7500有效合成、3套共同序列及12张probe齐备，尚未训练。显式人工训练／CPU指标入口已实现；模型运行、GPU恢复与完整精确AP尚待实测。见[第2天最终准备](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/第2天配对输入与人工训练准备.md)。

第1天已完成并核验，`input_ready=true`。固定R5为11670／11884／12070／12245／12303；全部3335规范RGB身份无精确重复，test GT未解码。见[最终输入核验与适用性局限](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/input/第1天输入准备与适用性核验.md)。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-data
```

`verify-data`读取原包、封存审计、规范文件及科学产物核对hash，约需数十秒至数分钟，取决于磁盘；不重新解码RGB／GT、重抽样、生成或模型运行。`input_ready`以封存`completion.json`中的视觉审阅为准，准备配置里的false保留“当时待视觉审阅”的历史含义。

首次准备入口（已经运行，不应为改变样本重复运行）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli prepare-data
```

空目录才做首次准备；完整同身份缓存只走核验，不重新导出、RGB解码或抽样。仅有transaction而无preparation_complete、未知文件／目录、hash或代码变化时保留并停止，不删除后重跑。`preparation_transaction.json`在完成后保留，是原始事务身份而不是完成标记；完成以preparation_complete和最终completion为准。

准备使用[已核验原包](D:/华兴源创/实验/KolektorSDD2.zip)及[既有审计](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_input_audit_v1/archive_verification.json)，只导出3335规范图像／GT对，原copy和pickle仅保留在原包。全部RGB解码仅算带尺寸的身份，不显示test或统计其外观；test_GT不解码。视觉范围为固定R5及N16，标注／输入适用性不代表真实生成或检测收益。

数据及split：`D:/华兴源创/实验/弱缺陷Demo/data/fewshot_synthetic/ksdd2_v1/`。

科学结果及训练侧总览：`D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/input/`。

全部训练继续由用户手动启动；下一步以[一周执行计划](D:/华兴源创/计划/少样本真实工业缺陷合成训练一周执行计划.md)第六节为准。

## 第2天配对准备与核验

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli prepare-pairs
```

此命令只使用R5／N1885，CPU准备3套10000步共同序列、7500份合成和12份probe。启动需可用RAM≥2GiB、磁盘≥80GiB；完整同身份缓存只核验。它不会导入Torch。中途资源不足、五次几何尝试失败、未知／部分目录或身份变化时保留并停止，不能删目录重跑绕过，也不能因失败替换背景、幅度或种子。

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-pairs
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli training-status --condition probe --seed 2026101906
```

`verify-pairs`核对已有完整输入，不重新生成；原始训练RGB／GT在实际使用时另核对其第1天文件hash。第2天不调用旧`verify-data`，避免读取Q／E／U原始图像字节；来源只核对封存元信息、原zip整体身份及允许的训练材料。Q／E／U内容均不得用于准备。

## 人工训练入口（仅输入关口通过后由用户启动）

技术probe：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli train-probe --seed 2026101906
```

这条命令会更新参数：独立监督分割网，5真异常＋3合成＋4正常共12张，1000步。lr为1—800步1e-4、801—900步2e-5、901—1000步4e-6。需启动时可用RAM≥3GiB、空闲显存≥6GiB；每100步RAM采样，低于3GiB停止。FP32／batch1／workers0，严格确定性算子不支持、非有限或OOM就停止，不自动换实现。**本模型耗时、峰值和GPU完整恢复尚未实测**，不能沿用旧DRAEM运行时间作承诺。

自有目录：`models/fewshot_synthetic/ksdd2_v1/probe_2026101906`与`results/fewshot_synthetic/ksdd2_v1/probe_2026101906`。每200步存完整检查点。Ctrl+C不额外保存；同一人工命令只允许恢复身份一致的自有完整保存点，未保存步数会丢失。`.tmp`、有pt无对应json、未知文件或不明确最终预测事务须查因，不自动续或重复前向。CPU核验只读json／hash，不加载检查点。最终12张预测只做一次，完成后不会自动计算AP或启动正式组。

每个运行在模型初始化前独占创建`active.json`；另一个命令及遗留标记均停止。RAM不足、OOM、非有限或其他错误另保存`failure.json`并保留所有现场，不能用同命令直接绕过；先诊断。只有明确人工中断且自有完整保存点清单正常时才释放自己的运行标记。正式训练门槛会完整核对probe检查点、最终结果、封存精确AP与gate身份，而不只读取一个通过布尔值。

资源字段的当前局限：`elapsed_update_seconds`包含最终probe的12张预测，不能直接当纯更新耗时；`peak_allocated_MiB`／`peak_reserved_MiB`只代表最终进程会话，恢复前会话峰值尚未聚合。第3天预算说明必须披露此口径，不把它们称全程纯训练耗时或跨恢复完整峰值。完整恢复与这些资源口径仍待人工运行核验。

用户完成后，只读核验和独立CPU精确指标入口分别是：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-probe --seed 2026101906
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli probe-metrics
```

`probe-metrics`不导入Torch；精确float32位序直方图约需7.94GiB磁盘，两张表按完整同分值ties计算AP，另与全12张sklearn核对至1e-10。准入RAM≥3GiB、磁盘≥80GiB，不抽像素；实际磁盘速度尚未实测。最终AP≥0.80与资源／有限性通过才允许正式训练。这只是训练侧拟合门槛。

正式入口已编写，但当前未获准运行；需第3天probe及预算审阅通过，再依次手动A／B、seed2026101903／1904／1905。例：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli train-formal --condition A --seed 2026101903
```

每组从对应seed重新初始化，无probe或旧模型权重；各10000步，完成8000／9000步后各乘0.2，每1000步保存自有完整检查点。同seed两组同初始化／真实曝光／背景序列，仅B的2500干预槽使用合成。正式入口不会自行开始下一组、Q校准或E评价。校准／E模型评分入口本日未接入，不提供这些命令。
