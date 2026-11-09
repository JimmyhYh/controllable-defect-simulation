# 新gate准入下的正式人工训练

当前状态：本轮一周任务已完成，图件及[独立研究报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练研究报告.md)齐备。平均AP增益为正但不稳定，全部冻结成果及历史失败保留；下一项仅审阅独立曝光匹配提案，以一周计划第六节为准。
仅改变准入来源和输出位置，复用冻结GN监督分割、CE＋Dice、Adam、FP32／batch1／workers0及全部训练数据／日程。独立probe训练侧AP99.76363089%与交叉核验通过只是技术准入，不是合成收益。probe权重不作正式初始化。

顺序固定：2026101903的A、B，2026101904的A、B，2026101905的A、B。各10000步，从各自seed重新初始化；同seed两组共享背景／真实图窗／翻转序列，B初始化hash必须与A相同。A真实5000＋正常5000；B真实5000＋正常2500＋合成2500，四类合成各625。三套7500份合成只核验复用，不重新生成。

## 首组A，由用户手动启动

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli train-admitted-formal --condition A --seed 2026101903
```

这是会更新参数的入口，助手不会启动。启动时RAM≥4.5GiB、空闲显存≥6GiB、磁盘≥80GiB；运行中每100步采样RAM，低于原3GiB条件即停止。4.5GiB只是更保守的启动余量候选，不是充分性保证；probe成功时实际起点6.92GiB。全6组同资源要求，不自动关闭程序或降低预算。

更新1—8000用1e-4、8001—9000用2e-5、9001—10000用4e-6，每1000步保存自己的完整参数／优化器／全部RNG／序列游标／LR历史／输入与实现身份。Ctrl+C不额外保存；只有自己的完整保存点与明确事务允许同命令人工恢复，未保存进度丢失。GPU完整恢复仍未实测。OOM、非有限、RAM不足、未知锁、保存残片或失败标记保留并停止，不能删后重训绕过。

新模型父目录：`D:/华兴源创/实验/弱缺陷Demo/models/fewshot_synthetic/ksdd2_v1/formal_gate_v1/`。

新结果父目录：`D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/formal_gate_v1/`。

每组子目录如A_2026101903／B_2026101903相互独立。父级binding绑定独立准入协议；原100步失败、retry模型、旧gate和所有冻结产物不改。没有完整前组时不能开始后组，没有自动连续6组的命令。

## 只读准备／状态／核验

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-formal-admission
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli admitted-formal-status --condition A --seed 2026101903
```

完成首组后：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-admitted-formal --condition A --seed 2026101903
```

只读入口不导入Torch／反序列化权重／运行模型。完整同身份训练结果只核验，不更新参数或自行进入B。准备入口prepare-formal-admission仅创建科学协议，完整准备缓存只核验复用；未知／部分目录保留查因。训练启动时可导入Torch，但只读gate不得安装永久禁用模型的CPU保护；新适配器用完整只读hash／JSON链验证新gate，不调用旧失败目录的指标入口、不重算AP。

## 时间与下一阶段

全部6组已由用户手动完成并核验。更新段合计5082.627秒（84.71分钟），单组约11.75—24.34分钟，启动来源／准入hash核验与人工等待另计。准备时从probe线性推算的预算不是实际正式耗时。所有组记录无恢复；GPU完整恢复仍未实测。没有新增训练需求，不自动重训。

正常校准6/6及独立六模型来源汇总均已完成并核验；原整体失败及阈值保留。独立E1004真实评分入口与配置现已准备核验，E六组已全部完成并核验；六组真实指标已完成并核验；当前只整理第7天图件，不再运行模型或指标。冻结inference.py只提供格式与CPU原生窗口拼回，E评分已在real_evaluation.py独立实现，不修改旧CLI。不能将计划中的命令当作已实现入口，不能自动评价合成收益。见一周计划第六节唯一下一步及全部六组最终核验。
