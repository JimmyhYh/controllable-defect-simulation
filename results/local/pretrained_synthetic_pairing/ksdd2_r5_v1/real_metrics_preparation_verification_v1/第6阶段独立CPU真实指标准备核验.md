# 第6阶段独立CPU真实指标准备核验

独立指标模块、协议和显式入口已经准备并通过新进程只读复核。准备目录绑定六组E1004完整评分缓存、六份最终10000步权重、各自原Q200阈值、固定E原序、规范标签缓存身份和冻结精确AP实现；模型顺序为A01、B01、A02、B02、A03、B03。

本阶段只读取JSON元信息、目录清单、路径、文件大小及既有hash。规范标签数组读取0份，概率数组读取0份，实际AP、AUROC、TP／FN／FP／TN、召回、误报率与B−A差均未计算；Torch及模型包没有导入，模型操作为0。历史110异常／894正常和486461阳性像素没有在本阶段重新核实，后续实际CPU入口必须逐份打开规范标签后再确认。

未来实际入口固定每模型146,411,528个原生像素，全局精确像素AP不抽样、不平均逐图AP；整图分数使用已保存原生max，并按每模型自己的原Q200阈值严格大于判定。三对报告B−A像素AP百分点差、均值和样本标准差，同时报告AUROC、召回、正常误报率及计数差。任何部分或未知事务都保留并停止，不取成功交集、不替换指标、不自动恢复。

准备时可用RAM为7.322GiB，D盘119.644GiB，E盘772.929GiB。未来实际计算入口的硬门槛为RAM至少3GiB、D盘至少80GiB、E盘至少80GiB。E1004已经被历史实验看过，后续结论只能作为固定单类别、固定R5与划分、三初始化下的探索性训练组成证据，不能称为新未见验证、总体显著性或屏幕收益。

已实现的只读准备核验命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics verify
```

已实现但本阶段未运行的实际CPU指标命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics run
```
