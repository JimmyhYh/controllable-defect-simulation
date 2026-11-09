# 独立纯真实正式基线

本入口只准备一个seed的正式基线，不包含合成、Q校准、E评分或指标。封存小试源码／入口／全局路径不改。

CPU准备与核验（旧CPU环境；不导入模型）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.formal_cli prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.formal_cli verify
```

`prepare`解码且hash核对R5／N1885训练侧1890份RGB与1890份GT，仅核尺寸、二值与角色合法性，无新增视觉选择、合成或模型操作；保存原生CPU输入和10000请求。未知／失败／不完整产物保留停止，不自动重做。

只有用户手动启动以下参数更新入口：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv-pretrained-seg\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.formal_cli train
```

seed2026102202、10000更新、奇R偶N各5000曝光、R各1000；N1885中均匀有放回选图／合法窗，R均匀非空窗，水平／垂直p0.5同步翻转。namespace0先预存1000个R5洗牌循环，再依序抽R窗；namespace1为N图／窗，namespace2翻转，均PCG64／SeedSequence，与模型随机流分离。

同一ImageNet ResNet18＋新U-Net解码器／单二分类头，不加载probe／旧任务权重。编码器BN统计固定且affine微调、decoder GN8、CE＋Dice含空GT；FP32／batch1／workers0，Adam encoder／decoder1e-5／1e-4，完成8000／9000更新后分别乘0.2。复用原model.py函数和随机状态序列化函数，不修改其全局路径或创建优化器作准备核验。

启动RAM≥4.5GiB、空闲显存≥6GiB、磁盘≥20GiB；每100步显示进度并采样，可用RAM<2GiB于明确更新边界保存完整停止点退出。小试8.88分钟仅作线性参考，正式约44分钟为按更新数外推，实际IO、增强、检查点与峰值尚未测量，不保证时间。

结果`results/pretrained_segmentation/ksdd2_r5_v1/baseline_v1`；模型`models/pretrained_segmentation/ksdd2_r5_v1/baseline_2026102202`；CPU输入／协议`results/pretrained_segmentation/ksdd2_r5_v1/formal_preparation_v1`。每1000步完整保存，共10点；受限`weights_only=True`读取经本组封存hash验证的optimizer。Ctrl+C不额外保存。部分／未知产物或failure不自动恢复，不能删锁或改名绕过检查。

仅已知运行暂停且本组完整保存点核验一致时，用户才可在同一`train`命令后显式增加`--resume-from`及本组最新完整点绝对路径；不提供未实际存在的恢复路径。恢复模型／optimizer／全部RNG／请求游标与下一更新LR，GPU完整恢复尚未实测。完整训练缓存同命令只核验，不更新。

训练后先只读核验：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.formal_cli verify-training
```

该核验无模型导入、加载或前向。未完成不视为训练完成；训练结束不自动校准、评分、计算AP或进入下一实验。所有代码／配置封存后，新评分或校准另建独立模块／入口。
