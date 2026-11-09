# 第3阶段CPU准备与人工小试

第1／2阶段封存代码、环境、源码／配置／权重均只读。本阶段新增common／data／preparation／model／training／scoring／metrics及薄CLI；所有模型库仅在用户显式训练／评分时延迟导入。准备阶段不加载权重或运行模型，不能说明可学习性已通过。

固定5张R各首个GT非空原生256窗，旧N16各首窗，共21窗。无缩放／合成／增强或对比度拉伸；右反射补边不参与损失／指标。CPU准备使用原.venv的NumPy1.26.4／Pillow12.3.0，不往新环境装包。prepare-inputs／verify-inputs核对固定输入；prepare-pilot首次需要完整逐图实际视觉review文件，已有完整缓存只核验；verify-pilot只读核验所有输入、源码、协议和标量。

已实现的准备核验命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.cli verify-pilot
```

准备完成后，以下入口会更新参数，**只能由用户手动启动**：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv-pretrained-seg\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.cli train-probe
```

seed2026102201，从通用编码器和新解码器开始，FP32／batch1／workers0，共2000更新、R／N各1000曝光。encoder／decoder LR为1e-5／1e-4，完成1600／1800步后共同乘0.2。每400步完整保存，固定最终2000步验收；不选中间最好权重。训练耗时与峰值待实测，启动RAM≥4.5GiB、空闲显存≥6GiB、磁盘≥20GiB。

模型与完整检查点位于models/pretrained_segmentation/ksdd2_r5_v1/probe_2026102201，训练结果results/pretrained_segmentation/ksdd2_r5_v1/probe_v1。Ctrl+C不额外保存，最近未保存进度会丢失。低RAM在明确更新边界保存本组资源停止点；任何不明确/部分事务保留查因。不自动恢复，用户显式`train-probe --resume-from`只接受本组最新完整点；GPU完整恢复未实测。检查点模型使用safetensors，自有optimizer文件仅核验hash后以weights_only=True恢复，随机状态为JSON；不会加载旧任务/probe权重。

训练结束后不会自动评分或计算AP。用户报告后先用原.venv执行`verify-probe`只读核验；明确授权后才使用新环境`score-probe`对固定21窗各评分一次，再在原.venv用`probe-metrics`计算CPU验收。`verify-probe-scores`和`verify-probe-metrics`只读复用，未知／部分事务不补推理、不重新累计AP。

最终技术门槛为全21窗有效域精确AP≥0.95、每R窗AP≥0.85、有限训练／概率、异常图非常量及参数／BN控制；正常分数和像素0.5占比仅描述，无独立分类头和SSN分类0.5门槛。全部21例展示，正常AP无阳性时为NA。它只证明同池拟合，不代表未见泛化、屏幕收益或合成训练提升。正式10000步仅核标量日程，本阶段没有正式训练入口。所有训练仍由用户手动启动。

本阶段完成后，上述小试代码／CLI／本说明的hash随协议封存。后续正式训练须另建模块及独立显式入口，不修改本次common全局路径或向已封存CLI追加功能来破坏小试身份。
