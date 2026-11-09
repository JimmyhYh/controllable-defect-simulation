# 第3阶段：固定CPU输入与独立技术小试

仅新增data.py、pilot_config.py、pilot_preparation.py、pilot.py、pilot_metrics.py及本页，不改第1／2阶段封存源码／环境或vendor。当前准备不导入Torch或模型，不运行训练／评分／指标。

CPU旧环境OpenCV4.11.0.86仅解码固定R5全部5张与旧N16全部16张，保存原始对比度预览、坐标／来源、标准数值NPZ；新环境后续只读该缓存，无需OpenCV或旧site-packages。原生GT不改；RGB线性／GT最近邻到640×232，ImageNet归一化，特征标签160×58按作者bilinear align_corners=True及>=0.5。未来人工运行在模型构造前核对实际Torch算子二值结果。不能因细标签丢失换图／膨胀／借U。

已实现CPU入口（旧.venv）：`ssn_baseline.data prepare／verify／accept-visual`；全部固定样例实际审阅后才accept。`ssn_baseline.pilot_preparation prepare／verify`冻结小试协议、完整1600更新序列、初始化seed2026102101及来源。`ssn_baseline.pilot status`只读状态，无模型操作。

实际手动训练入口（当前只准备，不自动执行）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv-ssn\Scripts\python.exe' -B -X utf8 -m ssn_baseline.pilot train
```

固定50epochs×32更新=1600；每epoch16个R5循环项＋全部N16无放回排列，真实／正常输入曝光各800。作者B→2B是特征视图；无额外RGB合成／翻转。仍为作者BN、Perlin／噪声、双头、focal＋截断L1与AdamW三组，完成40／45epoch后按作者gamma0.4衰减。模型峰值、耗时和完整恢复尚未实测，不能称兼容已通过。

仅最终1600步的同池拟合验收：特征全局AP≥0.95、各R5 AP≥0.85，分类头严格>0.5参照TP5/5且FP≤1/16、全部有限且特征图非常量。原生AP只描述，不参与门槛。这是预设工程检查，不是文献保证或独立验证，更不代表真实收益／屏幕效果。训练命令只训练保存，不自动评分或计算gate。

模型起点RAM≥6GiB、空闲显存≥6GiB、磁盘≥80GiB；运行RAM采样≥3GiB。资源／非有限／缓存／实际算子标签不一致即停止保留，不降规格／换数据或重试。完整缓存只读核验；partial停止。

资源准入后先写独立事务，再导入Torch／核对实际算子／构造模型；初始化或恢复失败也留现场，不能无记录重复。未来评分逐输入采样RAM并保存最低值，低于3GiB停止，不自动补评分。

完成epoch10／20／30／40／50存自有全状态检查点；Ctrl+C不另存。恢复需用户显式选本小试完整checkpoint并指定全新run-name（train --resume-from 参数与 --run-name已实现），仍到总1600步，旧事务不覆盖。GPU完整恢复和位级重放未实测，不保证。

其余已实现但本阶段不执行的入口：`pilot verify-train`只读训练完整身份；`pilot score`仅用最终权重对固定21输入各一次评分，保存原始特征／原生概率及分类分数，无GT参与模型评分、不改BN；`pilot_metrics compute`是独立CPU同池精确AP及gate，不导入Torch。未授权时不调用，不自动进入Q／E／U、校准或正式训练。

当前CPU准备核验命令：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m ssn_baseline.pilot_preparation verify
```
