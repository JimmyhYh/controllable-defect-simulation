# CPU真实指标v3资源兼容入口

v2在首个模型的精确AP资源准入前失败：新旧模块共用的Windows内存查询函数被设置了不兼容的结构体参数类型。不是RAM不足，也没有开始AP累计。v2失败现场保留，v3使用新协议与新结果目录，不续接或改写部分目录。

新增exact_ap_resource_compat.py通过私有WinDLL函数查询RAM，并给冻结fewshot_synthetic.metrics.exact_ap创建相同代码对象、仅替换资源依赖的独立globals副本。原函数、原模块globals、源码和所有数值算法不改；RAM3GiB、D/E盘各80GiB门槛保持。prepare含受保护CPU资源兼容核验，只编译函数定义并核资源入口，不执行AP函数体或读取GT/概率数组。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v3 prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v3 verify
```

手动启动新的独立CPU指标事务（六模型顺序执行，不训练或调用模型）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v3 run
```

结果位于E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results/real_metrics_v3。权重、原Q200阈值、1004图顺序、标签规则与原生像素/AP/AUROC/计数/三对B-A口径全部继承v2。只有实际运行后才能给出模型效果；完整同身份缓存只读核验，部分或未知事务停止保留，不自动恢复或重复AP。E历史已见，仅作探索性评价。训练始终由用户手动启动。
