# 第6阶段CPU指标资源兼容修复与v3准备

v2于第一个模型A01调用冻结exact_ap的资源准入时失败，报expected LP_Memory instance instead of pointer to M。Windows共享函数原型被较新资源查询设为另一结构体指针；旧查询因此拒绝参数，并非RAM门槛不足。失败目录保留，尚无AP表或模型指标；labels/summary.json记录v2已解码标签汇总，本阶段只读取其JSON，不重新解码，不把部分目录当完整指标结果。

v3独立保存源码、协议与结果，不改v1/v2、冻结fewshot_synthetic源码、环境、权重、tau和概率。exact_ap的代码对象、参数、BINS、CHUNK、ties及数值运算不变；通过新的独立函数globals副本仅适配c.resource，用私有WinDLL函数查询RAM，同时保留D/E各80GiB、RAM3GiB门槛，其余c服务仍引用冻结实现。原模块globals与共享DLL函数原型均不由适配器更改。不调用冻结模块的原始资源查询，不修改其路径或源码。

准备复现了原参数错误，验证独立资源查询和同代码对象/私有依赖绑定；仅编译冻结函数定义并检查其资源入口，没有执行AP函数体、导入NumPy/Torch或打开标签/概率数组。实际完整CPU指标仍未运行。六模型、固定E顺序、原tau、每模型146,411,528原生像素、全局精确AP/AUROC/计数/三对B-A及样例规则继承v2，不调参或重新校准。

本次可用RAM 6.864GiB、D盘 119.638GiB、E盘 772.788GiB。实际运行门槛RAM至少3GiB、D/E各80GiB；六套精确表约47.63GiB。未实测完整六组耗时；冻结扫描内部无连续RAM监控。任何部分或未知事务保留停止，不自动重累计、恢复或缩分母。

v3实际入口本次不执行，由用户明确手动启动新事务；不续接v2标签、AP表或部分结果。原v2没有AP累计，因此新事务不会重算已完成AP。E历史已见，仅作探索性组成证据，不称新未见验证或屏幕收益。

只读核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v3 verify
```

新独立实际入口（本阶段未执行）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v3 run
```

输出：E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results/real_metrics_v3。六组共用同1004唯一图，不能累计为6024独立图。训练始终由用户手动启动。
