# 第6阶段独立CPU真实指标准备与运行边界

本次只完成独立v2指标模块、协议、来源hash、GT路径／既有hash、六模型顺序和虚构标量准备。六组E1004评分完整缓存、最终权重及各自原Q200阈值均以已封存完成身份绑定；没有打开6024份概率数组。规范GT复用旧实验中已独立核验的1004份uint8二值标签缓存，本次只读取其JSON身份、路径、已有hash与文件大小，不打开标签数组，也不重新读取原始GT。

v1准备和代码保留不变。独立审阅指出v1缺少实际CPU依赖身份且会重复全量读取概率；v2分别绑定Python解释器hash／版本、NumPy／scikit-learn／SciPy版本，并将实际概率输入改为每份一次read_bytes后hash、内存解码、校验及yield。准备期只用标准库读取依赖元信息，没有导入这些数组／指标包。修正后只做静态与受保护CPU元信息核验，实际数组／第三方计算兼容仍待明确运行；独立审阅者的修正复核因使用额度未完成，不声称二次独立审阅通过。

实际计算入口将逐份核规范标签hash、uint8、{0,1}、原生尺寸和实际异常／正常／阳性像素分母；历史110／894与486461只作运行时一致性检查，当前不宣称重新核实。每模型按原生146,411,528像素全局计算精确像素AP，不抽像素、不平均逐图AP、不改用近似分箱；复用冻结`fewshot_synthetic.metrics.exact_ap`及既有sklearn／实际ties交叉证据。整图AUROC、各自原tau严格大于的TP／FN／FP／TN、召回和正常误报率同时保存，空分母或单类别写NA。

三对全部报告B−A：像素AP百分点差为主终点并汇总均值与样本SD（ddof=1），同时报告AUROC／召回／误报率百分点差及错误计数差。固定样例规则只使用第一对A01/B01，按E原序取A01首个TN／FP／TP／FN及四种A→B翻转，去重保留全部原因，空组NA、不替补；冻结ID时不打开RGB。

实际输出为`E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results/real_metrics_v2`。任何部分／失败事务均原样保留并停止，不自动恢复、重累计、取六模型成功交集或替换指标。完整同身份结果只读核验，不重复AP／AUROC；计数、率、ID和样例选择规则从保存整图分数与规范标签作语义一致性核对，AUROC核封存hash与合法范围，不宣称独立重算验证。E曾用于历史实验，结果只作本固定单类别／固定R5／固定划分／三初始化的探索性训练组成证据，不称新未见验证、统计显著、合成机理单因素因果或屏幕收益。

本次资源快照：RAM 7.140GiB，D盘 119.638GiB，E盘 772.929GiB。未来实际入口要求RAM≥3GiB、D≥80GiB、E≥80GiB；六模型精确表理论字节数为51136954416，运行时间尚未实测。所有训练仍由用户手动启动，本CPU指标不训练模型。

只读准备核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v2 verify
```

实际CPU指标入口（已实现，本次未运行）：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v2 run
```
