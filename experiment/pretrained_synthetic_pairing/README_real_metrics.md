# 独立CPU真实指标

模块`real_metrics.py`绑定六组已完成E1004评分缓存、各自最终权重和原Q200阈值。准备阶段只读取JSON元信息、文件hash与目录清单，不打开规范GT数组或概率数组，也不计算真实指标。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics verify
```

未来经明确授权后，实际CPU指标入口为：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics run
```

实际入口复用旧实验中已封存并独立核验的1004份规范uint8二值标签，不重新读取原始GT。运行时逐份核标签和概率hash、shape、dtype、有限性与原生顺序，再按每模型146,411,528个原生像素调用冻结`fewshot_synthetic.metrics.exact_ap`；同时计算整图AUROC、各自原tau严格大于的TP／FN／FP／TN、召回、正常误报率和三对B−A。六模型全部完成后才封存；部分／失败状态保留停止，不自动恢复、重累计、取成功交集或替换指标。

实际计算要求可用RAM至少3GiB，D盘和E盘各至少80GiB。E1004已用于历史实验，结果仅作固定单类别／固定R5／固定划分／三初始化的探索性训练组成证据，不称新未见验证、统计显著或屏幕收益。所有训练仍由用户手动启动。
