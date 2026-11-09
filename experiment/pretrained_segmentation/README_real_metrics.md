# 独立CPU真实指标

仅新增real_metrics_common.py／real_metrics.py／real_metrics_cli.py／本说明，不修改任何封存小试／正式／校准／E评分模块或全局路径。准备产物在results/pretrained_segmentation/ksdd2_r5_v1/real_metrics_preparation_v1，未来结果在同级real_metrics_v1。

`prepare`／`verify`只核CPU元信息／来源hash／标量，不解码任何GT或概率。NPY仅流式hash绑定既有1004图全量评分核验证据；实际原生像素146411528，GT异常／正常和阳性分母留待实际compute。原τ=7.264203168233501e-06保持不变，不重算Q分位数。

实际计算入口已实现，准备任务不会自动启动：
```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.real_metrics_cli compute
```

仅显式compute读取固定E规范GT及保存原生概率。复用冻结fewshot_synthetic.metrics.exact_ap计算全局精确AP；整图AUROC、严格大于原τ的TP／FN／FP／TN、召回／正常误报分别报告，不采样／平均逐图AP或重复计窗。无阳性／单类别／零分母按指标写NA/null，不取子集。首个TP／FN／FP／TN按冻结E原ID顺序，空组NA，保留选择原因，不打开RGB挑图。

`verify-results`只读完整指标／GT缓存与保存值，不重算AP／AUROC或解码原GT。完整同身份缓存复用；部分／未知事务、身份差异／failure均停止保留，不自动重算／恢复／补样／近似替代。独立exact磁盘事务也不能单独冒充总事务完成。

所有复用路径同时复核上游E准备、E评分及E／Q独立核验证据的完整文件／目录集合和hash；上游新增未知文件／空目录／锁也停止，不只核已登记文件。准备中只hash E概率，不解析。

准备／计算RAM≥3GiB、磁盘≥80GiB；完整结果只读核验2GiB／1GiB。精确表两份合计8522825736字节，保留全部表及hash；运行耗时与峰值待实测。E已见，只能作单seed／固定R5和划分／单类别／未知采集分组的探索性证据，不是新未见验证、稳定收益、预训练因果或屏幕效果。所有训练始终由用户手动启动。
