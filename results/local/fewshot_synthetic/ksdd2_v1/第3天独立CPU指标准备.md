# 第3天独立CPU指标准备

2026年10月19日。状态：指标代码、冻结协议、输入合法性与身份准备完成；未计算AP、sklearn、实际ties或gate，未运行模型／训练，正式0/6。

## 已准备内容

另建probe_retry_metrics.py及薄CLI入口，复用原metrics.exact_ap，不改原6份核心、probe_retry.py、冻结权重／输入／概率或原100步失败现场。新代码绑定43份科学文件hash（含5个检查点及元信息、最终完成清单、12张概率、12份输入包、复跑协议与binding）、旧核心／适配器、新指标实现与解释器依赖，权重只读hash、不反序列化。

固定全12张，704256有效像素／26414阳性。真实5张294400／24482，合成3张175616／1932，正常4张234240／0。逐图核对保存float32概率有限、范围[0,1]、原256尺寸、二值标签和有效域，补边排除；不平滑／拉伸、不重生成或模型前向。真实5张和合成3张均是训练输入，不是独立真实验证。

[冻结协议](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/probe_retry_metrics_preparation_v1/protocol.json)及[输入核验／资源](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/probe_retry_metrics_preparation_v1/input_validation.json)位于独立准备目录；未来指标放probe_retry_metrics_v1，不向严格封存retry_ram_v1增加文件。

## 实现的未来计算口径

全局精确float32位序AP使用全部有效像素与原ties，非抽样、非逐图AP均值；独立sklearn核对绝对差≤1e-10，独立unique核对实际distinct／ties，另给5真／3合成／4正常各组完整池化AP。正常0阳性AP为NA，不假造阈值或误报／召回。仅最终1000步AP≥0.80、训练有限性、RAM及交叉核验共同通过才报告训练侧拟合gate；失败结果同样封存并可只读核验，不挑中间权重或重算求好结果。

准入RAM≥3GiB、磁盘≥80GiB；两份uint32表约7.94GiB，chunk1048576。完整同身份指标缓存只核验hash／已保存字段，不重算AP、sklearn或ties。父目录在计算开始独占创建，部分事务、未知产物、身份变化、计数／交叉核验失败保留查因，不重累计或近似替代。内部exact完成但父事务未完成时也停止，不自动续算。完整数值、耗时和实际资源仍待后续执行，本轮没有做磁盘AP性能试跑。

## 本轮核验与执行取舍

CPU prepare和完整准备复用通过；强制禁止np.load及exact_ap后，完整准备仍能只读核验。复用新增数组加载0、AP调用0、Torch／sklearn导入0，指标输出目录不存在。语法、通过与失败gate一致性、错误分母／tie计数／布尔gate／正常AP的4类内存标量拒绝核验通过；标量样例不是实际研究AP，未写入科研结果或软件测试套件。

独立源码审阅发现旧独立verify_retry会间接安装只允许原PREP写入的永久保护，阻断新目录。先用无文件写入的audit事件复现，再只在新适配器中复用冻结复跑准备／binding检查，将已核验split／identity显式传给原训练核验函数，并保留初始化hash比较；新指标目录允许写入、原PREP仍拒绝写入，实际prepare通过。未改或绕开旧保护，代价是后续必须保留这组显式完整核验，不能换回旧无参入口。没有剩余Minor；审阅未判实际AP／数值一致性／性能，这些本轮明确未执行。

用户要求优先：只保存必要配置和最终说明，不建过程表、软件测试套件或Git提交流程。原训练输出与两个适配器旧身份核对不变；没有Q／E／U图像标签读取。准备完成不授权计算或训练，也不开放正式训练准入。

## 已实现命令

准备已完成，只读复核可用：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-probe-retry-metrics-preparation
```

后续明确执行CPU指标时的命令（本轮未调用）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli compute-probe-retry-metrics
```

完成后verify-probe-retry-metrics只读核验，不重算。完整用法和停止条件见[运行说明](D:/华兴源创/实验/弱缺陷Demo/fewshot_synthetic/README_probe_retry_metrics.md)。当前无AP、拟合通过或独立真实／屏幕收益结论。
