# 独立复跑的CPU指标入口

当前状态：本轮一周任务已完成，图件及[独立研究报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练研究报告.md)齐备。平均AP增益为正但不稳定，全部冻结成果及历史失败保留；下一项仅审阅独立曝光匹配提案，以一周计划第六节为准。
只使用retry_ram_v1最终1000步的12张原始float32概率及冻结训练侧标签；不使用旧probe-metrics，不加载模型或权重，不重新生成或读取Q／E／U图像标签。所有训练继续由用户手动启动，此模块没有训练入口。

## 当前阶段：准备

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli prepare-probe-retry-metrics
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-probe-retry-metrics-preparation
```

准备核对5个检查点文件及元信息hash、最终概率、输入包、复跑协议／绑定、旧核心与新指标实现、环境；只检查缓存合法性／分母，不计算AP或实际ties。权重只读hash，不反序列化。完整同身份准备缓存只核验，不重新生成或写入旧目录。

固定分母：12张，704256有效像素／26414阳性；真实5张294400／24482，合成3张175616／1932，正常4张234240／0。右侧补边排除，概率不平滑／拉伸；正常组AP为NA，不凭未校准分数定义误报或召回。真实5张也是训练材料，分组不能称独立真实验证。

独立准备：`results/fewshot_synthetic/ksdd2_v1/probe_retry_metrics_preparation_v1`。未来指标：`results/fewshot_synthetic/ksdd2_v1/probe_retry_metrics_v1`。不向严格封存的retry_ram_v1追加文件。

## 后续显式计算（本轮只准备，尚未运行）

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli compute-probe-retry-metrics
```

该命令只做CPU缓存指标，须后续明确执行；不会训练或运行模型。先核验完整输入和协议，准入可用RAM≥3GiB、磁盘≥80GiB。复用冻结metrics.exact_ap：两份uint32位序表共约7.94GiB，全局精确处理所有float32分值及ties，不抽像素、不平均逐图AP。与全12张sklearn结果核对至1e-10，并用独立unique核对实际distinct及ties数量；另给5真／3合成／4正常分组池化AP，正常无阳性为NA。

原最终AP≥0.80、训练有限性及训练RAM条件均满足且交叉核验通过，才能报告训练侧拟合gate通过；gate=false也完整保存并允许只读核验，不挑中间权重或重复求好结果。它不证明真实留出收益，不自动开放正式组或改原正式入口。

结果完整后，只读核验：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.cli verify-probe-retry-metrics
```

完整同身份结果只核验hash／已保存结论，不重算AP、sklearn或ties。部分结果、未知文件／目录、身份变化、计数或交叉核验不一致均保留停止，不能删除重跑、重复累加或改用近似AP。父目录在实际计算开始时独占创建；中断后即使内部exact已完成，未封存的整个指标事务仍不能自动续算。资源不足、精确实现或缓存出现异常先用systematic-debugging查因。

本轮只做CPU身份／输入／静态标量核验，未实测磁盘AP耗时、峰值及交叉核验结果，不能承诺时间或提前宣布gate通过。完整运行后再审阅六次正式训练预算与准入。
