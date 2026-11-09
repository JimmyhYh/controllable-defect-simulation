# 独立E1004真实评分入口

仅本分支新增五文件；不修改小试／正式／校准入口或其全局路径。完整协议及资源说明在 `results/pretrained_segmentation/ksdd2_r5_v1/real_scoring_preparation_v1/第6阶段E1004真实评分准备与运行.md`。

在项目根目录，用CPU环境 `.venv/Scripts/python.exe -B -X utf8 -m pretrained_segmentation.real_scoring_cli prepare` 或 `verify` 做元信息／hash／窗口坐标／虚构标量核验。不读Q／E／U原图、GT或概率，不导入模型，不计算实际阈值／指标。

实际评分须另行授权或用户手动运行：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv-pretrained-seg\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.real_scoring_cli run
```

run只用最终10000步模型、固定E原序和已冻结原生预测器，每图／每窗一次，保存原始float32概率与max；不读取GT，不计算指标、重新取Q分位数或训练。原τ=7.264203168233501e-06绑定其原来源。完整同身份输出仅CPU只读复用，任何部分／未知事务保留停止；无隐式补推、自动恢复、删锁或降低规格。新运行前残留runtime也停止。输出独立real_evaluation_v1，运行临时目录独立real_scoring_runtime_v1。

完整评分后的CPU只读核验：用 `.venv/Scripts/python.exe -B -X utf8 -m pretrained_segmentation.real_scoring_cli verify-results`。新增前向0；只读本组保存概率及原max核对，不读GT或计算AP／AUROC／计数。real_cache.iter_metric_inputs是未来CPU指标读取器，仅返回完整核验后的原序概率／GT来源元信息；实际GT解码及指标入口尚未实现，本次不能当作已评价。

CPU准备RAM≥2GiB／磁盘≥2GiB；新评分启动RAM≥4.5GiB、空闲显存≥6GiB、磁盘≥20GiB；每图前RAM≥2GiB／磁盘≥1GiB。时间／峰值以未来实测为准，准备和静态审阅不等于新入口已运行兼容。所有训练仍由用户手动启动。

E已被历史实验使用，只作探索性评价；不能称新未见验证、总体显著性或屏幕收益。本次准备不读取旧评价选配置，旧源码／环境／配置／权重／阈值／缓存／失败与成果保持不变。
