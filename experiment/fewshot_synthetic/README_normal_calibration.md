# 第6天：独立正常校准

当前状态：本轮一周任务已完成，图件及[独立研究报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练研究报告.md)齐备。平均AP增益为正但不稳定，全部冻结成果及历史失败保留；下一项仅审阅独立曝光匹配提案，以一周计划第六节为准。

本模块只做正式6模型的Q200正常校准，没有训练、优化器、E／U评价或GT读取入口。训练与原核心源码、旧CLI均不修改。当前本轮只准备；以下`run`会执行真实模型前向，须在后续明确授权后使用。

## 已实现入口

在新的PowerShell进程中使用项目原解释器，保留`-B`避免向冻结代码目录写入字节码：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration verify-preparation
```

两条命令均仅核对文件hash、元信息与虚构标量，不导入Torch／读取Q图像或标签／加载模型。完整准备缓存只核验复用；陌生或不完整准备目录停止保留。它们不会读取实际概率或计算实际阈值。

后续获准进行实际正常校准时，使用以下独立入口；本次准备不运行：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration run
```

顺序固定A03、B03、A04、B04、A05、B05；每模型每张Q原图只做一次完整评分，每个规定窗口前向一次。不能把窗口数当原图数。代码不存在自动训练、参数更新、恢复、补推理或E评分入口。

完整实际结果保存后，CPU只读核验使用：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration verify-results
```

该入口会校验缓存、读取保存概率及最大值，复核阈值算术和FP／TN计数，不打开Q原图／GT，不进行前向或重新选择规则。实际结果尚不存在时停止，不生成结果。`run`发现完整同身份缓存时也只核验，不加载Torch或重新评分。

## 冻结口径

仅六份最终10000步自有权重，不选中间权重，不用probe或其他组权重初始化。使用原作者结构的GN8监督分割，FP32／batch1／workers0、eval＋inference_mode，无可变buffer，无优化器。加载前核对文件hash、加载后核对最终参数hash；全部预测后再次核对参数不变。

Q200沿用split原顺序与编码文件hash。RGB除以255，L复制三通道，不缩放、拉伸对比度或翻转；256窗、stride128及H−256尾窗，右侧reflect补到256。只对原有效宽度拼回，重叠概率等权平均，保存原生H×W的float32异常softmax图及其最大值。不平滑、拉伸概率或更换整图分数。

每模型对完整200个最大值分别取q=0.95、linear分位数，严格`score > threshold`判异常。同池FP／200、TN／200只是校准池描述，不是独立正常验证，不能支持AUROC、异常召回或像素AP。相同分数的ties可使FP少于10，不为了得到10个FP调整阈值。

## 输出与事务

- 科学准备：`D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/normal_calibration_preparation_v1/`，严格封存protocol.json、Q_requests.json、CPU_validation.json及complete.json。
- 未来实际结果：`D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/normal_calibration_v1/`，独立六模型概率、分数、阈值与同池计数，以及必要的事务身份和完成封存。

新模块自带CLI，源码hash直接绑定科学协议；旧cli.py不修改，因此未来E评价新增独立模块不会因扩展旧CLI而改变此次实现身份。六份权重、封存训练结果、原核心／附加适配器、环境、来源与固定Q请求共同绑定；封存后不改本模块来追加E功能。

历史准入核验使用已封存admission、技术gate及原失败文件的静态身份，并核对保存计数表hash，不重新计算AP；不会再次查询历史失败试跑的PID是否存活，避免系统复用该编号造成无关拦截。旧retry／准入源码与原失败记录保持不变。

结果父目录独占创建。每个窗口前向前先持久保存当前模型／图像／y的cursor，图像完整保存后更新；异常、Ctrl+C、OOM、未知临时文件或硬中断保留全部现场，不自动删除或恢复。即使部分模型完整，只要六模型总事务不完整，重用`run`也停止，不自动继续剩余模型／窗口。完整同身份缓存须严格文件清单、hash、全200图顺序、原生概率与最大值、参数记录和阈值算术均通过才复用；不取成功交集。

## 资源与尚未实测部分

元信息共200张独立正常图、每模型844窗，六模型1200次整图评分／5064次窗口前向。每模型29202876原生像素，六份概率float32净载荷约0.653GiB，另有NPY头与JSON；不是1200张独立图像。

CPU准备至少2GiB可用RAM、80GiB磁盘。实际运行启动／每模型前至少4.5GiB可用RAM、6GiB空闲显存、80GiB磁盘；每张图像前检查RAM≥3GiB与磁盘，不自动降batch、精度、窗口或切换CPU。采样不保证连续内存或GPU峰值，实际资源由运行结果记录。

GPU加载／前向、整图推理耗时和实际峰值本轮未实测，不用训练时间换算推理时间，也不做隐含前向的资源试跑。实际阈值与FP／200尚未知；须后续单独授权。所有训练仍由用户手动启动，本模块没有训练功能。
