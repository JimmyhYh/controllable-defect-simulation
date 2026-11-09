# 第6天正常校准：独立续接

当前状态：本轮一周任务已完成，图件及[独立研究报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练研究报告.md)齐备。平均AP增益为正但不稳定，全部冻结成果及历史失败保留；下一项仅审阅独立曝光匹配提案，以一周计划第六节为准。

本模块只处理原事务从未开始的A_2026101905与B_2026101905，各自新的Python进程。原normal_calibration_v1的失败／cursor及四份完整缓存原样保留，旧run仍拒绝部分事务。独立assembly最终只引用六份来源，不伪造旧事务成功。

## 准备与边界

独立入口为`fewshot_synthetic.normal_calibration_continuation`，不修改旧CLI或源码全局路径。独立准备协议绑定两份新源码、旧准备／权重／环境、诊断、原token和失败位置、前四全部文件hash及固定Q200请求。

准备／只读准备核验命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration_continuation prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration_continuation verify-preparation
```

CPU准备会读取保存的scores／summary元信息及800张概率文件的hash／NPY头，以核对来源、原序、维度、float32和完整性；不解析概率数组数值载荷、重新计算实际阈值、读取Q／E／U原图／GT、导入Torch或运行模型。虚构标量核验与真实缓存元信息分别标注，不能把“虚构标量没读真实分数”误写成整个准备没读取scores元信息。

准备目录严格独立：`results/fewshot_synthetic/ksdd2_v1/normal_calibration_continuation_preparation_v1/`，只含protocol.json、Q_requests.json、source_manifest.json、CPU_validation.json、complete.json。新运行根为`normal_calibration_continuation_v1/`，最终引用根为`normal_calibration_assembled_v1/`；准备不会创建后二者。

## 独立两组及CPU引用汇总均已完成

A05、B05均已完成并只读核验；各200图／844窗、参数保持不变。A05阈值0.008925506938248865、B05阈值0.0003234100207919243，同池FP均10/200，不是独立验证。两组评分段分别62.078秒／22.844秒，不能据耗时差异解释资源根因。

阶段D已实际CPU执行assemble、独立verify及完整缓存只读复用；严格输出仅sources.json、verification.json、complete.json，原四与新二仅来源引用，不复制概率／分数／阈值。不再运行Q模型或重取分位数；E独立评分已准备并核验，E六组实际评分均已完成并核验6/6，CPU真实指标已实际完成并核验，当前只整理第7天图件。

[B05最终核验](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/normal_calibration_continuation_verifications_v1/B_2026101905/阶段C独立B05正常校准核验.md)。

## 科学口径与资源

继承原GN8、FP32／batch1／workers0、eval＋inference_mode、无buffer；只加载指定final10000，加载及评分后核参数hash，无优化器或参数更新。256窗／stride128／尾窗H−256、右reflect、去补边原生等权拼回、原始float32异常softmax／max评分不变。每新模型全200图后只计算一次q0.95／linear／严格大于阈值，分别保存同池FP／200、TN／200，不能称独立验证或计算真实异常召回／AP。

CPU准备需RAM≥2GiB、磁盘≥80GiB；每次实际运行在Torch导入前及模型创建前复核RAM≥4.5GiB，空闲显存≥6GiB、磁盘≥80GiB，每图前RAM≥3GiB并核磁盘。不得自动降门槛、精度、窗口或回退CPU；没有含前向的资源试跑。

原前四模型每组评分段约16—18秒，只是参考，不含新进程启动、来源hash／头核验、CUDA与权重加载。A05评分段已记录62.078秒，启动／来源核验与加载另计；B05评分段22.844秒、最低图像边界RAM6.049622GiB，不能把评分段当端到端耗时或把边界采样当全程峰值。阶段D CPU汇总前RAM5.811729GiB、磁盘206.219093GiB；预核验后assemble段16.063秒，尚无CPU全会话峰值统计。

## 事务与最终引用

固定原token／failure／cursor且原后两目录缺失、源码证明资源检查发生在模型创建前，才准入本例外；不是通用断点恢复。新目标独占创建、token独立，每窗口前向前保存cursor，完整图保存后更新。完整同身份缓存只核验；任何部分目标／陌生文件／临时文件／失败／不明确前向保留停止，不删除、重跑或另换目录躲避保护。失败后先查因，不自动继续下一模型。

最终`normal_calibration_assembly`模块只在两份新结果均完整后按原六模型顺序核验并建立引用清单，不复制图、分数或阈值。对应CPU汇总及独立核验CLI均已实际执行通过。`resolve_calibrations()`只读返回经核验的源路径和hash，供未来独立E评价使用；不会安装阻止未来E调用者加载Torch的永久CPU hook，本身不导入Torch或写文件。命令行汇总／核验则仍启用CPU保护。

六模型仍只有200张独立正常图，1200次整图评分／5064窗不能当1200张独立样本。原失败整体一直保留未完成，六组完成由新的assembly表示；当前无E评价或真实收益结论。公司材料隔离，全部训练始终由用户手动启动，本模块不训练。

独立静态审阅仅保留一个效率Minor：汇总／resolver核验B05时会再次读取已经核过的A05缓存（200图），不增加前向或重算阈值。本轮不优化冻结实现；两组新进程实际评分与本阶段CPU汇总均有完成证据；其他环境、中断及跨进程峰值仍未经证明。

## 已实现CPU汇总／只读核验命令

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration_assembly assemble
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.normal_calibration_assembly verify
```

完整同身份assembly再次assemble仅核验复用；未知／部分目录保留停止。本次已在禁止所有写入的CPU边界中核验完整复用，三文件hash不变。verify只读核验六份来源与概率数值，不重新取实际分位数或运行模型。六份权重仅文件hash；参数／buffer一致是保存记录核验。继承检查对虚构arange／ones标量的quantile不等于重算Q阈值。
