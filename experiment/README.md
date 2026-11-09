# 公开工业基线与受控合成 Demo

## 长期运行要求

所有模型训练、含参数更新的资源试跑、重训与断点续训由用户在PowerShell手动启动。助手准备代码、命令和协议核对，并在用户训练后检查结果；执行后续计划也不自动启动训练。建库、推理或评价入口不能隐式调用训练。完整项目约定见[AGENTS.md](D:/华兴源创/AGENTS.md)。

## 第四周第5天：完整真实评价、Demo与图件已完成

首轮种子20261020，PatchCore参照与A/B各10000步权重、Vnew479阈值保持冻结。Tfinal完整1100张（正常350、异常750），每模型／图像只评分一次，原始分数与float32连续图保存于results/weak_localization/real_evaluation。全部72089600像素，正8135158、负63954442；GT最近邻缩放256，无正掩码变全零。RGB全图双线性256，PatchCore另做ImageNet归一化，DRAEM使用float32/255；无额外增强或跨图归一化。

| 模型 | 全局像素AP | 整图AUROC | TP／750、FN | FP／350、TN |
|---|---:|---:|---:|---:|
| PatchCore | 85.8349% | 0.908632 | 635/750、115 | 45/350、305 |
| DRAEM-A | 13.9807% | 0.499497 | 16/750、734 | 9/350、341 |
| DRAEM-B | 15.6421% | 0.503730 | 28/750、722 | 13/350、337 |

B−A像素AP+1.6614个百分点，召回+1.6000个百分点，但正常误报+1.1429个百分点；绝对效果仍很低，一对训练不证明稳定收益。cylinder占正像素97.7659%，高全局AP不能代表小缺陷定位：PatchCore mura类型＋全部正常图的像素AP18.7696%，spot仅0.5400%。完整类型指标、分母与样例规则见[metrics.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/real_evaluation/metrics.json)，已整合到[主报告第20.8节](D:/华兴源创/报告/显示屏缺陷检测调研报告.md)。

像素AP为原float32分数的精确非插值AP，磁盘外排序、相同分数组计数，不分箱或逐图平均。全池及各类型＋正常共享一次排序；每类型与正常的首8张实际保存预测共48张，与sklearn进行数值核验，差不超过1e-12。模型逐个推理后退出GPU进程，独立CPU进程计算指标。推理／保存约135.790、236.122、241.920秒（PatchCore另含N80显示尺度）；指标约66.176、54.589、61.133秒，总指标进程183.124秒。推理GPU峰值allocated366.81／675.37／675.37MiB，CPU累计峰值1.607／1.801／1.803GiB；指标阶段独立峰值RAM未单独记录，不能套用推理峰值。计时边界详见主报告。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' localization.py evaluate
& '.\.venv\Scripts\python.exe' localization.py predict --image '.\data\weak_localization\AnoLCD\display\test\cylinder\000.png' --model PatchCore --output-dir '.\results\weak_localization\single_demo_PatchCore'
Start-Process -FilePath '.\results\weak_localization\single_demo_PatchCore\preview.png'
& '.\.venv\Scripts\python.exe' figures_localization.py
```

evaluate默认分进程执行score与metrics；可加--stage score／metrics执行对应阶段。完整缓存核验后返回inference_cached／metrics_cached，不重复推理。未完成预测只补没有有限分数的路径；身份不同停止查因，不自动重建库、改阈值或训练。

predict支持--model PatchCore／A20261020／B20261020；默认B，可指定任意公开单图及独立output-dir。预测不读取GT，已有Tfinal图复用保存响应；无缓存的单图才推理。已实际运行同一预定cylinder/000.png：PatchCore输出异常，B输出正常，保留该漏检对照，不挑最好样例。输出preview.png、prediction.npz与prediction.json；PNG被占用时自动用新文件名，不重跑模型。图5／图6及CSV、SVG、PDF在报告/图示，绘图只读成果。显示范围为A/B统一0—1、PatchCore由N80像素q99.5固定0—0.985474652；不改变原始评分或AP。

当前训练和本轮评价均已完成。Tfinal已被观察，今后以它调参只能当开发观察，新的改善确认需另留独立数据；无多种子稳定性证据。以下第4天及更早部分为历史快照，当前以本节为准。

恢复边界：若恰在PatchCore最后一张测试分数保存后、N80显示范围文件写入前中断，恢复入口会复用全部测试分数，但不会自动补建缺失的显示范围，届时应先查因，仅补N显示范围，不能为此重推Tfinal。本次display_scale.json与N热图已完整并通过核验。完整缓存的metrics输出包含较长配置，简明数值可直接查看上述指标表或metrics.json。

## 第四周第4天：首轮训练与Vnew校准已完成

2026年10月13日核验用户手动完成的A/B：种子20261020，各10000步，无中断恢复；四份最终权重及训练源码哈希匹配。两组首层初始化摘要与完整B/T/M组件链摘要相同，各有4791个有效合成异常、5209次正常／无有效掩码分支；A替换0次，B低强度替换2395次，实际2395/4791=49.9896%。奇数末尾有效异常不替换。A训练76.243分钟、B75.112分钟；显存allocated峰值各2286.73MiB，reserved2442MiB，CPU峰值分别2.940／2.846GiB，系统可用RAM每100步采样的最低值分别1.456／1.405GiB。实际配置仍为256、batch1、workers0、FP32及固定学习率日程。

复用新N80的819向量PatchCore库及两组最终权重，仅对固定Vnew479张正常图逐张评分，每模型／图像组合一次。三方法均采用全图RGB双线性缩放256、float32/255；PatchCore另做ImageNet归一化，DRAEM不额外归一化。不增强、裁剪或跨图归一化，不读取真实异常或测试标签。PatchCore保留建库时的作者占位前向／BN初始化行为，校准不fit；DRAEM采用eval、无梯度推理，重建图与原输入拼接，整图分数为异常softmax概率最大值。PatchCore整图分数为原始最近邻平方L2的最大patch距离。

固定τ=quantile(Vnew_scores,0.95,method='linear')，严格score>τ报警。校准原图身份、最终模型身份、路径／分数对应和全部保存文件已核验，并从保存分数独立复算阈值及报警数。

| 模型 | 固定阈值τ | 校准正常图报警／分母 | 本次加载、读取、评分与缓存耗时 |
|---|---:|---:|---:|
| PatchCore | 1.8928574562072753 | 24/479（5.0104%） | 55.931秒 |
| DRAEM-A | 0.999303674697876 | 24/479（5.0104%） | 106.263秒 |
| DRAEM-B | 0.9995600938796997 | 24/479（5.0104%） | 109.031秒 |

这些报警数来自确定阈值的同一Vnew，是校准统计，不是独立正常误报成绩；同为24/479不证明方法效果相同。A/B阈值接近1，后续仍按预定最大概率评分与严格大于规则评价，不据此调参。Tfinal1100仍未推理，尚无真实AUROC、异常召回／漏检、独立正常误报或像素AP，也没有检测提升结论。

校准显存allocated峰值PatchCore366.81MiB、A/B各675.37MiB；CPU进程峰值1.312／1.529／1.531GiB，各模型加载前可用RAM2.129／1.853／1.822GiB。仍为私人RTX5060 Laptop、PyTorch2.8.0+cu128环境。时间不含Python启动／导入、初始资源身份核对与最终总配置写入；CPU峰值为同一进程累计峰值，不是三组独立进程测量。

最终冻结输入为[results/weak_localization/frozen_config.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/frozen_config.json)，包含三个固定模型、各自阈值、文件身份、配对核验和处理规则；原始路径与float64容器保存的原始分数见Vnew_scores_*.npz，数值精度来自模型原始输出，不是新增计算精度。calibration_*.json包含各模型校准结果与资源口径。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' localization.py calibrate
```

已有完整结果时，该命令核对源文件与缓存身份后返回calibration_cached=true，不重新评分。未完成时仅补NPZ中仍缺失的路径，已保存有限分数不再推理。若JSON写入时中断导致完成标记损坏，当前会停止并需查因修复；本次完整结果及缓存返回已核验，不受影响。完整训练断点恢复仍未实际验证。

第3—4天现已完成；第5天需要补齐完整Tfinal评价、单图Demo与图件。evaluate／predict新任务入口尚未实现，旧demo.py仍只对应leather。以下早期准备内容按当时快照保留，以本节为当前状态。

## 第四周第3天：手动启动训练（运行方式与早期快照）

2026年10月13日，新N80 PatchCore正常库已建立，保存在models/weak_localization/patchcore，共819个1024维向量；实测加载、建库及保存约8.92秒，结果见results/weak_localization/patchcore_bank.json。旧leather库与结果保持原样。没有读取Vnew／Tfinal图像进行评分。

DRAEM-A最初自动训练已停止并交由用户手动启动，当时最后输出900／10000步且尚无检查点，该未保存进度不作为成果。此后用户已从头完成A/B各10000步，最终状态见上方第4天部分。

在Windows PowerShell中执行以下两行，只启动A、种子20261020，不启动B或评价：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' localization.py train --condition A --seed 20261020
```

没有保存结果时从第0步开始；当前已有完整A/B结果，再运行各自训练命令会核验最终权重后返回training_cached=true，不启动训练。程序训练时每100步显示step、loss、lr和elapsed_seconds，每1000步在models/weak_localization/draem_A_20261020保存resume.pt与resume.json。中断后执行同一命令，会核对身份并从最近完整保存点恢复（显示RESUMED）；两个文件不完整或身份不一致时会停止，需先查因。Ctrl+C不会额外保存当前步，可能丢失最近保存点之后的进度；不要在正在写检查点时关闭窗口，也不要同时启动两份训练。完整恢复路径尚未经过实际中断恢复验证。

前900步实测约0.47秒／步，10000步估计约80分钟；第2天保守预算约1.83小时／组。电脑负载和检查点写入会影响实际时间。接通电源、留出内存、避免电脑睡眠，并保持PowerShell窗口开启。训练只用固定N正常背景与DTD素材，参数由first_round_config.json固定，无需手动改学习率或挑权重。

结束时应显示completed=true和steps=10000，并生成models/weak_localization/draem_A_20261020/reconstruction.pth、segmentation.pth及results/weak_localization/training_A_20261020.json。中途loss只是训练状态，不能解释为真实检测提升。新任务train入口已实现，evaluate／predict入口尚未实现；下面第2天和第1天部分保留当时快照。

## 当前首轮范围：只训练一对A/B

2026年10月13日用户调整第四周计划，正式训练／测试前将当前范围改为种子20261020的一对A/B；每组10000步，方法、数据划分、合成和评价口径不变。原种子20261021／20261022为后续可选重复，不等待它们才评价首轮。

当前执行配置是[first_round_config.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/first_round_config.json)，protocol的active_execution_config记录其身份。正式训练入口读取此配置；原training_config.json及下面三种子时长为第2天历史预算，原prepare／probe缓存仍保持有效。首轮A/B已由用户手动完成并核验，总训练耗时约2.523小时；实际资源与Vnew校准见上方第4天部分，Tfinal尚未评价。

首轮模型清单只有PatchCore、A20261020、B20261020；第5天交付单次真实指标、A/B差值、定位图及Demo，不虚构多种子误差线。后来补两对时保持原配置、完整汇总已有与新增结果；同一Tfinal不变成新的独立验证。若看过测试后改方法或强度，后续确认需新的独立数据。

## 第四周第2天：划分与训练资源配置已冻结

2026年10月13日第2天完成，尚未执行第3—5天。实际新划分为N80／Vnew479／Tfinal1100；PCG64种子20261012，以已知RGB身份组隔离，原图路径及完整测试保留见[split.json](D:/华兴源创/实验/弱缺陷Demo/data/weak_localization/split.json)。1840张训练来源正常图未使用，物理采集批次分组未知，不能声称跨屏幕独立。

32对训练侧诊断中λweak=0.1全部通过有效变化门槛，掩码外变化为0；[完整残差](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/synthesis_check.json)。四张training_pairs预览均为训练合成，不是测试预测或真实Mura。诊断全部强制有效异常，正式采样保留作者随机正常／异常分支，两者区分。

A/B资源试跑分别40步（10预热、30计时），只用N前8张及DTD。FP32可运行，损失／梯度有限、权重更新，初始化与40步B/T/M组件流复核一致。A/B每步中位0.527070／0.518573秒，显存峰值2286.74MiB，reserved2442MiB；CPU进程峰值约1.767／1.958GiB，最低系统可用RAM约0.380／0.402GiB，仍需留出内存余量。[实际试跑数据](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/probe_results.json)

冻结配置为256、batch1、num_workers0、Adam1e-4、每组10000步、8000／9000步学习率乘0.2、三个配对种子20261020／21／22。预估每对3.660小时、三对10.981小时，仅为短试跑预算推算。试跑参数不用于正式训练，也没有保存模型权重；正式模型从配对初始化重新开始。[training_config.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/training_config.json)

内存选型依据为保守估算，而非建库实测：作者CPU特征列表／拼接峰值按两份计算，加1GiB固定峰值预留，扣除当前进程驻留后与80%可用RAM比较；256／160未满足，80满足。固定后不随临时内存改善扩大N，后续建库实际失败须查因，不静默改数据。

以下是已实现的PowerShell入口。完整缓存时核对固定划分、配置与保存数据后返回，不重生成诊断或运行GPU试跑；资源／源代码身份变化时停止说明。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' localization.py prepare
& '.\.venv\Scripts\python.exe' localization.py probe
```

当前没有新任务正式train／evaluate／predict入口；旧demo.py仍对应leather。原作者train_DRAEM.py依赖TensorBoard和不同默认配置，不用于本计划直接训练。本次资源包装复用模型／损失，无需安装TensorBoard。协议与缓存保存于results/weak_localization，未读Vnew／Tfinal原图或评分、未改变旧V阈值／旧库／旧结果。

## 第四周资源准备：第1天已完成，尚未训练

2026年10月13日继续第1天：用户手动下载的AnoLCD.zip已解压并核验，完整清单见[dataset_manifest.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/dataset_manifest.json)，资源状态见[protocol.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/protocol.json)。实际训练正常2400、测试正常350、测试异常750、掩码750；4250个文件通过CRC与完整解码，均1024×1024，掩码为非空0／255，与异常一一对应。

跨训练／测试有一组RGB内容完全相同：train/good/1114.png对应test/good/168.png与284.png。保留完整测试，训练1114只标记排除，不删除；可用正常候选2399张、2391个已知重复身份组。测试仍1100张，只有1095种RGB内容，物理来源独立性未知。包中只有display任务目录，五类异常分别cylinder200、line150、mura50、scrathes100、spot250；保留原目录拼写scrathes，在报告解释为划伤，不改原文件名。

另有3对完全相同RGB的cylinder测试图，其二值掩码边界不同，掩码IoU分别约0.960／0.978／0.983。完整原测试图与原掩码均保留，不按标注差异选样或修标签；此为标注一致性限制，掩码结构核验不等于语义正确性核验。

README公开宣称正常训练2100、1000×1000，与本包不同；原因未明。包内没有独立用途条款或来源分组记录，用户已提供作者公开仓库，其明确发布数据以支持复现实验，据此在个人非商业研究范围确认第1天准备完成，当前day2_ready=true；尚未执行第2天或训练／评分；该完成声明包含文件核验、重复处理规则和用途范围核对，不代表训练已完成。下列2026年10月12日内容是下载前资源快照。

2026年10月12日按[第四周执行计划](D:/华兴源创/计划/第四周执行计划.md)执行第1天的非数据资源准备。用户使用流量，本次没有下载数据集、权重或安装依赖；仅取得4个固定提交的DRAEM作者源码，共27142字节。第1天真实数据核验尚未完成，第2天未开始。资源配置见[新任务protocol.json](D:/华兴源创/实验/弱缺陷Demo/results/weak_localization/protocol.json)，它还不是冻结训练／评价协议。

- 设备快照：i7-14650HX，RAM15.622GiB／可用2.641GiB，RTX5060 Laptop8151MiB／空闲6751MiB，驱动595.97，D盘空闲约260.224GiB；Python3.12.7、torch2.8.0+cu128、torchvision0.23.0+cu128、NumPy1.26.4，CUDA可用。没有实例化模型或测量训练峰值。
- DRAEM：沿用提交2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef，补齐model_unet.py、loss.py、train_DRAEM.py和tensorboard_visualizer.py；原生成源码与许可保持不变。模型／损失／生成模块可导入，作者训练入口的TensorBoard依赖当前缺少，不能直接宣称训练命令已可运行。第2天再适配计划内训练入口，不在此安装旧Linux环境依赖。
- PatchCore作者模块、Apache许可及138223492字节ImageNet V1权重已核对；权重SHA256为95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8。旧正式库、配置和阈值保持不变。
- DTD：原先只解出20张，本次从本地625239812字节压缩包补齐1860张train1素材，1880张全部通过文件／图片完整性核验，原20张保持相同内容。train1／val1／test1清单互斥，没有解出val／test图片。未来训练必须按固定train1列表读取，不直接扫描外部图片。
- AnoLCD：只发现D:/华兴源创/实验/AnoLCD.zip.baiduyun.p.downloading临时文件，没有读取其内容、续传、改名或解压；完整数据、用途条件、像素标注、来源分组和任务类别尚未核验。用户主动延后下载期间，不触发自动tile备选。

手动取得[作者提供的AnoLCD压缩包](https://pan.baidu.com/s/1y4ul30uijZ_oC5brcqEEfw?pwd=2468)后（提取码2468），将完整包放在D:/华兴源创/实验并提供完成后的路径，即可继续第1天数据核验。下载中临时文件不等于完整数据集。DRAEM和DTD无需重新下载，第一至三周命令仍按下文使用。

最新状态（2026年10月12日）：第三周第1—5天已完成，图3已更新、图4已补齐并整合到主报告。H48结果以[h_evaluation.json](D:/华兴源创/实验/弱缺陷Demo/results/calibration_stability/h_evaluation.json)为准，原始分数及每次误报见[h_scores.npz](D:/华兴源创/实验/弱缺陷Demo/results/calibration_stability/h_scores.npz)。下面保留的C阶段summary.json中H_evaluated=false是第2天快照，不是当前H状态；C摘要的冻结快照保持原样，图3页脚已更新为H已评价；绘图只读保存数据，不重新推理。

### 第3天：H48正常误报评价（已完成）

H48全部为正常图，未入正常库且未参与C阈值计算。沿用正式R80库、既有预处理和原始评分，对每张H只评分一次，再用已保存的全部500×4阈值评价，严格score > threshold。未fit、重推C、读取E/P原图或根据H修改任何设置。

| 校准张数m | FP中位数／48 | FP的5%—95%经验范围 | 全部500次FP范围 | FPR的5%—95%经验范围 |
|---|---:|---:|---:|---:|
| 8 | 4/48 | 0—7 | 0—18 | 0%—14.583% |
| 16 | 4/48 | 0—7 | 0—11 | 0%—14.583% |
| 32 | 4/48 | 3—4 | 3—7 | 6.250%—8.333% |
| 48 | 4/48 | 4—4 | 4—4 | 8.333%—8.333% |

每次分母为同一H48，500×4整数FP保存于h_scores.npz的FP_by_repetition，除48即可复算每次FPR。统计分位数为linear，经验范围不是总体置信区间，500次不是24,000张独立测试图。规范τC48=1.5711944103240967和原V阈值=1.5751205444335934均误报4/48、正确判正常44/48；两个参照在H上的判定相同。主要比较m16减m48的FPR差，中位数0个百分点，5%／95%经验分位数−8.333／6.250个百分点；不挑误报0的重复当最佳方法。

H分数最小1.267087、中位数1.348029、最大1.708006。H4/48不能与旧E21/32直接解释为算法提升；m48恒定仍来自全C阈值固定。没有新异常评价、AUROC、异常召回或像素AP，旧E92/92不转记为新阈值召回，工业结果不能称屏幕效果。

本次加载保存库0.930秒，H48读取／预处理／同步推理4.027秒，均值约0.0839秒／图；CPU指标计算0.002661秒。不含Python启动／导入、身份核对及文件保存。进程峰值工作集1327.86MiB，PyTorch显存allocated343.93MiB、reserved374MiB；运行前可用RAM2.716GiB、GPU空闲6838MiB，仍为i7-14650HX／RTX5060、PyTorch2.8.0+cu128，未触发备选。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' calibration_stability.py evaluate
```

已有完整H结果时，上述入口核对身份、冻结C阈值和保存指标后展示缓存，不重新加载模型推理；文件不一致或只存在一份结果时停止，不自动覆盖。旧demo.py evaluate仍为第二周E入口。

文件安排：H另存两个科学结果文件，未向原C NPZ追加或覆盖原summary，以保持第2天数据的字节身份和calibrate／图3入口。h_evaluation.json记录C冻结时间、C文件哈希、H数组哈希及分母；这是科学结果，不是过程表。图3及图4现已完成，更新只读取科学结果，未改变阈值或选择新方案。

第三周更新：2026年10月12日，[第三周计划](D:/华兴源创/计划/第三周执行计划.md)第1—5天已完成，C/H评价、图3／图4和最终报告已整合。第1天固定正式R80库627×1024，原R余下96张正常图按PCG64种子20261010划为C48/H48；245张原正常图内容身份一致且无重复，C/H及排除集合按原图互斥。第2天仅对C48评分和校准抽样，未读取H/E/P原图、建库或改动第二周结果。完整实际方法、结果、资源、运行入口及后续判断已整合到[主报告第20节](D:/华兴源创/报告/显示屏缺陷检测调研报告.md)。

最终科学协议位于[protocol.json](D:/华兴源创/实验/弱缺陷Demo/results/calibration_stability/protocol.json)，固定抽样种子20261011、500次、m=8/16/32/48、嵌套无放回、q=0.95、linear、严格score > 阈值。C分数及全部阈值已在H评价前冻结，当前实现prepare和calibrate入口：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' calibration_stability.py calibrate
```

上述命令适用于查看／核对已有C成果。prepare入口仅用于第1天协议与资源准备，调用形式为calibration_stability.py prepare，会核对全部正常图；不是只读C任务的附带步骤。calibrate只核对冻结源与C48，已有结果时核对缓存并重新计算保存分数的抽样统计，在模型加载前返回，不重新推理或改写文件。它不会调用prepare，也不会读取H/E/P原图。不一致或只有一个结果文件时停止说明，不自动覆盖；evaluate已实现为独立H评价入口，saved-v未启用，不能用旧demo.py evaluate替代第三周研究。

第1天核对：i7-14650HX、24逻辑线程、15.622GiB RAM，可用约1.956GiB；RTX5060 Laptop 8151MiB显存，核对时空闲6820MiB，驱动595.97；D盘空闲263.50GiB。Python3.12.7、PyTorch2.8.0+cu128及NumPy1.26.4与既有环境一致，CUDA元信息可用、计算能力12.0；当时仅核对元信息，未新增GPU计算或模型运行。动态余量不保证后续资源。CIM接口在本会话无法访问，RAM改由GlobalMemoryStatusEx、CPU由注册表核对；未将CIM失败所得空值／0当作硬件事实。主方案未触发V49备选，若后续推理条件不满足，按计划延期H评价。

第1天重复核对时可用RAM在约1.18—2.01GiB间波动，GPU空闲约6806—6820MiB。一次独立只读审查未发现第1天的阻塞问题。审查指出的后续H统计描述字段已在第2天summary.json中补为FP、FPR中位数及5%／95%经验分位数、linear方法；原C阶段未填写H数值；现已另存H评价结果，不更改冻结协议。

### 第2天：C48评分与校准抽样（已完成）

沿用正式R80库、ImageNet V1、既有初始化、全图224×224 RGB／ImageNet处理与原始平方L2最大patch分数；C48只顺序推理一次，batch1、eval／无梯度。没有新增图像因素、异常训练或正常库训练。结果见[summary.json](D:/华兴源创/实验/弱缺陷Demo/results/calibration_stability/summary.json)，原始科学数组见[scores.npz](D:/华兴源创/实验/弱缺陷Demo/results/calibration_stability/scores.npz)。NPZ包含48个C分数、对应路径、500×48抽样索引、500×4阈值及全C阈值，没有H分数。

规范τC48为**1.5711944103240967**；原V阈值仍为1.5751205444335934，没有覆盖原正式配置。500次有限C池抽样的描述统计如下，所有分位数为linear：

| 校准张数m | 阈值中位数 | 5%—95%经验范围 | 范围跨度 | 相对τC48绝对偏差中位数 |
|---|---:|---:|---:|---:|
| 8 | 1.527738 | 1.432462—1.914844 | 0.482382 | 0.059483 |
| 16 | 1.559842 | 1.486244—1.743872 | 0.257627 | 0.048363 |
| 32 | 1.568323 | 1.529424—1.598667 | 0.069243 | 0.022591 |
| 48 | 1.571194 | 1.571194—1.571194 | 0 | 0 |

每次从同一C48生成一个排列，四档取嵌套前缀，内部无放回；500次不是500个独立数据集。m=48包含全池，零波动来自构造，不证明总体稳定。本池中较小档位的经验跨度更宽，只支持池内敏感性观察，不证明解决官方E误报或屏幕检测提升。这一C阶段尚未评价H；H结果现见h_evaluation.json，真实异常召回和像素AP仍未新评价。

实际加载约1.384秒；C48读图／预处理／同步推理共4.070秒，均值约0.0848秒／图；CPU抽样统计约0.006684秒。以上不含Python启动／模块导入、身份核对和最终文件保存，不作为产线吞吐。进程峰值工作集约1326.86MiB，PyTorch显存峰值allocated343.93MiB、reserved374MiB。资源核对快照可用RAM1.581GiB、GPU空闲6799MiB，环境与第1天一致；当前主方案成功，未触发V49备选。

C科学数组与摘要已在H评价前冻结，缓存复核确认排列、阈值、分位数和全部48档恒定一致；重复calibrate未改文件。Day1协议及58个既有核心代码、资源和结果文件哈希保持不变。第三周第3天可读取这份冻结结果，但不能根据H重新挑m或改变阈值。

### 第4—5天：两张图与最终报告（已完成）

已根据保存C数据制作[图3阈值稳定性](D:/华兴源创/报告/图示/figure3_calibration_stability.png)，包含全部500次阈值的经验累积分布、各档中位数及5%—95%经验范围、原V和全C参照；新增[图4正常误报与C/H分数](D:/华兴源创/报告/图示/figure4_normal_holdout.png)，图3和图4均有SVG/PDF；图注见[主报告第20.1节](D:/华兴源创/报告/显示屏缺陷检测调研报告.md)。图3为183×145mm、PNG300dpi，SVG保留文字，PDF使用TrueType字体。未重跑模型、重新划分或生成新分数。

原summary.json的H_evaluated=false保留为C阶段快照，当前h_evaluation.json中H_evaluated=true，H分数与FP/48已保存。图3注记已更新，图4已新增；现有绘图入口同时验证C/H科学数组、描述统计及全部FP后重画两图，不加载模型。主方案成功，未启用V49备选。

C/H结果、两图、统计表及资源条件现已完整整合，保留全部预定阈值的FP/48，不看H挑最优m或重复。已有C池分析不能保留新阈值的真实异常召回，也不能解释屏幕效果；旧demo.py predict继续使用第二周原V阈值。主报告第20.4节区分了待验证的误报变化、正常覆盖问题与独立异常／屏幕数据条件。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' figures_calibration.py
```

本目录仅用于私人设备上的个人学习与非商业研究，使用公开资源，公司材料保持隔离。更新：2026年10月10日，第二周第1—5天已完成：资源取得、固定划分、正式正常库、V阈值、完整独立E评价、单图Demo、受控合成、两张核心图与主报告整合。没有真实屏幕实验，也没有使用合成异常训练检测器。

当前正式结果：整图AUROC=1.000；异常检出92/92、漏检0/92；正常误报21/32（65.625%）；补充全像素AP=0.467002。阈值1.57512054仅由49张V正常图确定，正式库627个1024维向量，仅来自固定R前80张。**高AUROC不代表当前判定可靠：正常误报较高，需要保留为基线失败结果。**完整协议、分母、误报路径、运行条件与局限见[真实评价结果](results/真实评价结果.md)。


合成观察另行报告：P正常误报3/20，固定B/T/M、仅改变λ的强组检出20/20、弱组18/20；弱组实际残差幅度约为强组1/5、能量约1/25。合成不进入正常库，不与E合并，不解释为训练收益。两张图及图注见[主报告第18节](D:/华兴源创/报告/显示屏缺陷检测调研报告.md)，综合讨论与第三周问题见第19—20节。

## 运行入口

当前环境、权重、正式库与结果已经就绪，无需重装或重复建库。优先使用PowerShell；复制以下命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' demo.py predict --image '.\data\mvtec_ad\leather\test\color\000.png'
$demoResult = Get-Content -LiteralPath '.\results\single_demo\prediction.json' -Raw -Encoding UTF8 | ConvertFrom-Json
Start-Process -FilePath $demoResult.preview_path
```

检测正常示例时，将图片参数替换为 `.\data\mvtec_ad\leather\test\good\000.png`。也可明确指定其他适用的公开图像完整路径；仅用公开图像，保持公司材料隔离。每次只读取所给图片，不查标签或配对背景；只更新 `results/single_demo/` 的单图数值和预览，不重建库、不重定阈值、不修改完整评价结果。输出是整图判定与评分、热图和叠加，没有像素判定阈值。

预览通常为 `preview.png`。Windows图片查看器可能占用该文件，导致直接覆盖报 `OSError: Invalid argument`；程序现在会保留已打开的图，将本次预览保存为带随机后缀的新文件。`prediction.json`中的 `preview_path` 和终端 `PREVIEW` 行始终指出本次实际图片。以下打开命令读取这个实际路径，避免打开上一张旧图。

若使用命令提示符CMD，对应命令如下。不要将PowerShell的&和Set-Location直接贴入CMD：

```bat
cd /d "D:\华兴源创\实验\弱缺陷Demo"
".\.venv\Scripts\python.exe" demo.py predict --image ".\data\mvtec_ad\leather\test\color\000.png"
".\.venv\Scripts\python.exe" -c "import json,os;os.startfile(json.load(open('results/single_demo/prediction.json',encoding='utf-8'))['preview_path'])"
```

完整评价入口为 `demo.py evaluate`，已经实际执行成功。已有完整结果时该命令核对固定库／配置后展示已保存指标，不重新推理E或重建库。删除结果或换配置不属于本轮任务。

第1天正常小批入口保留，PowerShell命令如下：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' demo.py probe --device cuda
Start-Process '.\results\day1_normal_probe.png'
```

重跑probe只更新第1天正常小批数值和预览，保留已经固定的formal_configuration、数据包来源和计时口径，不会改动R/V/P/E划分或第2天正式库／结果。它仍然是8张建库＋4张正常图推理。当前界面为命令行与静态预览，没有网页界面。

以下仅用于恢复包环境或核对划分；当前目录已有.venv，日常使用不需要执行：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 --cache-dir '.\.cache\pip'
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt --cache-dir '.\.cache\pip'
& '.\.venv\Scripts\python.exe' demo.py prepare --category-root '.\data\mvtec_ad\leather'
```

重新运行prepare只接受与已有文件一致的划分。重建环境需要网络访问；已缓存权重及数据后，本地建库／推理无需访问网络。

`data.py`负责固定划分、图像与标签预处理；`detector.py`封装作者实现；`evaluation.py`建立正式库、V定阈值、E评价与单图输出；`demo.py`统一提供prepare/probe/evaluate/predict/synthesize入口；`synthesis.py`负责合成探针，`figures.py`读取保存成果绘图。第2天正式库与阈值保存于 `models/formal_patchcore/`，第1天小批库不能替代它。图像建库与推理无梯度且使用网络eval模式，未做梯度微调。未新增或运行软件测试，也没有提交或过程表。

作者初始化默认会在eval前用全1占位输入查询特征维度，使BN运行统计更新一次；本轮沿用第1天已固定的这一源码行为，未根据E切换初始化方式。ImageNet V1是骨干权重来源，并非所有BN缓冲量保持原值。该初始化没有读取任何R/V/P/E图像；实际建库／预测均eval，重新加载重复相同初始化且分数一致。详细口径及局限见真实评价结果。

剩余问题：第2天无运行阻塞，主要实测局限为正常误报较高。正式建库6.602秒，V49读取及推理4.513秒，E读取及推理12.596秒（124张，单张均值约0.102秒）；进程峰值工作集约1984MiB，PyTorch显存峰值allocated约553MiB、reserved620MiB，口径见真实结果。正式评价函数内部总耗时27.903秒，不含Python启动／导入、下载安装及最终JSON写入。作者接口的 `pretrained=True` 弃用提示不影响当前ImageNet V1加载。Windows FAISS原生读写不支持本项目中文路径，封装层改为其字节序列化配合Python文件读写，已实际保存、重新加载并核对单图评分一致；只改文件接口，未改变索引、距离或抽样。第3天DTD与DRAEM受控合成已完成，见下方合成结果；本次工业指标不能代表屏幕／Mura效果。

首次在独立研究副本中运行evaluate，需要先准备公开leather、固定作者代码、V1权重、适用路径的split.json以及第1天已固定的formal_configuration；资源完整且尚无正式结果时才会建库、V标定、E评价。probe只输出小批结果，不会自动补齐正式配置字段。合成还需DTD split 1清单及固定20张素材。现有入口不会自动下载／解压全部资源，也没有--force或--rebuild参数。本轮不删除原结果，不利用E重选参数；新配置应另定协议并保留本周成果。

predict、evaluate和synthesize沿用保存配置中的CUDA设备；它们没有--device切换参数。只有probe接受--device。更换CPU／设备属于新的配置验证，不能未经核对直接覆盖本周结果。

## 资源和配置

- 数据：[MVTec AD 官方入口](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，固定 leather 类别，CC BY-NC-SA 4.0。用户提供的包为 `D:/华兴源创/实验/leather.tar.xz`，524889948字节，SHA256=`704ffa48b1dc2fc44824aa50442d9febc399dd92a5d6fd8156f8aa0373d15f24`。此哈希标识本地包，没有与官方公布校验值比较；包内原始许可证和说明已保留。数据位于 `data/mvtec_ad/leather/`，不重新分发。
- 检测器：[作者 PatchCore 项目](https://github.com/amazon-science/patchcore-inspection)，Apache-2.0；固定提交 `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`，来源压缩包为 `https://codeload.github.com/amazon-science/patchcore-inspection/zip/fcaa92f124fb1ad74a7acf56726decd4b27cbcad`。保留完整作者源码及许可证，核心算法未修改。
- 骨干：作者 `backbones.load("wideresnet50")`，对应 torchvision WideResNet50-2 的 ImageNet V1 预训练权重。[官方权重定义](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/resnet.py)。权重已下载到 `models/checkpoints/wide_resnet50_2-95faca4d.pth`，138223492字节，SHA256=`95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`，已核对文件名前缀哈希。
- 固定 layer2+layer3、patch3、嵌入1024、1% approximate greedy coreset（作者默认投影128、起始点10）、近邻1，不微调、不集成。GPU提取特征和抽样，CPU FAISS检索。
- 整图 RGB 双线性缩放224×224、/255、固定 ImageNet 均值／标准差；不裁剪、不逐图对比度拉伸。此配置与作者示例的预处理有明确区别，是小规模 Demo，不称完整论文复现。
- 正常训练路径排序后用 `numpy.random.default_rng(42)`（PCG64）排列：V取前floor(0.2N)，P随后最多20张，R取剩余；E保留官方test全部图像和标注。划分保存为 `data/split.json`，不使用E选型。第1天不计算V阈值，不运行E或P检测。

## 设备

已核对：Windows私人设备，Intel Core i7-14650HX、24逻辑线程、物理内存15.62GiB；核对时可用内存2.45GiB。NVIDIA GeForce RTX 5060 Laptop GPU，显存8151MiB，核对时空闲7417MiB，驱动595.97。D盘当时空闲约276GiB。动态空闲量不代表运行时保证。

项目独立虚拟环境 `.venv` 使用本机 `D:\Python\python.exe`（Python3.12.7），不更改全局包。实际安装 PyTorch2.8.0+cu128、torchvision0.23.0+cu128、numpy1.26.4、FAISS-CPU1.12.0、timm1.0.20、scipy1.16.2。已实际执行CUDA矩阵运算和作者网络建库／推理，GPU计算能力12.0（sm_120）兼容此构建。完整已安装版本见 `requirements-lock.txt`，最小直接依赖见 `requirements.txt`。

**正式配置选择前80张 R，GPU提取特征／coreset，CPU FAISS。**R本身仍保留176张，具体选用路径是固定排列中的前80张，保存于 `results/day1.json` 的 `formal_configuration`，不重新抽样。小批进程中系统可用RAM从2.55GiB降到1.73GiB；依据作者实现的float32特征列表＋拼接数组，全部176张的这两部分约1.05GiB，80张约0.48GiB，尚未计入其他临时内存。当前内存余量较小，故选80张保留余量，GPU显存充足；此选择未使用V、P或E检测响应。这是资源估算，不是正式建库实测或耗时保证。

## 小批链路与资源核对（非正式评价）

划分实际固定为 **R=176、V=49、P=20、E=124**。E中正常32张、异常92张；异常类别为color、cut、fold、glue、poke。本日只核对E目录、92张掩码的一一对应和几何尺寸，没有输入E做推理。R/V/P互斥，覆盖245张正常训练图；245张原图内容哈希均不同。

8张R图建库，随后用另外4张R正常图推理，两批互斥。原图RGB、1024×1024，保留全图缩放至224×224；小批库为62个1024维向量（8×784个特征的1%向下取整）。首次运行实测如下，重跑后的最新数值以results/day1.json为准：

| 项目 | 本机实测 |
|---|---|
| 8张正常图建库 | 0.645秒 |
| 4次单图读取及推理 | 0.0667、0.0751、0.0723、0.0755秒 |
| 原始整图分数 | 2.07754、1.81301、1.86932、1.61108 |
| 输出热图 | 4×224×224，数值全部有限 |
| PyTorch显存峰值 | allocated约352MiB，reserved约374MiB |

建库计时包括读图、预处理、特征提取、coreset和FAISS建库，不含骨干初始化／权重加载。单图计时包括读图、预处理、特征提取、CPU FAISS查询和热图上采样／平滑，CUDA已同步。显存峰值在模型初始化后重置统计，含常驻模型，不含其他应用或全部CUDA驱动内存。四次测量规模很小，不是生产吞吐量结论。

数值及最终配置见 `results/day1.json`，原始评分／热图见 `results/day1_normal_probe.npz`，可视预览见 `results/day1_normal_probe.png`。预览每行依次为正常原图、热图、叠加图，四张共用[0,1.76861]显示尺度。**正常图中的红色区域不能直接解释为缺陷：本日未定阈值，这些输出只证明链路可运行。**

## 第3天：固定P上的合成探针（已完成）

复用MIT许可DRAEM合成模块，固定版本2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef。DTD官方split 1训练1880张按路径排序、PCG64种子42排列取前20张，素材来源限研究用途；下载包625239812字节，哈希和实际素材路径见最终JSON。原官方包在data/dtd-r1.0.1.tar.gz，只解出20张需要的纹理和split 1清单。没有读取E图像／标注／衍生素材，也没有用P、合成异常重新建库或调阈值。

20对B/T/M固定，仅改变λ0.5与0.1，全部有效。P正常误报3/20，强组检出20/20，弱组18/20；热图最高点在操作M内分别16/20、14/20。弱组分数在20/20对低于强组，平均差−4.849511；掩码外残差全部为0，弱组实际幅度约为强组1/5、能量1/25。模型收到float32数组，PNG仅用于查看。P正常已经误报的3对不能解释为新增缺陷检出，所以另给17张P判正常的背景上强组17/17、弱组15/17。

完整口径、两张未检出分析、资源及适配处见[合成敏感度结果](results/synthetic_probes/合成敏感度结果.md)。原始float32 B/T/M及配对图位于results/synthetic_probes/pairs，评分／热图在scores_heatmaps.npz，源文件／素材／输出哈希和数值见evaluation.json。synthesis.py复用作者纹理增强和Perlin硬掩码；禁用背景随机旋转、随机beta和随机正常分支。作者纹理增强中的Gamma只是固定T的一部分，没有对背景或成对图额外加Gamma因素。固定λ0.1超出原随机范围，本结果为受控适配而非完整DRAEM复现。

PowerShell：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' demo.py synthesize
Start-Process -FilePath '.\results\synthetic_probes\first_pair.png'
```

有已完成结果时，核对固定库／权重／划分及合成源、素材、输出哈希后展示结果，不重新生成或推理。第3天内部耗时约9.205秒（不含导入／下载安装及最终JSON写入），60张数组推理3.233秒（不含读图与预处理）；进程峰值约1441MiB，PyTorch显存allocated约344MiB、reserved374MiB。计时不能与第2天含读图的计时直接比较。本轮只评价响应，没有训练收益，也不能声称屏幕/Mura效果。第二周成果已整合至主报告第16—20节，第三周实验尚未执行。

## 第4天：两张核心科研图（已完成）

两图及图注已放入[主报告第18节](D:/华兴源创/报告/显示屏缺陷检测调研报告.md)。[图1真实评价](D:/华兴源创/报告/图示/figure1_real_evaluation.png)展示首张正常、首张异常、首个正常误报及完整E指标；[图2合成响应](D:/华兴源创/报告/图示/figure2_synthetic_response.png)展示首个有效P配对、固定B/T/M、实际残差及20对分数轨迹。两图均有SVG、PDF、600 dpi TIFF及CSV科学源数据，统一保存在“报告/图示”。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' figures.py
```

该命令只读取已保存结果，不运行模型或生成新合成图；核对指标、选定GT、合成输出及固定配置后重画。两图共同热图色阶[0,6.0877156257629395]，首对残差共同色阶[0,0.19150328636169434]；只是显示尺度，没有改V阈值或原始数值。PNG/TIFF高dpi导出不增加224×224分析网格的原生信息。没有新增过程表、软件测试或提交流程。第5天综合整合已完成；第三周建议研究正常校准样本数与阈值稳定性，依据和边界见主报告第20节。第三周该实验现已完成，实际结果见上述第三周部分。
## 新增：可控缺陷生成试用

2026年10月15日第9天阶段4完成：仅CPU缓存计算，主强度像素AP G0=55.1564%、G1=99.0349%，弱强度11.7830%／93.0047%；G1检出224/230、208/230，正常FP1/20，G0为69/230、10/230及FP2/20。完整类型、亮暗、六模板、配对、资源和局限见[共享合成留出最终结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天共享合成留出评价最终结果.md)。`holdout-metrics`／`verify-holdout-metrics`已实现；完整结果只核验，不重算AP或前向。结果仅限共享G1渲染器域，真实独立数据仍待核验；下一项第10天图件和报告，以下为阶段历史。

2026年10月15日第9天阶段3完成：G0后G1各480次模型输入评分（20正常＋460有效异常），原始概率与整图最大值已保存，共960张float32热图；参数／BN缓冲不变，完整缓存复用新增前向0。没有训练、重校准或检测指标计算。实际资源、身份与命令见[阶段3只读评分结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天阶段3只读评分结果.md)。`holdout-score`完整缓存只核验，`verify-holdout-scores`显式只读核验；下一项为阶段4 CPU缓存指标，入口尚未实现。以下保留阶段历史。

2026年10月15日第9天阶段2完成：20背景各解码一次、480请求各render一次，460有效／20无效，两强度各230/240有效，正常20齐备。float输入／实际标签及manifest已冻结，全量数组核验与完整缓存复用通过，未模型前向／评分／训练。实际分母、PNG核验限制及命令见[阶段2共享输入结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天阶段2共享输入生成结果.md)。只读入口为`python -m controlled_generation.cli verify-holdout-data`（使用项目虚拟环境）；`holdout-generate`已有完整同身份缓存时仅核验，不重复生成，未完成事务停止查因。下一项阶段3评分须另行实施，尚未提供评分命令；以下保留阶段历史。

2026年10月15日第9天阶段1完成：S_bg20原顺序、六模板、亮暗与±0.04／±0.02、种子20261031的480异常请求＋20正常ID已冻结，G0／G1最终权重与实际阈值身份一致。没有读取S_bg字节、生成或模型前向，实际有效数仍未知。[准备结果及已实现命令](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天阶段1准备结果.md)。只读元信息入口为`python -m controlled_generation.cli verify-holdout-preparation`（使用本项目虚拟环境）；生成／评分入口尚未实现。下一项为另行授权的阶段2CPU生成，原模型／阈值与旧成果不变。以下按阶段保留历史说明。

2026年10月15日实际正常校准已完成：G0／G1阈值分别为0.6124536782503127／0.7155899673700332，固定Q_cal40、q0.95／linear／严格大于规则，同池FP均2/40。保存结果独立复算与完整缓存复用核验通过，无训练或留出评价，不能据同池FP称真实检测提升。详见[实际校准结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/calibration/第8天实际正常校准结果.md)。下方未执行校准等为各阶段历史记录。

校准准备已完成：原始异常softmax最大值、Q_cal40的q0.95／linear／严格大于口径及独立缓存身份已冻结。本次没有打开Q_cal图像／分数、加载模型或前向，没有实际阈值。后续授权评分命令和资源／中断限制见[第8天校准准备](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/calibration_preparation/第8天校准准备与运行.md)；GPU执行尚未实测，旧训练／评价成果保留。

2026年10月15日当前状态：G0／G1各10000步和全部检查点／最终权重均已核验；G1用时79.43分钟，初始参数与共同序列身份匹配G0，正常／异常各5000次。G1采样最小可用RAM0.792835GiB，资源余量情况与核验限制见[G1核验结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/formal_training/G1_20261030/第8天G1训练核验结果.md)。用户要求暂不校准，尚无检测效果结论；下方等待G1等内容保留历史含义。

2026年10月15日最新进度：G0完整10000步、10检查点及最终权重已核验，实测78.39分钟，正常／有效异常各5000次；G1尚未开始。下一步由用户手动执行 `python -m controlled_generation.cli formal-train --condition G1 --seed 20261030`（使用本项目虚拟环境）。可复制PowerShell命令、资源及结果见[G0核验说明](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/formal_training/G0_20261030/第7天G0核验结果.md)。未校准或评价，无检测质量结论；下方“尚未训练”等为准备时历史记录。

第7天已准备独立正式G0／G1共同配置、训练侧CPU采样核验及人工训练入口；正式训练步数仍为0，没有模型前向／校准／评价。当前由用户手动运行G0：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli formal-train --condition G0 --seed 20261030
```

每组10000步、同seed20261030从头初始化，完成8000／9000步后各乘0.2，不加载probe权重。预计约80—100分钟／组，启动前≥3GiB可用RAM、≥16GiB空闲磁盘，建议≥4GiB空闲显存。每1000步保存，自有完整检查点可按同命令恢复；恢复日程仅核验标量与代码，完整GPU恢复尚未实测。G1待G0完成，不自动训练或评分留出材料。输出、实际CPU核验和资源见[第7天准备与运行](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/formal_preparation/第7天准备与运行.md)。下方保留试跑与旧实验说明。

当前独立后段衰减试跑已由用户完成1000步，最终训练池AP99.9659%、拟合gate通过，实测7.74分钟。核验结果和单次／衰减前轨迹差异的限制见[第6天B4核验](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe_lr_decay/第6天B4真实试跑核验.md)。下方历史命令仅核验完整缓存：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli lr-decay-probe-train --seed 20261029
```

同8图／种子／1000步／stateless BN不变，801步起学习率2e-5，结果独立保存。后续第7天需先准备共同正式配置及人工入口，当前不启动正式训练。B2历史准备与运行信息见[第6天B2说明](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe_lr_decay/第6天B2准备与运行.md)。下方保留原归一化试跑历史结果。

第6天独立BN一致性试跑已由用户完成1000步，AP38.0213%仍未通过0.80门槛，实测7.90分钟；原AP2.0923%失败结果保留，诊断99.0203%不替换任何训练结果。下方命令仅核验完整缓存，不重训；当前资源、输出和失败分析见[第6天试跑说明](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe_batchstats/第6天归一化试跑说明.md)。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli batchstats-probe-train --seed 20261029
```

仅查看新准备／训练状态用batchstats-probe-status，不启动训练。正式G0/G1仍未启动。

2026年10月14日已完成CPU参数化Mura式／线状生成、24个请求的实际掩码核验及旧PatchCore响应展示，未做新训练或新真实评价。入口与PowerShell命令见[可控生成运行说明](D:/华兴源创/实验/弱缺陷Demo/controlled_generation/README.md)，图件与分母见[初步结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/可控生成初步结果.md)。以下原运行说明及旧成果保留。

