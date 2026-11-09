# 可控缺陷生成试用

## 当前：第9天四阶段已完成，模拟域指标齐备

主强度全局像素AP为G0 55.1564%、G1 99.0349%，差+43.8784百分点；弱强度11.7830%／93.0047%，差+81.2217百分点。G1检出224/230和208/230，正常FP1/20；G0为69/230和10/230、FP2/20。全部预定组别与230/240共同有效配对已计算；本阶段CPU缓存、无Torch／前向／训练／重校准，最终权重与阈值不变。见[最终中文结论与资源](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天共享合成留出评价最终结果.md)。这是G1渲染器域比较，不能外推真实屏幕收益。

只读核验指标和源缓存，不重算AP或加载模型：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli verify-holdout-metrics
```

`holdout-metrics`已实际完成；完整同身份结果仅核验复用。未知metrics产物、未完成事务、锁或.tmp保留查因，不自动覆盖。下一项第10天依据保存的12个去重样例绘图，仍只读缓存。以下保留阶段历史。

## 当前：第9天阶段3两组评分缓存齐备，指标未计算

G0→G1各480次模型输入评分，先20正常再460有效异常；同一RGB数组、原顺序、最终10000步权重与原阈值。960张原始float32概率图及最大值已保存，参数／BN缓冲未变。完整缓存复用无新前向，无训练／校准／真实评价。资源、事务边界和科学局限见[阶段3结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天阶段3只读评分结果.md)。

只核验评分缓存，不加载模型或新增前向：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli verify-holdout-scores
```

`holdout-score`也已实现并完成，完整同身份缓存只核验复用。部分条件或不明确前向保留查因，不自动补推理。阶段4指标模块尚未实现，不提供指标命令；当前没有检测提升结论。以下保留阶段历史。

## 当前：第9天阶段2共享输入已冻结，尚未评分

20背景各解码一次、480固定请求各render一次；460有效／20无效，两强度各230/240有效，正常20齐备。20无效请求均为亮方向裁切后实际残差0，未重试或补数；全部无效材料和原因保留。输入、实际float标签与manifest已冻结，两模型尚未前向、评分或训练。分组分母、残差、PNG核验限制和资源见[阶段2实际结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天阶段2共享输入生成结果.md)。

核验完整共享输入（读取文件hash／NPZ，不解码源图、不渲染／前向）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli verify-holdout-data
```

`holdout-generate`也已实现并实际完成；完整同身份缓存存在时仅核验复用。`generation_started.json`保留单次生成边界；不完整事务、错配、owner锁或.tmp须保留查因，不自动重生成。后续评分／指标模块未实现，不提供尚不存在的命令。以下保留阶段历史。

## 当前：第9天阶段1元信息已冻结，尚未生成或评分

固定S_bg20原顺序、六模板、亮暗与±0.04／±0.02、20261031种子规则，480异常请求＋20正常ID已写入独立protocol／requests。G0／G1最终10000步权重及实际阈值身份核验通过。没有打开S_bg图像／字节、调用render或执行前向，实际有效异常数未知。详细分母、禁止救样本规则及资源见[阶段1准备结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/shared_holdout_v1/第9天阶段1准备结果.md)。下一项为另行授权的阶段2CPU生成；未实现生成／评分命令。

只核验元信息、请求和模型文件身份：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli verify-holdout-preparation
```

`holdout-prepare`也已实现；现有完整同身份准备只核验复用。后续另建holdout_data.py及生成身份，不改已冻结holdout_preparation.py或原渲染／训练／校准实现。以下保留历史说明。

## 当前：两组实际正常校准已完成

各组用固定Q_cal40，原始异常softmax最大值、q0.95／linear／严格大于规则不变；G0阈值0.6124536782503127，G1阈值0.7155899673700332，同池FP均2/40，不能当作独立正常误报验证。详见[实际正常校准结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/calibration/第8天实际正常校准结果.md)。两组calibration-run完整缓存已核验复用，不会重新加载模型或做前向；不得换配置重评分选结果。S_bg和真实评价尚未执行，新独立数据待核验。GPU正常评分已执行，硬中断恢复尚未实测；下方暂不校准／等待训练等为历史。

## 当前：G0／G1训练均已核验，暂不校准

2026年10月15日，两组各10000步、各10个完整检查点和最终权重核验通过，共同初始参数／背景分支序列身份／LR历史一致。G1实测79.43分钟，正常5000次，亮／暗Mura式和线状各1250次。G1训练期间采样最小RAM0.792835GiB，不能声称全程满足3GiB余量。详见[G1核验结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/formal_training/G1_20261030/第8天G1训练核验结果.md)。

没有校准阈值或检测质量结论，下一项可单独准备校准入口；本次未执行前向或读取留出评价。只读保存状态可用 `formal-status --condition G0` 或 `formal-status --condition G1`；正式训练命令已有完整结果时仅核验缓存。以下等待训练等为历史说明。

## 当前：G0完整10000步已核验，等待人工训练G1

2026年10月15日核验G0的10个完整检查点、日程和最终权重通过；实测78.39分钟，正常／有效异常各5000次。检测质量尚未评价。完整结果见[G0核验说明](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/formal_training/G0_20261030/第7天G0核验结果.md)。当前命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli formal-train --condition G1 --seed 20261030
```

预计80—100分钟，各10000步、相同初始化及8000／9000衰减日程；≥3GiB可用RAM、≥16GiB磁盘，建议≥4GiB空闲显存。每1000步保存，同命令恢复自有完整点，GPU恢复未实测。G1尚未启动；不自动校准／评价。下方保留准备时和旧试跑历史，G0原训练入口仅核验完整缓存。

## 第7天：正式共同协议与人工训练入口

第7天只准备两组输入、共同配置及人工入口，正式模型尚未训练。配置与全部预定训练侧请求的CPU核验见[第7天准备与运行](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/formal_preparation/第7天准备与运行.md)。当前先由用户手动启动G0：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli formal-train --condition G0 --seed 20261030
```

两组各10000步，seed20261030从头初始化；共同FP32／CUDA／batch1／workers0／stateless BN与作者损失。第1—8000次更新1e-4、第8001—9000次2e-5、第9001—10000次4e-6。G0是DTD纹理与Perlin常规合成，G1是现有可控亮暗／线状采样，比较整个生成方法；不共享异常掩码。两组均用固定N_gen80，同背景与正常／异常序列、实际浮点残差标签及五次重试上限。probe权重不作初始化。

每1000步保存完整自有检查点；同一命令恢复自己的完整保存点，Ctrl+C不额外保存，未保存步数丢失。预计每组约80—100分钟，基于1000步试跑7.74分钟粗估，正式耗时、峰值资源与完整GPU恢复尚未实测。启动前至少3GiB可用RAM、16GiB空闲磁盘，建议4GiB空闲显存；两组顺序运行，建议合计保留32GiB磁盘。输入、源码、软件版本、恢复身份或资源不一致时停止，不能自动更改预算或覆写旧成果。

模型独立保存于models/controlled_generation/formal_draem/G0_20261030；训练结果位于results/controlled_generation/formal_training/G0_20261030。G1需等G0完整并核验初始身份，不自动启动下一组或校准。只查看准备／状态可运行verify-formal和formal-status --condition G0，它们没有模型前向或参数更新。新独立真实数据仍待核验，S_bg／Q_cal未读取或评分。

下方为各轮试跑及已有生成Demo的历史说明。

## 当前：衰减试跑1000步已完成，最终拟合门槛通过

新独立试跑保持同8图／seed20261029／1000步／stateless BN／作者损失，后段学习率第801—1000步2e-5；用户已完成1000步，最终AP99.9659%、原固定gate通过，实测7.74分钟。输入／5检查点实际日程／初始与最终权重／AP重算核验通过。旧恒定对照38.0213%及原成果保留；衰减前轨迹已存在差异，不能从一次结果证明学习率是根因或真实检测提升。详见[第6天B4核验](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe_lr_decay/第6天B4真实试跑核验.md)。下方历史命令只核验完整缓存，不重新训练；正式G0/G1仍待准备和用户手动执行。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli lr-decay-probe-train --seed 20261029
```

准备时估计8—12分钟，实际7.74分钟，每200步保存；新配置／数值目录results/controlled_generation/learnability_probe_lr_decay，新模型目录models/controlled_generation/learnability_probe_lr_decay_20261029。首次从头初始化，不从旧800步恢复。第800步保存最后使用1e-4、下一步2e-5；本次无恢复，完整恢复路径仍未实测。完整结果存在时仅核验缓存。原资源门槛与B2历史准备信息见[准备与运行说明](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe_lr_decay/第6天B2准备与运行.md)。

仅核验输入或保存状态，可运行verify-lr-decay-probe、lr-decay-probe-status。以下保留前两轮及生成Demo历史说明，当前训练入口以本节为准。

本模块使用CPU可解释图像模拟；prepare／generate／demo等入口不训练。已完成24请求、第5天准备及用户首轮1000步，原门槛未通过，继续第6天归一化排障。正式G0/G1及新独立真实验证仍待完成。生成结论见[初步结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/可控生成初步结果.md)，历史准备条件见[第5天准备结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/training_preparation/第5天准备结果.md)。

## 首轮1000步已完成，当前先排查归一化

第6天A只读诊断已经完成：200／400／600／800步训练池AP均超过99%，最终1000步降至38.0213%，重建与分割损失同时退化。完整结果及两图见[第6天A诊断](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/probe_learning_diagnosis/第6天A可学习性诊断.md)。保留最终gatefalse，不取800步替代；下一项仅审阅后段学习率日程的单因素候选，尚未确认根因或开始新训练。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli diagnose-learning
```

diagnose-learning已有完整缓存时核验保护来源／代码／数组哈希并复用，不重推理。新诊断首次只对固定8张／5检查点进行eval／inference_mode，逐个mmap本地检查点，仅复制权重，不实例化或加载优化器；需≥3GiB可用RAM及≥2GiB空闲显存。图件重画入口为`python -m controlled_generation.probe_learning_figures`，只读保存数组。

**独立新试跑已由用户完成1000步，AP38.0213%，gate仍false**：同8样本／同种子／同1000步，两个网络BN采用track_running_stats=False、无运行缓冲量，让训练与单图eval使用一致的当前输入统计。从头初始化，初始参数核验与原试跑相同，不加载旧训练权重。输入／保存权重身份及AP重算通过，实测7.90分钟；尚不能进入G0/G1。下面命令只核验完整缓存，不重训。详见[新试跑说明](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe_batchstats/第6天归一化试跑说明.md)。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli batchstats-probe-train --seed 20261029
```

启动前需≥3.0GiB可用RAM，预计8—12分钟。每200步保存，断点恢复路径尚未实测。输出模型在models/controlled_generation/learnability_probe_batchstats_20261029，结果在results/controlled_generation/learnability_probe_batchstats；原试跑不覆盖。核验状态用verify-batchstats-probe和batchstats-probe-status，不训练。新权重必须按同一stateless BN结构加载；通过后G0/G1也需共同重新冻结该设置，不能只更改一组。

以下保留原首轮命令与历史结果，原命令不启动本次新试跑。

用户已完成以下首轮命令：实际约7.93分钟，原eval训练池像素AP2.0923%，门槛未通过。只读BN诊断见[核验结果](D:/华兴源创/实验/弱缺陷Demo/results/controlled_generation/learnability_probe/可学习性核验与BN诊断.md)。再次运行原训练命令只核验缓存，不生成新结果；下一轮独立协议和命令见上方，尚未开始正式G0/G1。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli probe-train --seed 20261029
```

固定8个训练侧样本、4个背景，FP32 batch1，共1000步。不是正式G1训练；当前采样先以正常亮度余量限定可行位置，320个请求／328次尝试均已核验，不根据检测分数选样。启动前需至少3.0GiB可用RAM；准备时约2.56GiB，请先手动释放部分内存。旧步速粗估8—12分钟，实际时长及恢复路径尚未实测。每200步保存，Ctrl+C丢失未保存进度；完整检查点同命令恢复，保存事务不完整时在任何更新前停止并提示查因。不自动扩步数或开始正式训练。

输出在models/controlled_generation/learnability_probe_20261029及results/controlled_generation/learnability_probe。完整训练后保存同8样本的训练池AP、loss及资源，只是拟合诊断。只查看准备／保存状态可运行：

```powershell
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli verify-probe
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli probe-status
```

prepare-training可核验并复用既有采样及8样本材料，不训练；参数身份变化时停止，不静默覆盖。今天已经完成准备，无需重新执行24请求或旧PatchCore。

## 查看已有成果

在Windows PowerShell中复制：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
Invoke-Item -LiteralPath '.\results\controlled_generation\figures\all_requests.png'
Invoke-Item -LiteralPath '.\results\controlled_generation\figures\control_pairs.png'
```

第一张展示全部24个请求，第二张展示预定的两类首对：原图、模拟图、请求区域、实际掩码、实际残差和冻结PatchCore热图。正常图及模拟图显示范围相同；残差统一[0,0.04]，热图统一[0,5.6253]，显示范围不是报警阈值。原整图阈值为1.8928574562072753。

## 复现生成与核验

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli prepare
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli demo
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli audit
```

prepare固定N_gen80／S_bg20／Q_cal40正常图路径，不打开旧测试图。只生成首4张N_gen背景上的24个请求。首次CPU生成与保存实测约4.20秒，启动、读图与复核另计；再次demo核验已有缓存，不覆盖身份不同的成果。更改核心渲染代码、请求或划分后，原缓存会拒绝复用；需要显式安排新试验，而不是覆盖旧结果。

## 自己指定位置与尺寸

可编辑[椭圆示例](D:/华兴源创/实验/弱缺陷Demo/controlled_generation/examples/mura_ellipse.json)或[线状示例](D:/华兴源创/实验/弱缺陷Demo/controlled_generation/examples/line.json)。256工作坐标中，center为[x,y]，ellipse的axes为两个半轴，line的length为中心线长度、width为宽度、angle_deg为角度。amplitude>0为变亮，<0为变暗；这是[0,1]RGB中加入的幅度，不是百分比反射率或物理亮度。

下面生成新单图，输出目录必须尚不存在：

```powershell
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli generate --spec '.\controlled_generation\examples\mura_ellipse.json' --background 'display/train/good/0191.png' --output-dir '.\results\controlled_generation\custom_ellipse_01'
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli generate --spec '.\controlled_generation\examples\line.json' --background 'display/train/good/0191.png' --output-dir '.\results\controlled_generation\custom_line_01'
```

当前只允许已固定N_gen／S_bg内的公开正常背景。Q_cal、旧测试图和任意外部图不会悄悄进入生成。输出为background.png、generated.png、requested_mask.png、effective_mask.png、effective_mask_png.png、arrays.npz和result.json。查看result.json中的valid与invalid_reasons：无效样本只供检查，不能作为合格异常训练。

float实际标签M_eff来自最大RGB残差>float32(1/255)。PNG量化后独立计算最大通道差>1码值的标签。二者在当前24例一致，其他参数下不保证一致；以后使用PNG训练时必须采用匹配PNG的有效标签并重新核验。

## 冻结检测器的响应与重画图

```powershell
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli predict
& '.\.venv\Scripts\python.exe' -m controlled_generation.cli figures
```

predict只加载旧AnoLCD N80 PatchCore库与原Vnew阈值，不建库、不校准、不训练、不评分真实测试图。首次实际28图响应约3.85秒，Python／CUDA启动与文件核验另计；再次调用先核验生成材料并复用已保存分数。figures只读保存图像和响应，输出PNG／SVG／PDF及科学源数据CSV。自定义单图生成目前不自动做模型推理，避免误把生成等同评价。

## 参考来源与局限

[AnoStyler官方源码](https://github.com/yulimso/AnoStyler)的形状模块及MIT许可固定于19a41c9965a9ce3f2057890850e50efe0bcb58d6，放在vendor/AnoStyler。两个blob请求实际调用msp_dot，再裁取轮廓并缩放到指定包围框。线状采用其中心、旋转和线段构造思路，改为显式参数和像素中心距离宽度规则；圆／椭圆／多边形及亮暗渲染为本项目实现。

没有运行AnoStyler的完整风格网络，也没有训练生成模型。当前Mura式场是在有限区域内加入平滑亮暗变化，区域边界仍存在截断；不是测得的屏幕物理模型。4个背景和12对请求不能代表整个屏幕缺陷分布，24/24整图报警不能称作真实召回100%或证明定位精准。所有后续DRAEM训练仍由用户手动启动。
