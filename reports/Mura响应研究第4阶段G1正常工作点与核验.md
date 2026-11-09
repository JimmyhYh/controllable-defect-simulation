# Mura响应研究第4阶段G1正常工作点与核验

日期：2026年10月29日。范围：G1新正常工作点评分与独立核验，不进入D_fit。

**本次按授权仅运行G1一次Q200→F100，300个真实输入各评分一次，完成封存后另起进程只读核验通过。新tau=0.6587551027536391；Q同池FP10／TN190（分母200），F检查FP7／TN93（分母100），F误报率7%，预定Wilson双侧95%区间约3.4319%—13.7495%。两网络参数与49个stateless BN／空buffer逐图不变；没有训练、建库、扰动生成或统计拟合。**

依据：[冻结研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[总执行计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)、[正常工作点准备](D:/华兴源创/报告/Mura响应研究第4阶段正常工作点准备.md)、[共享正常输入](D:/华兴源创/报告/Mura响应研究第4阶段CPU正常输入取得与核验.md)、[已核验PatchCore工作点](D:/华兴源创/报告/Mura响应研究第4阶段PatchCore正常工作点与核验.md)。私人设备、公开材料、个人学习与非商业研究，公司材料隔离。

## 一、来源、身份与实际执行

初始检查G1目录不存在，PatchCore已有完整结果，准备与共享输入身份通过。启动前可用RAM约8.05GiB，空闲显存7899MiB，满足GPU启动4.5GiB RAM／4GiB空闲显存／40GiB磁盘门槛。实际调用已实现score-normal --model G1一次，没有调用PatchCore评分入口，没有更换权重、输入或配置。

- normal_preparation protocol：`ea64281c4ca714f30d557dc7215a57805329b61f6d8f9ba5e95cbe0903bfe4d7`。
- normal_inputs完整清单：`bcccbbf1c2fa1001d3459bf9944bdd5a6ad934f57104ac05babe330b591de9da`。
- 新G1完整清单：`75dd6b85f50bd6ebc98591aa90496e6c74dd61e039905fccfecec299b0c7bcec`。
- 新G1 verification：`1e3043be8415047a5d9baaf3fdbd18fc6db6625c27ff811f875fd4b278c55218`。
- 新G1 result：`5ac4bfc0e28904fab8fe1b345dc6341f54e188acd6c7978d984c4ca5ecb42072`。
- 原PatchCore完整清单仍为`2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec`。

G1仍为20261030正式训练的最终10000步作者重建／判别网络，仅完整字节hash并按已绑定路径加载双权重。运行时strict加载、FP32检查和最终参数hash核对通过：
- 重建参数：`fd72e5431eb1447250cee7dc198e114ded73c84230f82c83f037a0548f2273ba`。
- 判别参数：`c2f2f0d851d913f6d1a473883dbbad9056882153bc20bd34d4bbafd8b3a519d8`。
- 重建state_dict：`f2e398ee01d576199f6cacc7c175824df9f1520a91d66583dca66083449e9394`。
- 判别state_dict：`6fd6cc4666b8c25c9bb3d4a9a41450940cf2e846b8ee1eda4584ed50596e9749`。

输入仅来自已保存Q200／F100规范RGB256 float32/255，没有额外ImageNet归一化、增广或重新量化。保持作者结构、22＋27共49个stateless BatchNorm2d、running buffers为None、空named_buffers、eval／FP32／batch1／workers0、4线程、关闭TF32及cudnn benchmark。预测函数只消费RGB与模型，不接收GT、区域或阈值。

运行在加载后及每个输入后核stateless BN与参数／状态；两网络保存的before／after签名完全一致、named_buffers均为空、无训练态模块或梯度。只读核验依据保存证据，不重新构造网络统计BN或加载参数。每个输入各执行一次重建和判别，合称一次G1输入评分；没有将内部两次网络调用算成600张图，也没有作者ones构造探测。

## 二、新工作点、检查分母及限制

严格按固定Q200原序评分200张，以q0.95／linear／严格score>tau保存G1自己的新tau，之后才开始F100。没有复用PatchCore tau或旧G1 tau，没有根据F改变阈值或评分口径。

| 来源 | 正常图分母 | FP | TN | 正常误报率 | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| Q200 | 200 | 10 | 190 | 5% | 定阈值同池计数，不是独立验证 |
| F100 | 100 | 7 | 93 | 7% | 固定tau后的正常误报检查 |

新tau精确值为`0.6587551027536391`。Q分数非全常量，tau不在Q观测极值或概率0／1端点上；不因结果更改q或换分数。保存热图为原始float32异常通道softmax，256×256且有限、位于[0,1]；整图分数严格为原生图最大值，没有平滑、池化、拉伸或resize。

F100 Wilson双侧95%区间为[0.03431926106727268, 0.137495147390735]，沿用预定z=1.959963984540054。它是预定二项近似，不涵盖未知采集组相关；100张正常图不能保证总体误报率为5%或7%。F没有用于重新校准。

两模型各使用同一Q200／F100，历史PatchCore工作点保留。至此600次模型输入参考仍只对应300张唯一正常背景，F检查仍是同100图，不能累加成独立200图或据此作模型能力排名。两者原始分数尺度不同，各自tau只用于自身固定评分规则。

本阶段只有正常工作点，没有对D_fit扰动评分，没有异常召回、AUROC、像素AP、POD曲线、SEMU预测或真实屏幕效果的新结论。G1在相似参数化模拟上训练过的渲染器偏好限制仍保留。

## 三、实际运行与独立核验的计数

| 操作 | 本次实际G1运行 | 随后独立只读核验新增 |
| --- | ---: | ---: |
| G1管线模型加载 | 1 | 0 |
| 最终参数tensor文件加载 | 2 | 0 |
| 真实输入评分 | 300 | 0 |
| 作者ones探测 | 0 | 0 |
| 实际Q校准 | 1 | 0 |
| 原图打开／解码 | 0／0 | 0／0 |
| 参数更新／训练 | 0 | 0 |
| render／统计拟合 | 0／0 | 0／0 |
| 已保存实际tau重新取分位数 | 首次Q完成时取1次 | 0 |

300个输入都有完整提交记录，按Q200后F100，逐图state_unchanged为true；完成事务没有failure／inflight，不能用尝试计数替代部分事务的成功数。新目录严格606文件：300热图NPY、300逐图JSON、5份顶层元信息JSON及complete，实际约0.073552GiB。

保存的运行期verification boundary在最终封存hash前记录array_opens=1200；运行入口结束时1500。独立核验array_opens=1500，涵盖共享RGB缓存核验600次、热图封存hash300次、逐记录热图hash300次及热图加载300次。均是缓存访问，不是新增原图解码、独立样本或模型前向。

没有读取D_fit或旧N原图／数组，D_hold／S_bg／Q_cal／旧测试／GT及其他旧概率均未打开。PatchCore仅核已有工作点元信息／complete身份，没有重新前向、重取阈值或打开其热图。

## 四、实际时间与资源

| 指标 | 本次保存实测 |
| --- | --- |
| Q200评分＋逐图状态核对＋缓存 | 192.137966秒，约3.20分钟 |
| Q段每输入参考平均 | 0.960690秒 |
| G1所属事务至验收字段 | 295.614997秒，约4.93分钟 |
| PyTorch GPU allocated峰值 | 675.207520MiB |
| PyTorch GPU reserved峰值 | 756.000000MiB |
| 启动可用RAM | 8.026203GiB |
| 301次采样中最小可用RAM | 6.638119GiB |
| 启动空闲显存 | 7899MiB |
| 启动D盘空闲 | 113.383350GiB |

Q计时从加载后到最后Q缓存提交，包含参数／BN核对及IO，排除分位数计算；0.960690秒不是纯GPU网络延迟。事务计时包含构造、300输入及状态／来源核验，排除之前正常输入完整核验和之后封存文件hash；不把它与Q相减当作精确F纯评分时间。

GPU峰值从CUDA准入后、构造前重置起，覆盖构造和全部输入，不含更早解释器导入／CUDA准入上下文；是PyTorch分配／保留计数，不是显卡总占用。最小可用RAM是系统余量，不是进程RAM峰值。此次确认G1在当前设备和输入上完成一次运行，不保证其他设备或负载，亦不代表缺陷检测效果。

## 五、独立只读核验与产物

完整封存后另起新进程执行已实现verify-normal --model G1，退出码0。核完整目录／全文件hash、来源与原序、全部300热图shape／dtype／有限性／[0,1]、原生max与保存score精确一致、固定tau计数与区间、逐图和前后状态证据；不再次加载模型、前向或从Q重算tau。

产物：[G1目录](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1)、[新Q阈值及计时](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/q_result.json)、[Q／F计数与区间](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/result.json)、[状态与资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/verification.json)、[来源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/sources.json)、[完整清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/complete.json)。

独立核验实际返回如下，tau_recomputed=false：

```json
{
  "status": "NORMAL_MODEL_CACHE_VERIFIED",
  "model": "G1",
  "threshold": 0.6587551027536391,
  "Q": {
    "FP": 10,
    "TN": 190,
    "normal_denominator": 200,
    "FPR": 0.05
  },
  "F100": {
    "FP": 7,
    "TN": 93,
    "normal_denominator": 100,
    "FPR": 0.07
  },
  "F100_Wilson95": [
    0.03431926106727268,
    0.137495147390735
  ],
  "tau_recomputed": false,
  "boundary": {
    "source_decodes": 0,
    "array_opens": 1500,
    "model_loads": 0,
    "tensor_loads": 0,
    "model_input_forwards": 0,
    "author_placeholder_forwards": 0,
    "parameter_updates": 0,
    "render_calls": 0,
    "real_calibrations": 0,
    "statistical_fits": 0,
    "original_file_opens": 0,
    "original_unique_paths": 0,
    "blocked_attempts": [],
    "model_packages_loaded": [],
    "Q_F_acquisition_only_if_explicit": true,
    "D_hold_S_bg_Q_cal_old_test_GT_reads": 0,
    "unrestricted_pickle_deserializations": 0,
    "forward_counter_semantics": "attempts before call; only committed rows are completed outputs"
  },
  "model_complete_sha256": "75dd6b85f50bd6ebc98591aa90496e6c74dd61e039905fccfecec299b0c7bcec"
}
```

已实际运行的只读核验命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.inference verify-normal --model G1
```

完整同身份缓存只读复用，未知／部分事务保留查因，不用score-normal补推或删标记；本轮未修改封存代码／协议、权重、旧tau或旧成果。两个模型各自complete独立封存，normal_scoring_v1是分模型容器，不另向已封存子目录追加汇总文件。

## 六、进度与下一项边界

至此第4阶段两模型新正常工作点分别完成核验：PatchCore及G1各有独立tau和Q／F结果，同一300正常背景不重复累计。当前只完成G1任务；没有自动进入D_fit评分、开发或拟合。

下一项仅按既有计划准备第5阶段D_fit只读评分的独立模块／协议及已实现入口，不修改现有已封存inference双模块或generation_v2，不重新生成／改阈值。实际评分与CPU统计拟合仍分阶段，拟合和所有训练由用户手动启动；D_hold必须等待统计模型及分析口径冻结才解封。唯一可直接发送的下一步指令维护在总计划第六节。

## 七、交付前身份与独立审阅

交付前只读对比本轮开始时的快照，原51份受保护源码／元信息／报告的SHA256全部不变，5个保护目录的路径／尺寸／修改时间清单签名全部不变，涵盖旧失败、D_fit完整输入、正常准备、共享正常输入及原PatchCore评分。G1仍为606文件，300条提交记录与共享输入原序、角色和工作RGB身份一致；q_result提交先于首个F记录。报告中的独立verify返回与本次实际返回一致；总计划仅一份最新摘要、一条第5阶段CPU准备指令，链接核对通过。该交付核对只读JSON／文档／源码及目录stat，没有读取数组或权重载荷，不能将其称为第二次全载荷核验。

随后进行一次新上下文只读审阅，未发现Critical／Important／Minor问题。审阅核对结果、来源、源码口径、时序、计数、资源范围及计划状态；未再次执行项目入口、读取数组／权重或计算分位数，载荷完整性依赖前文已实际完成的独立CLI核验证据。物理采集独立性、总体误报、缺陷检出、SEMU预测、D_fit／D_hold效果、模型优劣或因果来源不在这些证据可判断范围内，均不作结论；若把它们视为已证实，会超出本阶段正常数据与有限分母能够支持的范围。

