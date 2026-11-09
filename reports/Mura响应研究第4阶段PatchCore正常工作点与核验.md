# Mura响应研究第4阶段PatchCore正常工作点与核验

日期：2026年10月29日。范围：第4阶段首模型PatchCore新正常工作点；G1未启动。

**发现用户已手动完成的完整PatchCore缓存后，本轮只执行只读核验，没有再次调用评分入口。独立verify-normal通过：新tau=1.852951288223265；Q同池FP10／TN190（分母200），F检查FP6／TN94（分母100），F误报率6%，预定Wilson双侧95%区间约2.7786%—12.4768%。本轮新增模型加载、真实输入前向、ones探测、实际校准与阈值重算均0。**

依据：[冻结研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[总执行计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)、[封存工作点准备](D:/华兴源创/报告/Mura响应研究第4阶段正常工作点准备.md)、[共享正常输入取得与核验](D:/华兴源创/报告/Mura响应研究第4阶段CPU正常输入取得与核验.md)。私人设备、公开资源、个人学习与非商业研究，公司材料隔离。

## 一、完整缓存复用与来源

首次检查即发现PatchCore完整目录、complete存在，且无failure／inflight；G1目录不存在。因此没有运行score-normal、不重复300次评分，也不从Q分数重新取分位数；用已实现verify-normal --model PatchCore在新进程只读核验既有结果，退出码0。

- 正常工作点protocol：`ea64281c4ca714f30d557dc7215a57805329b61f6d8f9ba5e95cbe0903bfe4d7`。
- 共享normal_inputs完整清单：`bcccbbf1c2fa1001d3459bf9944bdd5a6ad934f57104ac05babe330b591de9da`。
- PatchCore评分complete：`2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec`。
- 已有评分verification：`7ff7ae5c239a59b3c71c8724d990070c786263412ed832acb64c4446e7abcc61`。
- 已有result：`8e35071bca8616702faf69fc3b3f0db5014426cb316ae4201ac376948ce807d4`。

准备与共享输入完整核验、固定作者源码／环境及ImageNet V1／N80库字节身份由既有显式核验链检查；没有加载模型来“再验证”输出，没有解析或执行作者pickle。D_fit只读取既有来源／完整性JSON元信息，不打开其输入数组，D_hold／S_bg／Q_cal／旧测试／GT及其他旧概率均未读取。

实际模型仍是原N80、819个1024维向量、patchsize3／coreset0.01／1近邻PatchCore。保留原RGB256 float输入和ImageNet归一化，eval／FP32／batch1／workers0。整图分数是作者最大平方L2最近邻patch分数；256热图保留作者双线性缩放及sigma4 Gaussian处理，无额外平滑、拉伸或跨池归一化。**tau不是[0,1]概率阈值，整图分数也不是平滑后热图max。**

## 二、实际工作点及分母

本模型只消费已保存正常RGB，固定Q200原序后F100原序，各真实输入一次。Q完成后按q0.95／linear保存新tau，严格score>tau；之后才进入F100，没有用F改阈值、选分位数或改分数，旧tau保持原状。

| 来源 | 正常图分母 | FP | TN | 正常误报率 | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| Q200 | 200 | 10 | 190 | 5% | 定阈值同池计数，不是独立验证 |
| F100 | 100 | 6 | 94 | 6% | 固定tau后的正常误报检查 |

新tau精确保存为`1.852951288223265`。Q分数非全常量，tau不在Q观测分数极值上，没有因工作点退化换规则。F100预定Wilson双侧95%区间是[0.027786123963188192, 0.1247681544589592]，z=1.959963984540054；沿用保存结果和核验，不做阈值搜索或重新校准。

F区间包含5%，但100张正常图仍有明显不确定性，不能据6/100保证总体误报率为5%或6%。该区间按预定二项近似计算，不覆盖未知采集组相关或物理面板分组。Q与F合计仅300张唯一正常背景；构造探测不是额外正常图，未来第二模型也不能把同300图再次累计成独立600图。

这里没有真实异常或数字扰动评价，不能据此计算／宣称异常召回、AUROC、像素AP、POD曲线、SEMU预测能力或真实屏幕收益。已有正常工作点是后续冻结响应研究的基础，不是缺陷检测效果验收。

## 三、已有运行计数与只读核验新增计数

| 操作 | 已完成运行的保存计数 | 本轮只读核验新增 |
| --- | ---: | ---: |
| 模型加载 | 1 | 0 |
| 参数tensor加载 | 1 | 0 |
| 作者构造ones探测 | 1 | 0 |
| Q200＋F100真实输入评分 | 300 | 0 |
| 实际Q校准 | 1 | 0 |
| 原图打开／解码 | 0／0 | 0／0 |
| 参数更新／训练 | 0 | 0 |
| 扰动render／统计拟合 | 0／0 | 0／0 |
| 实际tau重新取分位数 | 首次Q完成时取1次 | 0 |

Q/F原图在前一CPU取得阶段已各解码一次，本模型及本轮核验只读规范缓存，不重开原图。已有运行的model_input_forwards=300均有完整提交记录；model_input计数是前向尝试计数，只有本次完整缓存和全部提交闭合才能将这300称为成功评分。不存在未明inflight，不把部分尝试当完成。

已有运行保留作者单张ones特征维度探测及BN初始化语义，探测1次与真实输入300次分别记。状态快照在构造、加载和eval之后建立；所有300条记录state_unchanged为true，保存的前后完整状态签名一致：
- 参数SHA256：`6372d614e547b82ad3683dab5f7ab453897c13d765c73871fb26305f1a4e4587`。
- state_dict SHA256：`1a33eb6df3f1ea53b04d7913bf4039dbc8f13db5f5ed7c9263a39afc7f9c25f9`。
- 159个命名buffer记录，训练态模块为空、无梯度；159不是BN层数。

不把“作者初始化前BN从未变化”作为结论；保留的是原构造语义，评分阶段未改变参数或buffer。此次只读核验依据保存签名及每图记录，不重新加载参数或执行前向来生成新状态。

独立核验先核正常输入，再核完整评分目录／全部产物hash、300份热图dtype／shape／有限性和记录身份、固定分数与保存tau下FP/TN及区间、前后状态及保存操作计数。热图仅本组PatchCore缓存；未读其他模型旧概率。目录虽通用命名probabilities，其中PatchCore保存的是连续分数热图，不把它当单位概率。

本轮array_opens=1500：正常输入文件hash300次＋RGB加载300次、PatchCore热图封存hash300次＋逐图记录对应hash300次＋热图加载300次。它们不是1500张独立图，也不是新增前向；原图计数与模型计数均0。实际tau_recomputed=false。

## 四、实际时间与资源范围

以下来自用户已完成运行的封存记录，不是本轮只读核验的耗时：

| 指标 | 保存实测 |
| --- | --- |
| Q200评分＋状态核对＋缓存 | 507.851043秒，约8.46分钟 |
| Q段每输入参考平均 | 2.539255秒 |
| 模型所属事务至验收字段 | 839.295159秒，约13.99分钟 |
| PyTorch GPU allocated峰值 | 366.806152MiB |
| PyTorch GPU reserved峰值 | 410.000000MiB |
| 启动可用RAM | 9.652290GiB |
| 301次资源采样中的最小可用RAM | 8.225193GiB |
| 启动空闲显存 | 7899MiB |
| 本组最终缓存实际大小 | 约0.073615GiB |

Q计时从模型加载后到最后Q缓存提交，包含逐图参数／buffer核对和IO，排除取分位数；**2.539秒不是纯GPU网络前向耗时**。事务总计时包含构造、300图评分及状态／来源核对，排除之前CPU输入完整核验和之后封存文件hash。不能用总时长减Q时长冒称精确F100纯评分时间。

GPU峰值在CUDA准入后、模型创建前重置，覆盖构造及300输入；不包含更早解释器导入／CUDA准入上下文，也不是显卡全设备总占用。最小可用RAM是系统余量，不能当本进程RAM峰值。本组确认了当前PatchCore路径能完成一次运行，不等于测过G1兼容或所有设备／后续负载。

## 五、独立只读核验证据

[PatchCore完整结果](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore)严格606文件：300热图NPY、300逐图JSON、5份顶层元信息JSON和最后complete。全部目录、hash和记录闭合，未取成功子集、未另开重试目录。

产物：[新Q阈值及计时](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/q_result.json)、[Q／F计数和区间](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/result.json)、[状态与运行资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/verification.json)、[来源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/sources.json)、[完成清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/complete.json)。

独立进程实际返回如下，退出码0：

```json
{
  "status": "NORMAL_MODEL_CACHE_VERIFIED",
  "model": "PatchCore",
  "threshold": 1.852951288223265,
  "Q": {
    "FP": 10,
    "TN": 190,
    "normal_denominator": 200,
    "FPR": 0.05
  },
  "F100": {
    "FP": 6,
    "TN": 94,
    "normal_denominator": 100,
    "FPR": 0.06
  },
  "F100_Wilson95": [
    0.027786123963188192,
    0.1247681544589592
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
  "model_complete_sha256": "2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec"
}
```

核验命令已实际运行，可只读复用：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.inference verify-normal --model PatchCore
```

本轮不修改冻结双模块／配置／模型／旧阈值、已有评分缓存或历史结果，只整理本报告及总计划。完整缓存只读复用；若以后身份改变、出现failure／inflight或目录不完整，保留查因，不用score-normal补推或删除标记。

交付前对49项受保护源码／元信息hash及相关目录stat核对一致；包含共享正常输入305文件和PatchCore606文件，旧D_fit／失败现场仍只核stat而未重读数组。另完成一次独立只读结果审阅，无Critical／Important／Minor项；审阅核对300份逐图JSON、身份、分母、计时范围和计划交接，没有另开数组／原图／模型载荷或重新计算tau。载荷正确性沿用本轮已完成的独立CLI证据，参数内部状态采用保存签名；不将审阅通过外推为全项目审计、G1兼容或缺陷收益。

## 六、下一项与结论边界

PatchCore本轮正常工作点已核验，可为后续预定响应研究使用新冻结tau；本轮没有读取D_fit刺激数组或作任何缺陷评分。G1正常工作点、D_fit响应及统计拟合、D_hold留出预测均尚未进行。

下一项只按相同正常输入和预定规则完成G1正常工作点，须另行明确启动，不复用PatchCore的tau、不自动推进D_fit。唯一可直接发送指令维护在总计划第六节；所有训练及统计拟合仍由用户手动启动。

