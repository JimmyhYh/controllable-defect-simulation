# Mura响应研究第5阶段D_fit只读评分准备与核验

日期：2026年10月29日。范围：独立评分模块、CPU元信息／身份／虚构标量与资源准备，**没有执行D_fit评分或统计拟合**。

本阶段CPU准备与独立新进程核验通过。固定60背景、7560请求及60基线共7620引用，对应7305唯一float RGB；与已评分Q200／F100的工作内容hash重合为0。若后续明确启动，每模型需新增7305次真实输入评分，两模型合计14610次；本次实际新增前向、作者构造探测、模型／tensor加载、数组读取、原图读取、生成、训练、校准及拟合均为0。

依据：[执行计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)、[冻结研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[D_fit完整输入及独立证据](D:/华兴源创/报告/Mura响应研究第3阶段独立身份适配与CPU取得结果.md)、[PatchCore正常工作点](D:/华兴源创/报告/Mura响应研究第4阶段PatchCore正常工作点与核验.md)、[G1正常工作点](D:/华兴源创/报告/Mura响应研究第4阶段G1正常工作点与核验.md)。私人设备、公开材料、个人学习与非商业研究；公司材料隔离。

## 一、本次核对了什么

复用已封存的inference静态检查器，在隔离子进程核preparation_v1；核正常准备、两模型最终资源文件字节hash、环境包元信息及各自正常工作点的完整清单、全部记录JSON与独立只读证明。没有调用旧score-normal／verify-normal，也没有重新取Q分位数、计数实际误报或打开正常热图。

两模型自己的新tau固定为：

| 模型 | 本轮保存tau | 正常工作点完整清单SHA256 |
| --- | --- | --- |
| PatchCore | 1.852951288223265 | 2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec |
| G1 | 0.6587551027536391 | 75dd6b85f50bd6ebc98591aa90496e6c74dd61e039905fccfecec299b0c7bcec |

同时核fit_inputs_identity_v2的封存JSON、14883文件严格目录清单／stat及已有全量数组独立核验证据，绑定其complete SHA256为3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593。核固定60背景和7560原序请求、全部引用与唯一内容索引的关系。当前不重新打开NPY／NPZ，不能把复用既有全量证据称为本次再次验证数组载荷；未来实际评分时，每份被评分RGB会先核文件字节hash及规范工作内容hash。

仅新增[fit_inference.py](D:/华兴源创/实验/弱缺陷Demo/mura_response/fit_inference.py)，没有修改inference.py、inference_runtime.py、generation_v2.py、inputs.py、旧模型或模块全局路径。原生构造、预测、状态签名辅助函数只在未来显式运行分支复用，CPU入口不导入模型包或数组包。

## 二、引用、去重与实际分母

| 项目 | 依据冻结元信息核对的数量 | 含义 |
| --- | ---: | --- |
| D_fit背景 | 60 | 统计背景数；采集物理分组仍未知 |
| 正常基线引用 | 60 | 每背景1份，后续同一热图供9个固定几何使用 |
| 扰动请求引用 | 7560 | 原网格全部保留，不筛弱档或零变化 |
| 全部引用 | 7620 | 60＋7560，不等于新增前向数 |
| 零实际变化请求 | 315 | 输入与对应背景相同，复用背景评分；不称缺陷FN |
| 非零实际变化请求 | 7245 | 弱于1/255的非零变化继续保留 |
| 测量未定义请求 | 0 | 此批既有测量状态，本次未重新测量 |
| 唯一float RGB | 7305 | 60基线＋7245非零输入，固定原索引顺序 |
| 与Q/F正常缓存内容重合 | 0 | 只比规范工作RGB摘要，没有打开缓存数组 |
| 后续新增输入前向／模型 | 7305 | 当前尚未执行 |
| 后续两模型新增输入前向 | 14610 | 两模型重复测同60背景，不是14610独立样本 |

PatchCore未来构造保留1次作者ones占位前向，独立计数，不算真实输入；G1没有此探测。当前准备期两者均为0。完整同身份fit评分缓存一旦存在，只读核验复用，新增前向为0；未知／部分状态停止，不自动补推或取成功交集。

协议也明确：若唯一输入与已评分正常缓存相同，须同规范RGB摘要、同模型／实现／FP32／预处理／状态／阈值绑定，并完整核对被引用正常记录和热图载荷后，才能只读引用，不重新前向。本批重合数为0，该分支不会被用来缩减当前7305预算。315个零变化属于D_fit内部去重，不是Q/F重用。

## 三、冻结的评分与未来CPU响应口径

直接消费fit_inputs_identity_v2的既有256×256×3／HWC／C连续float32缓存，保持[0,1]，不再除255、不量化、不经PNG回读或resize。每模型单独新进程，eval／FP32／batch1／workers0、4线程，关闭TF32及cudnn benchmark，构造seed42；不是新训练种子。

- PatchCore保留ImageNet归一化、N80只读库、最大patch原始分数和作者resize／sigma4热图；原生整图分数不替换为热图最大值。不建库、不执行作者pickle。
- G1仍为最终10000步重建／判别双权重，49个stateless BN（22＋27）、空buffer，无额外ImageNet归一化；保存原始float32异常softmax图及最大值。
- 模型初始完整签名须与其新正常工作点保存签名相同；每唯一输入后及封存前核全部参数／buffer／训练态／梯度签名不变。签名统一到JSON结构比较，保留内容和digest，不改变模型状态。
- 预测函数只接收RGB、模型及名称，不读M_geo、GT、残差或tau。保存原生热图、原生分数和完整引用，不在评分入口计算实际H_joint、拟合或曲线。

为下一项CPU后处理固定：H_alarm为原生score严格大于自身保存tau；H_joint还要求热图按C序展平的首个最大值位置在固定M_geo内。并列峰不能挑区域内的一个；PatchCore以自己的原生整图分数报警，不能用其热图max替代。另保留基线报警／峰位置、分数差及M_geo内响应差。60个基线供9几何复用不产生540独立正常图；零变化报警不称注入成功或缺陷FN，H_joint不是分割准确率或因果净POD。

本次仅用虚构Python标量／列表核严格大于、并列峰首个索引、区域外并列峰、PatchCore分数与热图max分离及引用算术，没有计算任何实际响应。统计拟合仍须另行准备并由用户手动启动；D_hold未解封。

## 四、独立协议、事务与停止规则

新准备目录为[fit_scoring_preparation_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scoring_preparation_v1)，仅7份JSON文件、约6.52MiB，包括来源、协议、唯一输入索引、全部引用、虚构标量核验、实际准备核验与complete。绑定新实现、冻结依赖实现、环境／模型字节身份、旧完整清单、两模型状态及各自原tau：

- 新实现SHA256：af72a578201e7359e309ccfaf1e5c107ae94ab05f294ca5e1af26748fa90dd99。
- 新protocol SHA256：0f27f9341f006d5d0b13472be16f0b4c0def36b07f6125564e8f14098ecd99a6。
- 新complete SHA256：e047ac4bbf81b9d67ce775c0866ae5a34f30a4f50e9485355a76b3923ad20a0c。

未来实际评分仅写新fit_scores_v1/PatchCore或fit_scores_v1/G1，不向已封存准备目录追加文件。构造前和每次唯一输入前写inflight；只有成功写出热图和记录才算完成。资源／输入／参数或BN异常及未知前向保留failure／inflight并停止，不删标记、降规格、换样、换seed、改幅度、自动续推、重校准或更换模型。缺失／部分缓存不重生成。

完整同身份缓存须严格核目录白名单、全部来源／输出hash、原序、每一原生float热图的shape／dtype／有限性及G1概率范围和原生max、保存状态及计数；只读复用，不重新构造、前向或取分位数。本次没有调用score或verify-scores：GPU实际导入／构造、数组载荷消费、参数／BN实际一致性和完整评分缓存复用尚待对应实际运行确认，不以静态通过冒充GPU兼容。

## 五、资源与运行说明

准备时实测可用RAM6.517578GiB、D盘空闲113.305687GiB；RTX5060Laptop总显存8151MiB、空闲7899MiB，驱动621.01。满足当前CPU2GiB、未来GPU启动4.5GiB RAM／4GiB空闲显存／D盘40GiB余量的门槛；未来启动时重新核，不自动降低规格。未安装或下载新资源。

按既有Q段平均（包括逐图状态核对／IO）的线性参考，7305输入约为PatchCore5.15小时、G1 1.95小时；不是本次实测，不是纯GPU前向性能，也不保证能按正常图线性外推。两模型原始float热图载荷约3.567GiB，另有NPY头、记录JSON、文件系统及事务余量，仍遵守40GiB空闲门槛。新评分路径的峰值／吞吐／真实连续运行及恢复未实测；未知事务不恢复。

已实现并通过源码语法、CLI及CPU元信息／标量核验的入口：

- prepare：仅CPU准备；完整准备缓存只读核验。
- verify-preparation：仅CPU准备身份核验。
- inspect：只读资源／元信息。
- score --model PatchCore或G1：未来显式单模型评分；本次不执行。
- verify-scores --model PatchCore或G1：未来完整评分缓存只读核验；没有缓存时停止，不创建评分或补推。

首组PatchCore人工评分命令（**本次没有启动**）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.fit_inference score --model PatchCore
```

仅查看已封存准备：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.fit_inference verify-preparation
```

不自动串联G1、CPU响应后处理或拟合。所有训练及统计拟合仍由用户手动启动。

## 六、独立核验证据与审阅

prepare成功后，另起新进程运行verify-preparation，退出码0，实际返回：

```json
{
  "status": "FIT_SCORING_PREPARATION_CACHE_VERIFIED",
  "output": "D:\\华兴源创\\实验\\弱缺陷Demo\\results\\mura_response\\semu_applicability_v1\\fit_scoring_preparation_v1",
  "complete_sha256": "e047ac4bbf81b9d67ce775c0866ae5a34f30a4f50e9485355a76b3923ad20a0c",
  "denominators": {
    "backgrounds": 60,
    "baseline_references": 60,
    "request_references": 7560,
    "all_references": 7620,
    "unique_float_RGB": 7305,
    "zero_actual_change_requests": 315,
    "nonzero_actual_change_requests": 7245,
    "undefined_measurement_requests": 0,
    "unique_overlap_with_Q_F": 0,
    "new_input_forwards_per_model": 7305,
    "new_input_forwards_two_models": 14610,
    "PatchCore_author_placeholder_forwards_separate": 1,
    "G1_author_placeholder_forwards_separate": 0,
    "request_variants_are_not_independent_backgrounds": true,
    "preparation_actual_model_input_forwards": 0
  },
  "boundary": {
    "source_decodes": 0,
    "array_opens": 0,
    "model_loads": 0,
    "tensor_loads": 0,
    "model_input_forwards": 0,
    "author_placeholder_forwards": 0,
    "parameter_updates": 0,
    "render_calls": 0,
    "real_calibrations": 0,
    "statistical_fits": 0,
    "model_packages_loaded": [],
    "array_packages_loaded": [],
    "original_file_opens": 0,
    "blocked_attempts": [],
    "D_hold_S_bg_Q_cal_old_test_GT_reads": 0,
    "unrestricted_pickle_deserializations": 0,
    "counter_semantics": "forward attempts before call; completed records prove success"
  }
}
```

父进程及其显式静态子进程均未读取真实输入数组／概率／原图或GT，没有导入Torch／模型、加载tensor、前向、生成、实际校准或拟合，权重仅字节hash。父进程也未导入NumPy／SciPy／PIL；被复用的封存resources隔离子进程可导入NumPy／SciPy，以固定RNG复核角色及请求元信息，但不重跑人工数组数值实验或读取真实数组载荷，不能把父进程包计数当作整个进程树的包计数。未来score内部才可读取本阶段固定RGB及自有热图；D_hold／S_bg／Q_cal／旧测试及GT始终没有读取入口。

本次一次新上下文只读审阅发现1项Important：原生状态签名的buffer描述是元组，保存JSON读回为列表，直接比较会让有BN buffer的PatchCore在构造后误失败。在封存准备前，仅新模块统一签名到JSON域；虚构嵌套签名已先确认原比较不等，再核规范后相等，初始／逐图／最终统一使用。没有为此运行模型，也没有改冻结runtime或BN。Critical／Minor无，未追加第二轮审阅或软件测试套件。

审阅未判断实际Torch／FAISS导入／构造兼容、真实状态不变、GPU峰值／7305输入耗时与实际缓存载荷正确性；当前限定只读元信息不能回答这些问题，因此保留给未来授权运行及严格载荷核验，不将它们列为已通过。也没有取得缺陷检出、SEMU适用性或真实屏幕收益的结论。本轮只完成准备，下一项唯一指令维护在总计划第六节。

交付前另作只读元信息核对：本轮开始时59份受保护源码／元信息／报告的SHA256均未变，7个保护目录的路径／尺寸／修改时间stat清单签名均未变，包含旧失败、完整D_fit输入、正常准备／输入和两模型正常评分。新准备7文件的全部SHA256及实现绑定一致，报告独立核验JSON与实际返回逐字段一致，所有本地链接存在；总计划仅一份最新摘要及一条当前指令。fit_scores_v1仍不存在，没有将准备数量记为实际模型评分。该交付核对没有打开图片、数组或权重载荷，不替代未来实际评分准入与载荷验证。

