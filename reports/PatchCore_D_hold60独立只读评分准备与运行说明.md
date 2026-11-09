# PatchCore D_hold60独立只读评分准备与运行说明

日期：2026年11月4日。私人设备、公开数据、个人学习与非商业研究，公司材料隔离。

## 当前结论

独立patchcore_hold_scoring已实现，prepare与另进程verify-preparation均通过，六文件准备缓存已封存。**本次只有CPU元信息、来源hash、语法与人工标量核验，没有读原图／输入数组／热图／概率载荷，没有导入NumPy／SciPy／PIL或Torch／模型包、解析模型tensor、实例化模型、前向、生成、建库、训练、校准或统计拟合。** 模型材料只做字节hash，没有执行pickle。当前独立根下只有preparation，没有run。

独立新上下文只读代码审阅：Critical0／Important0／Minor0。它支持准备交付，不代表冷启动、模型加载、GPU评分、实际中断行为或检测效果已验证。本次不自动评分，也没有修改任何旧模块、全局路径、作者源码、环境、数据、权重、阈值或成果。

## 冻结身份与元信息复用结果

[完整D_hold输入及独立核验证据](D:/华兴源创/报告/PatchCore_D_hold60_CPU输入取得与独立核验结果.md)与[输入run complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/complete.json)绑定，SHA256=5cbc1351afe334cefd23486aa35efd2143a9e38def506f385876c3e48d31e355。本次消费原完整核验记录与报告节选，核其身份及保存语义，没有重新打开输入数组冒充又一次实际数组核验。

六个D_fit拟合、全部变换及p0=0.4109788359788335保持，拟合run complete=983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649。本次不计算或更新这些统计量，也不把其作为检测器参数。

原PatchCore V1／N80／vendor／预处理、作者原始评分及tau=1.852951288223265绑定；既有正常及D_fit两组保存的参数／BN完整state签名前后相同。参数签名6372d614e547b82ad3683dab5f7ab453897c13d765c73871fb26305f1a4e4587，含buffer的state_dict签名1a33eb6df3f1ea53b04d7913bf4039dbc8f13db5f5ed7c9263a39afc7f9c25f9。它们来自保存元信息，当前没有加载参数重新计算。解释器及依赖只用安装元信息核对，实际运行兼容尚未实测。

固定7116个留出工作RGB内容hash与既有正常300、D_fit7305逐项集合比对，均重合0。既有参数／BN一致不等于不同RGB可复用，所以**旧输出复用0，新增真实输入前向预算7116，作者ones构造探测另计1**。7620引用不是7620次前向，更不是7620独立背景。

通用复用必要条件已明确：工作RGB及保存输入字节身份、完整参数／BN状态、预处理、评分定义和tau全部一致，且原缓存完整、来源闭合。本固定池没有符合内容条件的旧输出，因此运行不实现额外读取旧热图的旁路；若未来发现重合或来源变化，停止审阅，不临时改变复用策略。自己的完整同身份评分缓存则只读核验，不再次构造模型或前向。

## 全部分母和未来运行口径

固定60基线＋7560请求＝7620引用，内容去重后7116唯一RGB。全部请求保留：504零实际变化／7056非零／其中2016弱非零空诊断M_eff／未定义0。零变化只引用背景，没有删成缺陷FN；不按M_eff删弱档，基线分母仍60。

未来显式run将：

1. 先核本准备、完整输入及原来源；RAM至少4.5GiB／D盘至少40GiB，构造前另由原运行时核CUDA可用及至少4GiB空闲显存；不CPU回退、不降低规格。输入过程中沿原至少2GiB可用RAM／40GiB磁盘条件。
2. 复用原inference_runtime.build_model，显式WRN50-2／本地ImageNet V1／819×1024的N80 FAISS库；原作者单ones探测及构造BN初始化保持，seed42，随后eval／FP32／batch1／workers0。不以导入改写、预热或更改TEMP／TMP作前置。
3. 只按固定唯一字典原序读取7116份保存float RGB，各输入只评分一次。校验字节hash及规范内容／dtype／shape／范围，不PNG回读、量化或再次除255。仅RGB传入原predict_rgb；区域、标签、拟合模型和阈值都不传给预测函数。
4. 保留原最大patch异常分数与作者float32热图，不能用热图最大值替代整图分数；不额外平滑、拉伸、改分辨率或输出尺度。每图后核完整参数／BN／训练模式／梯度签名与初始状态一致。
5. 在自有独立run记录构造、每图inflight、原生热图／分数及完整引用，只有输出与状态全部闭合才complete。只保存评分，不计算H_joint、配对差、SEMU预测、Brier／log loss或bootstrap。原tau只绑定，不重取分位数。
6. 已有完整同身份run先全量只读verify，不创建模型；部分／未知／failure／inflight停止保留，不删标记、续推、补推或取成功交集。未知前向不自动重复。健康事务仅在该项输出与状态已明确保存后清除自己的inflight。
7. 完整后另进程verify只读核全部记录／热图、引用、来源hash与状态，不读输入RGB或模型、不前向、不重新校准／拟合。

后续CPU响应整理与留出预测评价仍需另行准备，不会由run或verify隐式执行。G1刺激／拟合及真实评价未执行，当前没有SEMU留出预测价值结论。

## 实际静态核验、资源及审阅边界

7716项绑定来源只做允许的JSON／源码／资源文件hash或目录stat；输入NPY、旧热图等载荷没有打开。10项人工标量核验包含相同身份可复用、逐项改变六个复用条件均拒绝、分数等于tau不报警／严格大于才报警、引用与前向计数区别。没有实际统计拟合、阈值搜索或检测响应评价。

准备实测可用RAM6.860279083251953GiB、D106.31881713867188GiB，满足本次CPU2GiB／40GiB门槛。这是系统余量采样，不是进程峰值；本次没有调用CUDA探测显存，未来GPU条件必须在实际运行时重新核对。

历史D_fit完整评分事务约11434.136秒，7303新增输入＋2复用。按本轮7116新增输入简单线性参考约3.09小时；数据、IO、来源检查与系统负载可改变耗时，不能保证。新增原始float32热图净载荷约1.7373GiB，另有JSON、事务与校验余量，仍保持40GiB空闲门槛。未来run保存含构造和全部输入的CUDA allocated／reserved峰值及系统余量采样，不能混作全设备峰值或CPU进程峰值。

预封存静态查因曾发现新消费者误将旧报告“总体分母节选”当完整分项，以及误写原精度枚举；依据报告原schema和原协议fp32修正新模块，随后来源与人工标量核验通过。旧报告、配置及缓存均未改，也没有为查因读取图像／数组或启动模型。

对审阅未执行项的判定：冷启动／审计兼容、GPU、加载和7116前向、真实数组／热图内容、实际耗时／内存与中断行为超出本次授权，继续未知；代价是静态准备通过不能保证下一次GPU事务成功，实际遇到问题须保留查因，而非重跑救结果。已创建准备缓存的独立运行核验随后实际通过，见下方；它不是评分成功。

用户明确不建过程表／软件测试或提交流程，因此采用独立科学协议、人工标量与实际只读核验结果，不创建技能默认ledger／测试套件／Git流程。所有训练及统计拟合仍由用户手动启动。

## 实际独立核验证据与产物

```json
{
  "status": "HOLD_SCORING_PREPARATION_CACHE_VERIFIED",
  "denominators": {
    "backgrounds": 60,
    "baseline_references": 60,
    "request_references": 7560,
    "all_references": 7620,
    "unique_float_RGB": 7116,
    "reused_prior_outputs": 0,
    "new_model_input_forwards": 7116,
    "new_author_ones_probes": 1,
    "zero_actual": 504,
    "nonzero_actual": 7056,
    "weak_nonzero_empty_M_eff": 2016,
    "undefined": 0
  },
  "complete_sha256": "b496678690ed7394d54b3038f1e43676f5634892afb9c7aa34b25e1285b1a870",
  "boundary": {
    "counters": {
      "model_loads": 0,
      "tensor_loads": 0,
      "author_placeholder_forwards": 0,
      "model_input_forwards": 0
    },
    "events": {
      "source_RGB_reads": [],
      "heatmap_reads": [],
      "blocked": [],
      "NUL_opens": 0,
      "temporary_write_opens": [],
      "system_queries": []
    },
    "GT_original_reads": 0,
    "generation_training_calibration_fitting": 0
  }
}
```

- [独立评分模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_hold_scoring.py)
- [来源及六拟合／状态绑定](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation/sources.json)
- [冻结运行协议](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation/protocol.json)
- [7116唯一输入](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation/unique_inputs.json)、[7620引用](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation/references.json)
- [人工标量／资源／边界](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation/scalar_verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation/complete.json)

实现SHA256：7991c632b2b3349401f2dc68f8e4ab08269e8ad2bc4e21a466c89ea4422b76d9。
准备complete SHA256：b496678690ed7394d54b3038f1e43676f5634892afb9c7aa34b25e1285b1a870。

## 已实现命令

本次已实际执行prepare和另进程verify-preparation。重复核准备只读：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_scoring verify-preparation
```

**以下实际评分入口已实现，但本次未执行；下一步明确授权后才运行：**

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_scoring run
```

只在评分完整结束后另进程全量只读核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_scoring verify
```

run／verify实际评分缓存分支尚未实测；提供命令不等于自动开始或实际兼容已通过。准备仅为下一步完整评分建立固定边界，不声称检测／SEMU／真实屏幕效果。

