# PatchCore D_hold60 CPU输入取得与独立核验结果

日期：2026年11月4日。私人设备、公开材料、个人学习与非商业研究，公司材料隔离。

## 实际结论与本次执行范围

固定D_hold60的CPU输入与扰动缓存已完整生成，独立verify-inputs全量只读核验通过。本次任务开始检查时，run目录已经存在complete.json，且不存在failure／inflight；因此按完整同身份缓存复用规则，**没有再调用acquire，没有重复打开／解码原图或render**。本次实际执行verify-preparation和另进程verify-inputs；既有生成事务的数量及耗时来自封存记录，不能写成本次助手重新生成，也不能仅凭这些文件推断原事务由谁启动。

只完成CPU输入，尚未执行D_hold模型评分、响应整理、Brier／log loss、可靠性或bootstrap。没有模型导入／前向、训练、建库、校准、统计拟合或真实评价；没有重新打开旧训练160、Q/F300、D_fit60原图／数组／概率。原源码、preparation、六个拟合／p0／D_fit变换、V1／N80库、tau=1.852951288223265及全部旧失败不改。

## 身份、覆盖和预生成门槛

[原七文件准备](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation)的complete为4056569613a57ffb2308c545cdc45dd3193cecf6eff0eb0e78a0838a8e0c6d8c；本次verify-preparation返回HOLD_INPUT_PREPARATION_CACHE_VERIFIED。冻结D_fit六拟合run complete为983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649，p0=0.4109788359788335；本次只核保存身份，没有重新标准化、计算p0或优化。

既有取得记录与本次缓存复核确认：

- 原序60个D_hold背景，原文件各打开一次、解码一次，来源hash／尺寸／模式记录与冻结元信息一致；原始RGB新摘要与规范256工作摘要分别保存，旧组ID未被强制等同新摘要。
- 全部540个背景×几何条件均可定义；60背景均9/9可定义，部分及全不可定义背景均0。每几何实际背景分母60，满足至少40背景有一个几何可定义的停止门槛。
- 角色内60个工作RGB互异；与旧训练160、Q/F300、D_fit60的520条比较路径工作摘要无碰撞。比较只控制精确工作内容重复，不证明物理面板或采集组独立。
- coverage中formal_render_calls_before_gate=0，source_audit中的生成前边界为60次打开／60次解码／0次render；所有540条件和碰撞核对在正式生成前完成。

原文件hash核验发生在既有取得时。本次不重新读原图，不能冒称重新证明其当前字节；本次核的是封存输入、来源记录和测量链。

## 全部实际分母及保留状态

固定9几何×亮暗×7幅度×60背景，共7560请求，全部保留且可定义，未定义0；另60背景基线，总7620引用。

- 实际非零变化7056请求。
- 零实际变化504请求，均来自亮方向，引用自身背景，不称有效缺陷或漏检。
- 非零但D>float32(1/255)诊断掩码为空的2016请求，仍保留。其中幅度0.001／0.002各1008，不能因M_eff空而删除。
- 发生任意通道裁切的577请求；亮方向504、暗方向73。
- 内容去重后7116份唯一float RGB，包括60基线＋7056非零刺激；去重没有缩减7560名义请求或7620引用分母。

每个幅度均1080请求，非零1008／零变化72；主网格七幅度及亮暗／全部九几何的详细分母原样见manifest.summary。亮方向3780请求中非零3276／零504／弱936，暗方向3780中非零3780／零0／弱1080。每几何840请求、非零784／零56／弱224；所有几何均完整保留，没有按结果选择子组或补背景。

这些是数字输入状态，不是检测TP／FN／POD；还没有D_hold模型响应，不能据此判断SEMU的预测价值、缺陷逼真度或屏幕效果。独立背景分母是60，不是7116或7620；物理采集分组未知。

## 实际全量只读核验

本次在独立CPU进程执行已实现verify-inputs，exit_code=0，返回HOLD_INPUTS_COMPLETE_CACHE_VERIFIED。核了完整成员及hash、来源／实现／原tau身份、7116份规范float RGB、9份M_geo／alpha、7560份残差与诊断M_eff、全部请求原序、7620内容去重引用、540项Lb与碰撞覆盖、ROI预裁切代数及裁切、实际测量与未裁切能量比例；完整语义与保存verification一致。

原float32 RGB保持，不PNG回读、不量化或再次除255。实际未裁切能量比例最大相对误差5.489602433561558e-05，未超过冻结5e-4容差。零变化／弱非零均保留，不用M_eff筛选。

以下为实际CLI返回的总体分母与边界节选；按幅度／几何／亮暗全部分项已在同一返回中核对，与封存manifest一致：

```json
{
  "status": "HOLD_INPUTS_COMPLETE_CACHE_VERIFIED",
  "complete_sha256": "5cbc1351afe334cefd23486aa35efd2143a9e38def506f385876c3e48d31e355",
  "summary": {
    "backgrounds": 60,
    "clipped": 577,
    "defined": 7560,
    "geometry_arrays_verified": 9,
    "no_original_redecode_or_render_during_verification": true,
    "nonzero_actual": 7056,
    "references": 7620,
    "request_residuals_verified": 7560,
    "requests": 7560,
    "unclipped_energy_max_relative_error": 0.00005489602433561558,
    "undefined": 0,
    "unique_RGB": 7116,
    "weak_nonzero_empty_M_eff": 2016,
    "zero_actual": 504
  },
  "boundary": {
    "counters": {
      "original_file_opens": 0,
      "original_decodes": 0,
      "formal_render_calls": 0,
      "saved_array_reads": 37050,
      "model_forwards": 0,
      "statistical_fits": 0
    },
    "blocked": [],
    "forbidden_modules_loaded": [],
    "old_training_Q_F_D_fit_original_reads": 0,
    "GT_probability_reads": 0,
    "calibrations": 0,
    "detector_parameter_updates": 0
  }
}
```

37050是本次保存数组读取事件数，包含同一背景／几何／请求在不同核对步骤的重复读取，不是图像分母、前向数或重新生成次数。当前新增原图打开／解码／render均0；既有生成事务各为60／60／7560。没有把旧生成计数当成本次新增。

## 资源、时间与保留限制

既有生成事务记录elapsed_seconds=388.5782981000011，约6.48分钟，覆盖其started之后的取得、生成、CPU内部语义检查与最终来源检查，排除入口前置来源准备核验及最后封存。本次独立verify-inputs另行执行，没有精确单独耗时记录，不把6.48分钟当成本次核验时间或模型推理时间。

既有事务121次系统余量采样：开始RAM6.462882995605469GiB／D111.61154556274414GiB，最小采样可用RAM5.782039642333984GiB／D106.42959594726562GiB，满足CPU2GiB／磁盘40GiB门槛；这些不是进程峰值或实时全设备占用。封存run共14692文件、5625582274字节，约5.2392GiB。

本次只读核验后的另进程元信息检查：RAM6.3271331787109375GiB、D106.31884384155273GiB；7693项绑定来源hash无变化，源码及preparation／run complete身份一致，无failure／inflight。该检查未读任何数组或原图、未解析模型参数（权重仅字节hash），不重复全量数组核验。

沿用准备时审阅的两项非阻断限制：资源没有进程峰值；若未来守卫触发，异常路径可能仅留下部分目录而无结构化failure.json。本次没有触发守卫、部分事务或该失败路径，不修改冻结实现去补这些诊断字段。遇到未来未知状态仍保留查因，不自动重跑。

本次未开发或修改源码，没有新增过程表、软件测试或提交流程。执行边界判定是复用已有完整同身份输入，而非重复acquire；代价是本次不重新检验原图现时字节，但避免违反一次取得／生成约定。全部训练及统计拟合继续由用户手动启动。

## 封存产物与已实现运行说明

- [独立模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_hold_inputs.py)
- [生成前覆盖与碰撞](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/coverage.json)
- [60背景来源审计](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/source_audit.json)
- [全部7560请求及实际测量](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/manifest.json)
- [7116唯一输入及7620引用](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/unique_inputs.json)
- [既有取得／生成边界及资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/verification.json)
- [完整封存清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run/complete.json)

run complete SHA256：5cbc1351afe334cefd23486aa35efd2143a9e38def506f385876c3e48d31e355。
源码SHA256：81e191feda4268cab439c9ba86df7ea40c0afdec9e4cea923fa8de2667e35952，未修改。

本次实际执行且通过的两个只读入口分别在新CPU进程中运行：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_inputs verify-preparation
```

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_inputs verify-inputs
```

完整缓存只读复用，未知／部分／failure保留停止，不重生成、补数、改seed／位置／幅度或删除标记。下一项仅准备独立D_hold评分模块／协议：先按元信息核原正常／D_fit评分是否可复用及新增前向预算，不自动评分；没有尚未实现的评分命令。G1刺激／拟合与真实评价仍未执行，SEMU背景留出预测评价尚未开始。

