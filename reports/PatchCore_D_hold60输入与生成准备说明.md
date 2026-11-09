# PatchCore D_hold60输入取得与生成准备说明

日期：2026年11月4日。私人设备、公开数据、个人学习与非商业研究，公司材料隔离。

## 当前结论

独立模块已实现，prepare与另一个CPU进程的verify-preparation均通过；七份准备文件已封存。当前仅完成来源、固定字典、CPU人工数组与资源核验，**没有打开D_hold原图或保存数组、实际解码、生成正式扰动、模型前向、校准或统计拟合**。独立输出根下只有preparation，没有run。

本次没有增加软件测试套件、过程表、Git／worktree或提交流程。遵从用户明确范围，以scientific人工数组核验代替技能默认的测试／过程文件要求；没有自动推进取得或生成。统计拟合及全部训练继续由用户手动启动。

## 已核对的冻结前提

- [六个D_fit拟合、p0及独立核验证据](D:/华兴源创/报告/PatchCore_D_fit统计拟合准备与人工运行说明.md)：run complete为983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649，四主候选＋k=0.2／0.8敏感性均成功；p0=0.4109788359788335。系数与D_fit变换逐文件绑定，仅复用保存结果，没有重算标准化、p0或优化。
- [D_fit完整评分](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_scoring_v1/run/complete.json)、[完整CPU响应](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_response_v1/run/complete.json)、[原正常工作点](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/q_result.json)及V1／N80库身份保持，tau=1.852951288223265不变。相关模型材料仅字节hash，不解析tensor／pickle或加载模型。
- 原preparation_v1、fit_inputs_identity_v2与normal_inputs_v1的完整清单、来源JSON及既有核验记录均已绑定。旧数组仅核目录／清单与已有摘要元信息，未打开其NPY／NPZ载荷。当前来源映射共7693项；这不是重新核验了全部旧数组载荷。
- 固定D_hold60原序、9几何、亮→暗、0.001／0.002／0.004／0.008／0.016／0.032／0.064幅度原序，总7560请求；人工重建字典逐项等于原封存字典。各基线加请求为7620引用，实际唯一输入、零变化与可定义数量均须未来取得后确定，不能照搬D_fit数量。

G1刺激／拟合仍未执行。本准备只服务PatchCore主链；没有将G1改为通过或静默补齐。D_hold是本轮背景留出数字验证，不是独立真实Mura或新产品／批次验证。

## 独立实现及输入口径

[patchcore_hold_inputs.py](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_hold_inputs.py)另立入口，不修改原resources／inputs／generation_v2、评分、拟合或作者源码。CPU父进程只消费封存JSON与mura_response.inputs纯数组接口，不导入旧带守卫的资源／生成／模型模块。

保持原始对比度，PIL转RGB、双线性256，再float32除以float32(255)。保存输入是HWC／float32／C连续RGB；不PNG回读、量化或再次除255。原文件hash严格等于封存来源；旧RGB组ID只用于角色／排除，不要求等于新摘要。原始RGB摘要定义为1024×1024×3／uint8／C字节SHA256（无header），工作摘要用原inputs.array_identity的dtype／shape／C header＋float32字节定义。

生成沿原I=clip(B+float32(sign×amplitude)×alpha,0,1)，固定M_geo／alpha不随幅度改变。保存实际D=max(abs(I−B),RGB)及诊断M_eff=D>float32(1/255)，M_eff不参与准入或删弱档。实际DeltaY／C／SEMU、裁切、零变化与未定义测量均保留。

## 未来实际取得时的顺序与停止规则

1. 先核本准备完整身份、六模型／p0／变换、原tau和CPU资源。新run不存在才可取得；已有完整同身份缓存只读核验。未知／部分／failure目录保留停止，不续跑、删除标记或自动重生成。
2. 只对固定D_hold60，各原文件打开一次、解码一次，严格核文件hash／尺寸／模式，保存新原始／工作摘要和全部540项背景×几何Lb。原训练160、Q/F300和D_fit60仅消费来源审计中的工作摘要，不重新打开这些原图或数组。
3. 全60处理后比较角色内工作重复，并逐项对照520条旧背景路径的工作摘要；覆盖同时报告9/9、部分、全不可定义及逐几何分母。任一工作碰撞，或至少一个几何可定义的背景不足40，均在正式render前保存现状并停止。40是预算停止门槛，不是功效保证。
4. 条件全部满足才按7560固定请求各render一次；没有重试、补背景、换seed、位置、幅度或删零／弱／未定义请求。保存float输入、9份几何与7560份残差及完整manifest、唯一内容去重引用。
5. CPU核全部ROI预裁切代数、实际残差与掩码、未裁切能量比例、7620引用闭合和目录白名单；完整后封存。另进程verify-inputs全量hash／数组／测量复核，只读缓存，不重新解码原图或render。实际原图hash来自取得时记录，缓存核验不冒称重新证明原图现时字节。

保存基线和刺激的内容去重只节省缓存，不减少名义请求分母；60背景及其变体不能当7620独立样本。没有准备或执行D_hold检测器评分、统计预测比较或bootstrap；这些仍需后续独立授权与实现。

## 本次实际CPU证据与资源

人工核验覆盖6个人工背景×9几何×两亮暗×7幅度，共756个人工请求：保留225弱非零／空M_eff、126零变化及126未定义，均为人工数组计数，**不是D_hold实际结果**。486个未裁切能量配对最大相对误差5.489602433561558e-05，小于原5e-4容差；SEMU标量公式最大差0，坐标往返差0。39背景停止、40通过、角色内碰撞和跨旧池碰撞停止的人工判定均通过。

prepare实测可用RAM6.364128112792969GiB、D盘111.6171989440918GiB，满足CPU至少2GiB／D盘至少40GiB。它们是系统余量采样，不是进程峰值。实际60图取得／7560生成耗时和运行峰值尚未实测；不以人工核验或D_fit历史速度保证。输入最大7620份未去重float RGB约5.58GiB，残差／掩码未压缩量级另约2.31GiB，另有元信息及事务余量；实际去重数未知，继续保留40GiB门槛。

独立verify-preparation实际返回：

```json
{
  "status": "HOLD_INPUT_PREPARATION_CACHE_VERIFIED",
  "complete_sha256": "4056569613a57ffb2308c545cdc45dd3193cecf6eff0eb0e78a0838a8e0c6d8c",
  "boundary": {
    "counters": {
      "original_file_opens": 0,
      "original_decodes": 0,
      "formal_render_calls": 0,
      "saved_array_reads": 0,
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

一次新上下文只读审阅：Critical0／Important0／Minor2。当前准备范围可交付，保留两项非阻断限制：

- 若未来触发守卫，异常处理中的严格boundary可能再次抛出，导致没有结构化failure.json；已有部分目录仍保留，complete缺失仍拒绝复用，没有自动重试。遇此情形保留终端堆栈和目录查因，不将其误判成未运行或允许再次生成。
- 资源只有系统余量采样，没有进程峰值记录，后续报告必须据实说明，不能把最小可用RAM当进程峰值。

对审阅未执行项的判定：实际取得、解码、生成数值、完整输入缓存运行行为和资源峰值均超出本次授权，因此继续未知；代价是准备／人工数值通过不能保证实际数据分支一定成功，下一次实际执行须按停止规则处理。此次没有通过改冻结配置或补样消除未知。

## 实际产物与已实现命令

- [独立模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_hold_inputs.py)
- [协议](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/protocol.json)、[来源／冻结拟合与旧工作摘要](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/sources.json)
- [固定60背景](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/backgrounds.json)、[7560请求](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/requests.json)、[9几何](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/geometry.json)
- [人工数组／资源与边界](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/scalar_verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation/complete.json)

模块SHA256：81e191feda4268cab439c9ba86df7ea40c0afdec9e4cea923fa8de2667e35952。
准备complete SHA256：4056569613a57ffb2308c545cdc45dd3193cecf6eff0eb0e78a0838a8e0c6d8c。

本次已执行prepare与另进程verify-preparation。现在重复核准备可用：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_inputs verify-preparation
```

**以下实际取得／生成命令已实现，但本次未执行；须下一步明确授权后再运行：**

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_inputs acquire
```

仅在其完整结束后，另起CPU进程全量只读核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_hold_inputs verify-inputs
```

acquire与verify-inputs没有在真实留出材料上实测，不能将“已实现”写成“实际运行兼容已通过”。本次没有模型前向／训练／拟合，SEMU留出预测价值仍未得到评价。

