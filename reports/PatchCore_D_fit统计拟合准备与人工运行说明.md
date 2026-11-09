# PatchCore D_fit统计拟合准备、人工运行与核验

日期：2026年11月4日。私人设备、公开数据、个人学习与非商业研究，公司材料隔离。

## 唯一当前状态

用户已手动完成四个主候选、两个SEMU-only尺度敏感性拟合及共同p0。run完整封存，无failure或残留inflight；随后助手在新CPU进程实际执行已实现verify-fit，返回STATISTICAL_FIT_COMPLETE_CACHE_VERIFIED。六个预定优化均成功，系数、D_fit变换、p0及拟合池预测已冻结。

用户运行中实际优化器调用6次；本次助手核验新增优化器调用0，只按原D_fit缓存重算统计量并复核已保存系数的目标、梯度和预测，没有重新优化或修改结果。没有调用检测器、读取RGB／GT／热图、生成、校准或读取D_hold／真实材料。不能据拟合成功或当前训练池目标值判断SEMU有预测用途；背景留出评价尚未开始。

依据：[原研究设计第八节](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[完整CPU响应及独立复算](D:/华兴源创/报告/PatchCore独立D_fit缓存响应后处理结果.md)。旧设计、数据角色、原tau=1.852951288223265、输入、评分和响应缓存均保持。

## 实际数据、共同权重与固定表示

实际使用全部60个D_fit背景、每背景126个共同可定义请求，共7560行；315零实际变化保留。目标固定为H_joint，3107个1／4453个0。H_alarm仍仅作描述，未改成另一个拟合目标。请求原序、共同可定义域、全部三k特征和背景权重经过实际读取与核验，没有丢弱档、零注入或换背景救结果。

每背景总权重1/60，每行权重1/(60×126)，不作类别重加权；变体不是独立物理面板。共同常量参照已冻结为**p0=0.4109788359788335**，来自同一D_fit目标和背景权重。它是当前数字刺激预算下H_joint的常量预测参照，不是真实Mura发生率或人眼可见概率。

| 主表示／敏感性 | 原始特征，不含截距 | 系数数（含截距） |
| --- | --- | ---: |
| C-only | u=log1p(C_pair_pct) | 2 |
| C+area | u、v=log(A_geo)、u×v | 4 |
| SEMU-only，k=0.4 | log1p(SEMU_proxy[0.4]) | 2 |
| C+area+context | u、v、u×v、log(Lb)、背景std(Y)/Lb、亮／水平椭圆／斜椭圆指示 | 9 |
| SEMU-only，k=0.2 | log1p(SEMU_proxy[0.2]) | 2 |
| SEMU-only，k=0.8 | log1p(SEMU_proxy[0.8]) | 2 |

两个额外拟合仅为含k的SEMU-only敏感性；其余表示不重复拟合三次，p0共同使用。全部六项均保留，未选最佳k、模型、重复或中间系数。

交互项先形成原始u×v，再按D_fit背景权重估计连续列总体矩（ddof=0）。常量连续列固定0，不加epsilon；指示列保持0/1（亮=1／暗=0，水平和斜椭圆各一个指示，圆为参照）。这些约定在运行前已冻结，未来背景留出必须使用已保存变换；不会在D_hold重新估计均值／尺度。

## 六个实际拟合状态

原固定目标：背景等权二项负对数似然＋0.01/2×非截距系数平方和；截距不惩罚。SciPy L-BFGS-B、解析梯度、全0初值、maxiter2000、gtol1e-8、ftol1e-12；版本原默认maxcor10、maxls20、maxfun15000亦已固定，没有调参或随机重启。

| 候选 | 迭代 | 保存的惩罚目标 | 保存梯度无穷范数 | SciPy终止依据 | 状态 |
| --- | ---: | ---: | ---: | --- | --- |
| C-only | 9 | 0.315106761025 | 3.418533e-9 | gtol条件 | 成功 |
| C+area | 13 | 0.281453626838 | 1.168730e-9 | gtol条件 | 成功 |
| SEMU-only_k0.4 | 9 | 0.305948119935 | 6.040686e-9 | gtol条件 | 成功 |
| C+area+context | 19 | 0.212297225496 | 3.490590e-8 | ftol条件 | 成功 |
| SEMU-only_k0.2 | 9 | 0.305999327087 | 4.233106e-9 | gtol条件 | 成功 |
| SEMU-only_k0.8 | 9 | 0.306949171088 | 5.648724e-9 | gtol条件 | 成功 |

C+area+context梯度无穷范数约3.49×10⁻⁸，高于gtol=1e-8，但按预定ftol相对目标下降条件正常终止；其success=true、status=0保留。原协议不要求两个终止条件同时达到，没有为该候选放宽参数、补拟合或更改成功规则。

这些是D_fit拟合诊断，惩罚目标不是留出Brier／log loss，不能据0.2123最小认定含背景模型能泛化，也不能由SEMU拟合目标高低认定其预测优劣。六个系数向量、全部标准化均值／尺度／指示约定、优化状态及拟合池预测均原样保存；标准化系数不是物理阈值。

## 实际资源与核验范围

用户手动事务记录运行2.403923900001246秒（约2.40秒），包括实际数据读取、六个拟合及结果处理，排除入口前置来源检查、最后封存和之后独立核验。开始RAM6.5147247314453125GiB、D盘111.61935043334961GiB，结束RAM6.533927917480469GiB、D盘111.61724090576172GiB，满足原CPU RAM≥2GiB／D盘≥40GiB。这些是系统余量采样，没有完整进程峰值或独立核验单独耗时证据。

使用已有Python3.12.7／NumPy1.26.4／SciPy1.16.2，未安装或升级。实际SciPy拟合路径此次已运行成功，但不据单次成功保证其他输入或设备兼容。模型包未加载、检测器前向／校准／渲染／D_hold读取均0；没有检测器训练或参数更新。

独立verify-fit核了完整cache hash／成员、原来源和协议、共同请求与权重、p0、六个D_fit变换、全部系数所对应目标／梯度及所有7560行的拟合概率，新增优化器调用0。该验证会重算已完成D_fit统计量作一致性复核，没有把它作为新拟合或改写冻结结果。

```json
{
  "status": "STATISTICAL_FIT_COMPLETE_CACHE_VERIFIED",
  "complete_sha256": "983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649",
  "new_optimizer_calls": 0,
  "actual_statistics_recomputed_for_verification": true,
  "D_hold_or_real_reads": 0,
  "detector_forwards": 0
}
```

保留的非阻断审阅限制：verify-fit没有逐项独立重算coefficient_order、gradient_inf_norm、设计矩阵rank／condition等诊断字段；它们受完整文件hash覆盖。上表梯度范数是保存诊断，核心梯度向量已复算核对。故本报告称“完整缓存身份及核心数值复核”，不称所有诊断均独立重算。

## 准备历史与实际产物

此前prepare与独立verify-preparation均通过，真实请求解析、真实标准化、真实p0和优化调用均0；只在人工数组上核对背景等权、常量列、指示、截距与解析梯度，最大有限差分误差约1.04×10⁻¹⁰。那是准备阶段事实；之后用户才手动产生本次真实统计量与系数，不能继续用旧准备状态描述当前进度。

当前源码和准备协议未修改。一次准备阶段代码审阅Critical0／Important0／Minor1，Minor核验范围限制如上；本次没有新增源码、软件测试套件、过程表、Git或提交流程，没有为核验再次优化。

- [独立实现](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_fit_analysis.py)
- [准备协议与来源](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/preparation)
- [六模型系数、变换及状态](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run/models.json)
- [共同p0](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run/constant_reference.json)
- [实际行与权重绑定](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run/data_binding.json)
- [拟合池预测](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run/fitted_predictions.json)，仅用于拟合一致性核对，不是验证成绩。
- [实际运行边界与资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run/verification.json)
- [完整封存清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run/complete.json)

实现SHA256：55d9788d105c31a3e5cd1b82227c71e8c80493ca56919d8269733d5d984dcd1c。
preparation complete SHA256：00a2997392b9c995781b5c0160596644d67f975b71281644cf7abb0ba92b02c2。
run complete SHA256：983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649。
依赖响应complete SHA256：69f7c691b241a4175f4a8c1f2442fddc1ea6de978cc44572ee278f73d652adae。

## 已实现入口与当前下一步

用户已执行的历史人工命令：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_fit_analysis fit --allow-statistical-fit
```

**目前不需要重拟合。** 本次助手实际执行且通过的只读核验入口为：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_fit_analysis verify-fit
```

fit遇完整同身份结果只读核验，不重新优化；unknown／partial／failure／inflight保持停止，不自动续拟合、删标记或选最好结果。全部后续统计拟合仍由用户手动启动。

现在仅PatchCore D_fit拟合链已冻结并核验，G1刺激与拟合尚未执行，D_hold和真实材料也未取得。下一项可按原第6阶段准备独立D_hold60 CPU输入协议／入口，先绑定本次冻结模型、变换和p0，再预定义工作hash碰撞、全部9几何Lb覆盖与40背景停止门槛；当前不执行该准备，不读取留出图像或自动生成／评分。

只有后续在固定背景留出上完整比较六拟合、p0及预定指标，才能判断SEMU代理是否有预测用途。当前不声称总体显著、检测提升、真实Mura迁移或屏幕收益。

