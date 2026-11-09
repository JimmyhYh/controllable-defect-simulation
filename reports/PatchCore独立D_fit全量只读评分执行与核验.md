# PatchCore独立D_fit全量只读评分执行与核验

日期：2026年11月4日。私人设备、公开数据、个人学习与非商业研究，公司材料隔离。

## 唯一当前状态

原唯一一次run已正常退出（终端工具session89781，exit_code=0），7305份唯一输出全部完成并封存；随后在新的独立CPU进程执行已实现verify，正常退出并返回FULL_FIT_COMPLETE_CACHE_VERIFIED。实际7303次新增模型输入评分＋2份最小结果复用，另计1次作者ones构造探测，7620引用／7560请求／60基线覆盖完整，无failure或残留inflight。本次核验新增模型前向0、来源RGB读取0、tau重算0；没有导入Torch或启动任何新的评分，未进行CPU响应统计／拟合，未读取D_hold或真实评价。完成的是冻结原始分数与热图及其完整性核验，还没有数字响应曲线、SEMU适用性结论或真实屏幕效果。

本轮没有启动G1，没有模型训练、建库、合成、重校准、统计拟合或CPU响应统计。没有读D_hold／真实评价材料。原Temp调用者仍未知，旧失败及旧NO_GO条件结论保留。

## 固定范围和复用

| 范围 | 冻结数量／处理 |
| --- | --- |
| D_fit背景 | 60；不是7620个独立背景 |
| 正常基线引用 | 60 |
| 全部扰动请求引用 | 7560，原顺序保留 |
| 总引用 | 7620 |
| 唯一float RGB | 7305，固定unique_inputs原顺序 |
| 最小结果复用 | 2份：fit_b000和fit_b000_r04_circle_bright_a0 |
| 本次评分实际新增模型输入前向 | 7303，全部已完成并核验 |
| 本次评分实际作者ones构造探测 | 1，独立计数；不是新增数据样本 |
| 零实际变化请求 | 315，引用相应背景输出，不额外前向，不当缺陷FN |
| 非零实际变化请求 | 7245 |
| 测量未定义请求 | 0；这是生成元信息，不是模型响应统计 |

最小结果complete固定为51472f150d69d2d206fd26990a56ea95d35d6a9440ce1e7212c0e4bdffd038b6。准备阶段逐项核对其原始分数、两张热图、输入摘要、原模型／BN状态、预处理与tau；两份输入工作hash分别为c8320151fcd29f49b0b4740018274d4a97ed50305df52f84f921e923035ebf6b和e7bdd46541f306c27238dc09d8b6efba9a653afc803b65ad5390d4faf6296c12。

新评分结果只保存这两份的源引用与分数，不再次前向或复制改写旧热图。其余输入逐一读取原float32缓存，核文件hash及规范RGB摘要；不PNG回读、不再次除255、不重生成。所有7620引用保存到新run的references.json，315个零变化请求和其他去重引用不补推理。

## 模型和运行口径

直接复用封存mura_response/inference_runtime.py的build_model与predict_rgb；没有调用旧score入口、安装旧守卫、修改旧模块全局路径或实施C01导入改写。固定原作者显式torchvision WRN50_2、已有ImageNet V1和819×1024的N80 FAISS库；不fit、不改库。

沿原layer2／layer3、1024／1024嵌入、patch3／NN1、ImageNet归一化、eval／FP32／batch1／workers0／threads4、seed42、TF32与benchmark关闭。输入仅RGB，预测函数不读GT、M_geo、残差或tau。评分仍为原最大patch平方L2近邻分数，热图为原作者输出；不能用热图max替代整图分数。

原tau=1.852951288223265保持，q.95／linear／严格大于规则不变。本轮不重新取分位数、不依据刺激响应选配置，也不计算实际报警率、联合命中、响应曲线或拟合。

实际构造后的完整参数／state_dict／BN／training／gradient签名等于最小结果与原正常工作点；7303次新增前向后逐次检查通过，最终签名保持。独立CPU核验验证了保存的签名、全部逐记录状态标记与绑定来源。参数SHA256为6372d614e547b82ad3683dab5f7ab453897c13d765c73871fb26305f1a4e4587，完整state_dict SHA256为1a33eb6df3f1ea53b04d7913bf4039dbc8f13db5f5ed7c9263a39afc7f9c25f9。构造ones探测仍按原行为执行；状态不变指从构造结束开始的评分过程，不声称构造前后BN毫无变化。

## 资源和启动证据

准备时可用RAM8.725845336914062GiB、D盘113.50523376464844GiB；运行事务启动时RAM8.664134979248047GiB、D盘113.49884796142578GiB。显卡NVIDIA GeForce RTX 5060 Laptop GPU，驱动621.01，预查询总显存8151MiB／空闲7899MiB。沿原启动RAM≥4.5GiB／空闲GPU≥4GiB／D盘≥40GiB条件；运行中沿原全量协议RAM≥2GiB，不是沿用最小入口额外的逐输入4.5GiB门槛，未降低原全量协议规格。

本轮模型事务保存PID22036，终端工具session89781。另一次工具执行Get-Process未找到这个PID，但同一session随后继续输出575条完成进度、inflight推进至ordinal590，保存记录持续增加；故该PID查询不能当作本轮评分已经退出的证据。跨工具进程可见性原因仍未知，本次不另开诊断或重启评分。仅一次run已启动；正常依赖初始化、NUL、本地platform查询、正常临时文件与作者FAISS辅助初始化已获授权。继承当前TEMP／TMP，没有改环境、包版本或解释器，没有预热旧失败入口。

评分事务记录elapsed=11434.136152699999秒（约190.57分钟），其中构造21.62609280000015秒；逐新增输入的评分／状态检查／输出IO计时合计11295.57759879994秒。这些计时包括过程中等待，不是纯GPU延迟，也不含末尾来源复核、完整hash封存及独立CPU核验，不能当严格全流程耗时。完成4025份时elapsed=4046.2261614秒，完成4050份时elapsed=8048.2481709秒，两条进度输出间隔4002.0220095秒（约66.70分钟）；其原因仍未确认，不归因于休眠、某个依赖或模型，不为说明它而重跑。最终PyTorch GPU allocated峰值366.80615234375MiB、reserved峰值410MiB，覆盖构造与评分；结束可用RAM7.500038146972656GiB／D盘111.17873764038086GiB，是系统余量采样而非进程内存峰值。7303份新增热图的文件预算为1915372416字节，JSON／目录和事务成本另计。

## 独立实现、缓存与权限

实现：[patchcore_fit_scoring.py](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_fit_scoring.py)。

结果根：[patchcore_fit_scoring_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_scoring_v1)。preparation的sources／protocol／7305输入／7620引用及complete完整；run保存7305份record、7303份新增float32热图、references／started／result／verification及complete。两份复用热图引用原最小目录，未复制改写。run输出清单共14612份文件，加complete共14613份；目录和来源完整性已由独立verify全量检查。

实现SHA256：8382d936fec12ab0e45afcd4245da698c3b66daf14ed109efc9253b0e2e2494d。
准备complete SHA256：ff3ae176f3677d2fa994b66e8f22b035d92a0308480cc2c35f08661c19ba7538。
run complete SHA256：501b547812f6cb941e8f4f8ed7f92046c8451e330d5698302fb77d7578428f7e。

新边界仍拒绝网络、未列定模型资产／数组／原图和输出或正常临时目录以外的写入。原params.pkl只作字节hash，原权重仍weights_only=True。Python审计不是完整原生代码／操作系统沙箱，不能把未记录事件等同全部原生I/O已独立证明隔离。

原实现、vendor、环境、V1／N80库、tau、最小结果及旧失败保持。评分结束后和独立核验中，均再次核对绑定旧82份文件（含已完成最小入口）、最小preparation／run完整结果及原历史目录清单，来源闭合通过；该结论限被绑定的文件与目录，不是扫描所有项目文件后的泛化声明。核验程序只读取模型文件的字节hash，不解析tensor／pickle，不加载模型。

CPU静态／人工标量及实元信息核验已通过；按executing-plans完成一次独立只读代码审阅，Critical／Important／Minor均0，未改已封存源码。未建立过程表、软件测试套件、Git或提交流程。用户禁止这些流程优先于技能默认要求；技术决策记录在本最终产物中。审阅不替代实际全量验收，持续资源／外部干扰与检测效果不由静态审阅证明。

## 已实现入口与停止规则

prepare、verify-preparation、原唯一一次run以及正常退出后独立verify均已实际调用并成功。**无需重新评分，不要删除或修改完整缓存以触发重跑。**

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_fit_scoring run
```

上面是已完成的历史评分命令，展示仅用于复现说明。现在完整同身份缓存存在，run只读核验复用，不再前向；通常直接使用下面的verify即可。-B保持关闭普通字节码缓存写入。

评分已正常退出且run/complete.json齐备，本次已在新独立CPU进程实际执行以下已实现核验入口并通过：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_fit_scoring verify
```

本次verify全量核了cache清单／hash、7305条记录、7620引用、每份热图的shape／dtype／有限性、实际前向和复用分母及保存模型状态，逐项对照原来源／tau。它没有导入Torch、读取来源RGB、新前向、取新tau或计算响应统计；没有向preparation／run追加任何文件。

独立核验实际返回（exit_code=0）：

```json
{
  "status": "FULL_FIT_COMPLETE_CACHE_VERIFIED",
  "complete_sha256": "501b547812f6cb941e8f4f8ed7f92046c8451e330d5698302fb77d7578428f7e",
  "denominators": {
    "all_references": 7620,
    "backgrounds": 60,
    "baseline_references": 60,
    "independent_background_count_not_references": 60,
    "new_author_ones_probes": 1,
    "new_model_input_forwards": 7303,
    "nonzero_actual_change_requests": 7245,
    "request_references": 7560,
    "reused_minimal_outputs": 2,
    "undefined_measurement_requests": 0,
    "unique_float_RGB": 7305,
    "zero_actual_change_requests": 315
  },
  "new_model_input_forwards": 0,
  "new_source_RGB_reads": 0,
  "tau_recomputed": false,
  "CPU_response_statistics_performed": false
}
```

run发现已有完整同身份缓存时仅只读核验。failure、inflight、缺件、未知成员、来源改变、资源不足或不明确前向均保留停止，不自动续推、补推、重试、取成功交集、换样或放宽规格。旧失败入口永不作为本轮恢复命令。

本次评分与独立核验已经完成，没有自动进入G1、D_hold、真实取得或CPU响应统计。下一项按原第5阶段固定规则，另行授权CPU缓存响应后处理，才能回答冻结检测器在各数字刺激下是否报警／峰是否落入固定区域及配对变化；统计模型拟合仍须由用户手动启动。评分完整性通过不代表SEMU代理有效、可预测留出背景或真实屏幕检测能力已证实。

