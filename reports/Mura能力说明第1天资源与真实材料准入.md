# Mura能力说明第1天：资源、真实材料准入与构造失败定位

日期：2026年11月3日。执行依据：[一周计划](D:/华兴源创/计划/Mura检测器能力说明一周执行计划.md)第1天及[设计补充](D:/华兴源创/计划/Mura检测器能力说明与真实核验设计补充.md)。私人设备、公开资源、个人学习与非商业研究，公司材料隔离。

**第1天的元信息准入及只读失败定位已完成。固定50对真实Mura具备进入后续CPU取得步骤的来源清单条件，但本次没有读图或重新核实GT内容；模型评分仍未准入。已定位到“依赖构造过程的事务外临时写入被新守卫拒绝”，未取得实际触发者的调用栈，不宣称已修复或验证模型兼容。**

## 一、范围与实际产物

本次仅读取设计／计划、封存源码、报告及JSON元信息，对允许的5份模型资源作流式字节hash，对数组和真实源图只作目录／文件stat。没有导入Torch、模型或数组包，没有读取RGB／GT图像字节、NPY／NPZ／概率载荷、解析模型tensor或pickle，没有前向、生成、训练、校准、计算新tau或统计拟合。D_hold未解封。

本次新增独立科学准入缓存[day1_admission_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/day1_admission_v1)，仅5份JSON：

- [sources.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/day1_admission_v1/sources.json)：来源、资源字节身份、冻结核心环境、完整目录元信息与保护清单。
- [real_candidates.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/day1_admission_v1/real_candidates.json)：按000—049固定的全部50路径对、原清单hash、stat及历史角色。
- [guard_diagnosis.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/day1_admission_v1/guard_diagnosis.json)：失败身份、静态守卫差异、已知／未知与一项有界诊断提案。
- [verification.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/day1_admission_v1/verification.json)：本次资源及操作边界，准入仅限元信息、不等于模型可运行。
- [complete.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/day1_admission_v1/complete.json)：上述4文件的封存hash，SHA256为be272e26ff988e192d2d834bdb40b9ed2a9b6b2bd4c1e99e9f114c7730e2da1e。

未创建mura_capability代码包、运行模块或模型入口。新缓存不是新的图像／标签或模型评分结果，不向旧目录追加文件。后续仅完整同身份核验复用，未知产物保留查因。

## 二、冻结数字分支和正常工作点

全量核对7个完成目录的文件名闭合、重定向情况、646份JSON（含complete）的hash。非JSON载荷共15774份，仅核stat和既有清单，不打开内容；D_fit及正常缓存的stat摘要与已封存来源绑定一致。既有独立全数组／概率核验证据从原报告读取，与冻结评分准备中的证明逐字段核对，不能将此写成本次重新验证了数组载荷。

| 完成目录 | 含complete文件数 | 本次载荷处理 |
| --- | ---: | --- |
| preparation_v1 | 9 | JSON身份核对，不重跑抽样或数值实验 |
| fit_inputs_identity_v2 | 14883 | 14874份数组仅stat，复用旧全量独立证据 |
| fit_scoring_preparation_v1 | 7 | JSON、原实现及两模型来源绑定 |
| normal_preparation_v1 | 4 | JSON及冻结来源 |
| normal_inputs_v1 | 305 | 300输入仅stat |
| normal_scoring_v1/PatchCore | 606 | 300热图仅stat，核元信息与原证明 |
| normal_scoring_v1/G1 | 606 | 300热图仅stat，核元信息与原证明 |

D_fit仍为60背景、7560扰动请求和60基线、7620引用／7305唯一float RGB；315零变化请求复用背景，7245非零输入保留。与Q/F内容摘要重合0来自既有冻结元信息，本次没有重新计算工作RGB内容。请求和唯一输入都不等于独立背景或物理面板。

| 模型 | 本轮原tau | Q同池，分母200 | F检查，分母100 | F Wilson双侧95% |
| --- | ---: | --- | --- | --- |
| PatchCore | 1.852951288223265 | FP10／TN190 | FP6／TN94 | 约2.7786%—12.4768% |
| G1 | 0.6587551027536391 | FP10／TN190 | FP7／TN93 | 约3.4319%—13.7495% |

均沿用q0.95／linear／score严格大于tau，未重取分位数、计算实际新阈值或用F调整配置。两模型共享同300正常输入，F共享同100张，不能相加为600或200独立正常图。状态前后一致来自已有保存证据，本次没有加载参数重新测量。

PatchCore完整清单仍为2764ecf4c9fd942fda4db783c491848f49c7b9a67842d91a9ca27df3732a72ec；G1仍为75dd6b85f50bd6ebc98591aa90496e6c74dd61e039905fccfecec299b0c7bcec。原ImageNet V1、FAISS库、pkl元信息及G1两份最终权重共5个资源的当前字节hash均与正常准备协议一致；不加载／执行其中任何内容。

## 三、真实50对的元信息准入

来源是旧AnoLCD display/test/mura/000.png—049.png，与对应ground_truth/mura/000_mask.png—049_mask.png。冻结旧dataset_manifest SHA256为9d0bf06efd17660e2a0ab5e3e3d45337021b7305490f89f1c32512a4f479a270，旧split为feb206cd18eeb9594c64b30afead64577f34473bda57bcaef5a58b3294124ddb；本次均重新核对。

100个源路径均存在，文件大小与旧清单一致，未发现路径重定向。这里核对的是stat，不是当前源文件内容hash；旧清单中的1024×1024、掩码0／255和阳性像素数仍是历史标签审计记录，不能说本次解码核实。

全部50图在旧Tfinal；PatchCore历史评价过该池。旧清单中50个编码hash和50个RGB组ID各不相同，已知跨划分／测试内重复列表没有涉及这50张；这不排除近重复、同面板或256工作图碰撞。新工作摘要、GT缩放存活及有效分母均待后续CPU取得确认。

候选审阅顺序019／024／020／023／021，实际参考019／024／020，与原reference_generation/v1协议一致。候选和参考身份逐例保留，不剔除后把剩余45或47图称为独立验证。G1是否历史对这50图全部评分，当前证据不足，记未知。两模型未用这些真实异常训练的说法不能替代未见验证。

固定候选清单可进入后续明确授权的CPU取得；尚不能写成真实输入已准备、标签当前合法、50独立工件或模型检出结果已取得。

## 四、只读定位：知道什么，仍缺什么

### 4.1 失败现场及停止位置

原[PatchCore失败目录](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scores_v1/PatchCore)有started／sources／references／inflight／failure，没有complete；records和heatmaps均0。inflight是model_construction。失败四键来源绑定正常工作点、自己的原tau和fit评分准备complete，未发现串用其他模型或阈值。

保存计数：model_loads=1为进入构造分支的尝试；tensor_loads=0、作者ones=0、模型输入前向=0、数组打开=0、完成唯一输入=0。错误为程序自身RuntimeError：Write outside this independent transaction；被拒路径为C:/Users/Huang Yuhang/AppData/Local/Temp/7stlrcex。

因此能确认：停止在实际权重加载和输入评分之前，当前没有缺陷响应结果。这不是显存不足、学习率或检测效果问题的证据，也不是本次系统审批拒绝；不能通过删除标记或自动重跑旧score处理。

### 4.2 静态调用路径和守卫差异

可从源码确认的路径为：独立score写构造inflight → [原生build_model](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference_runtime.py:182) → 导入patchcore.patchcore → [作者backbones.py](D:/华兴源创/实验/弱缺陷Demo/vendor/patchcore-inspection/src/patchcore/backbones.py:1)顶层导入timm → timm的模型及可选Hub依赖。该阶段尚未执行源码后面的骨干参数加载。

[新守卫](D:/华兴源创/实验/弱缺陷Demo/mura_response/fit_inference.py:99)同时检查open模式和整数flags中的O_WRONLY／O_RDWR／O_CREAT／O_TRUNC，且写入只准入当前独立事务目录；[旧正常评分守卫](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference.py:75)只检查模式字符串。

对于虚构的mode=None和O_RDWR|O_CREAT|O_EXCL|O_BINARY，旧表达式不识别为写、新表达式识别为写。本次只作标量／源码比较，没有调用os.open、写临时文件或重现模型。这说明旧正常评分曾成功不能反证依赖没有临时写入，也不能据旧blocked列表为空推断其覆盖了全部flags型写入；不修改原报告或把旧结果改成失败。

[本地Python tempfile源码](D:/Python/Lib/tempfile.py:183)显示，首次确定默认临时目录时，会生成8字符名字、尝试os.open创建文件、写入后删除；它优先检查TMPDIR／TEMP／TMP。失败路径形态与这种可写性探测相容，但这仅是候选机制，不证明本次一定调用了gettempdir。

### 4.3 当前不能确认

failure没有保存traceback或写入调用栈。其1261项已导入模型模块列表的末尾含timm.models._pretrained，只能说明已导入状态，不能把列表尾项当实际调用者。当前可选依赖版本也不能倒推为失败时版本，未证明huggingface_hub、Torch或某个具体模块是根因。

本次资源进程的TEMP／TMP位于沙箱Packages临时目录，而失败保存的是用户常规Local/Temp路径；运行环境不同，不能把当前环境复制成原失败环境或把导入缓存状态假定相同。没有新的导入／构造／前向证据，尚不能验证任何临时目录适配能解决问题。

## 五、唯一最小后续建议：独立依赖导入写入栈诊断

先查清实际调用者，再决定最小构造适配。建议另建mura_capability/import_diagnosis.py及import_diagnosis_v1；不改旧fit_inference、作者代码、环境或失败目录。本次不开发、不执行该诊断，也没有对应已实现命令。

未来明确授权后，在独立子进程按原依赖顺序做导入诊断：使用新的审计记录器，在首次事务外写入真正发生之前记录目标、mode／flags、Python调用栈、已导入依赖身份及临时环境，并拒绝该写入。只允许诊断目录保存结果；不直接给整个系统Temp写权限，不预热gettempdir、不重试第二种环境或并行试多种修复。

边界：不调用build_model、不实例化检测器／优化器、加载权重、调用CUDA／前向、读图／GT／数组或旧测试分数；包导入本身是新增权限，不能在本轮只读任务中执行。只捕获调用栈不能证明模型兼容或预测等价。若未复现，或出现未列明副作用，保留并说明环境／导入状态差异，不进入自动构造或完整评分。

只有实际调用者与所需副作用明确后，才审阅“将必要临时活动限制到新事务自有目录”的独立适配。失败现场不续接，数字输入和真实输入的评分均等待新的明确准入，不削弱参数／BN核对或更改tau来绕过障碍。

## 六、后续CPU取得的固定范围

后续第2天的CPU取得可独立于模型评分准备，但仍需当次明确授权：只读固定50 RGB＋对应50 GT，每源在该取得事务中一次；原hash严格核对，RGB／GT分开计数、缓存和入口。RGB按原RGB转换、bilinear256、float32/255；GT原生二值保存并最近邻到256，记录面积和标签消失，不拉伸、膨胀、移位或换样。

旧RGB组ID只作历史角色；新解码RGB与工作RGB各用独立定义摘要。核固定50内部及与正常／数字工作输入的精确重复，保留源引用、冲突和分辨率损失，不把重复当独立样本。原生GT阳性而工作GT消失时，整图异常仍计TP／FN，空间命中NA并另列分母；不擅自删出整图分母。

不打开其他正常／异常图，不用原测试分数决定样本，不动D_hold、S_bg、Q_cal。没有配准干净B，不能把真实GT内外均值差当数字C_pair或直接作真实SEMU预测。首版真实核查限于实际报警、峰位置与GT面积及证据缺口。

## 七、资源、边界与本次执行说明

完成轮快照：可用RAM7.289597GiB，D盘空闲113.038208GiB；RTX5060 Laptop总显存8151MiB／空闲7531MiB，驱动621.01。满足当前CPU2GiB／D40GiB门槛；按元信息也满足未来启动RAM4.5GiB／显存4GiB，但不代表已验证CUDA或长期运行。

解释器Python3.12.7，项目.venv的解释器hash与原协议一致；8项核心包元信息均一致：Torch2.8.0+cu128、torchvision0.23.0+cu128、timm1.0.20、faiss-cpu1.12.0、NumPy1.26.4、SciPy1.16.2、Pillow12.3.0、tqdm4.67.1。仅读安装元信息，不导入这些包。

补充记录当前可选依赖：huggingface-hub2.2.0、safetensors0.8.0、httpx2 2.13.1、hf-xet1.7.0，httpx分发包未发现。原核心协议没有为这些可选依赖提供完整历史版本／源码身份，本次将观察到的相关源码纳入新来源hash；不认定依赖变化为失败根因，不安装、升级或换pin。

完整准入脚本的图像／GT／输入／概率读取、模型／数组包导入、模型前向、tensor加载、生成、参数更新、校准、阈值重算、统计拟合和tempfile探测均0；5资源只读字节hash。此前一次只读核对因假设错误的来源字段名KeyError停止，读实际四键签名后更正核对表达式并重新核验，没有改冻结实现、重试评分或创建数据／模型产物；该首次元信息核对也仅对5资源作字节hash，不能把完成轮计数当整个会话全部文件访问计数。

不建立过程ledger、软件测试、工作树或提交流程；本次仅做科学来源准入和静态诊断，必要JSON是最终证据。第1天完成不等于第2天CPU输入或模型问题已解决；唯一下一步维护在一周计划第六节，所有训练及统计拟合仍由用户手动启动。

## 八、独立进程只读核验

完整准入缓存封存后，另起进程只读核对新5文件及全部来源、695份受保护实现／元信息／资源hash、9个原保护目录的stat签名及100个真实候选源路径stat，退出码0。没有导入模型或数组包、打开源图片／GT／输入／概率、执行tempfile探测或运行模型。实际返回如下；该验证不重新计算分位数或读取数组载荷。

```json
{
  "status": "DAY1_ADMISSION_CACHE_VERIFIED",
  "complete_sha256": "be272e26ff988e192d2d834bdb40b9ed2a9b6b2bd4c1e99e9f114c7730e2da1e",
  "strict_output_files": 5,
  "protected_files_sha256_unchanged": 695,
  "protected_directory_stat_signatures_unchanged": 9,
  "real_candidate_pairs_stat_verified": 50,
  "source_RGB_GT_bytes_read": 0,
  "array_probability_bytes_read": 0,
  "prior_full_payload_proofs_reused_not_rerun": true,
  "actual_dependency_caller_confirmed": false,
  "model_import_or_forward_or_calibration_or_fitting": 0,
  "asset_access": "opaque hash only",
  "asset_hash_file_opens_this_verification": 5,
  "new_code_created": false,
  "scoring_ready": false,
  "CPU_acquisition_is_future_explicit_step": true
}
```

## 九、最终只读审阅与保留项

按executing-plans完成一次新上下文整段审阅，Critical／Important／Minor均无需要修改的事项。审阅仅读报告、协议、源JSON、指定源码和目录元信息，没有重跑完整核验或执行模型。

审阅未重测695文件hash／9目录stat，执行方以本节前独立进程实际记录及交付前核对承担该证据；不将审阅替代验证。真实RGB／GT内容、工作标签存活、近重复／物理独立性及数组／概率当前载荷仍未在本轮读取，继续保留为后续取得或既有独立证明的适用边界。触发临时写入的确切依赖、失败时环境、修复有效性和模型兼容仍未知，不改判为通过。

未来导入诊断尚未实现，不能宣称其副作用隔离已验证；它必须在新的明确范围中实现、静态核准并按实际审计记录判断，条件不满足则停止，不凭本日准入自动导入或构造模型。本报告没有留待偷偷实施的修复或小改动。
