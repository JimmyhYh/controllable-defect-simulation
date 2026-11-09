# Mura检测器响应与SEMU代理刻度执行计划

日期：2026年10月28日。执行方式：writing-plans形成独立计划，随后按executing-plans逐阶段执行。用户已选择先完成数字域刻度验证，不加入本轮真实Mura迁移。

**唯一当前摘要（2026年11月4日，PatchCore D_hold60独立只读评分准备已完成／另进程核验通过）：新增patchcore_hold_scoring，六文件preparation封存，complete=b496678690ed7394d54b3038f1e43676f5634892afb9c7aa34b25e1285b1a870；绑定完整D_hold输入、独立证据、六拟合／p0／变换、原V1／N80／参数与BN／预处理／评分／tau。固定7620引用／7560请求／60基线对应7116唯一float RGB，与旧正常300及D_fit7305内容hash重合0，预定旧输出复用0、新增输入前向7116、作者ones探测1另计。7716项来源及人工标量核验通过，数组／热图／概率／原图读取、模型导入／加载／前向、生成／训练／校准／拟合均0；当前仅preparation，没有run。实际GPU评分／完整评分缓存分支未实测，SEMU留出预测结论尚无；下一项仅显式评分与只读核验，不自动响应统计。全部冻结成果不改，训练及统计拟合仍由用户手动启动。**

当前适用规则：本页下方关于NUL／必要临时文件／系统查询／辅助初始化的禁止和NO_GO，均是旧授权条件下的历史记录；今后独立运行以[AGENTS.md运行边界变更](D:/华兴源创/AGENTS.md)为准。该变更不重置旧诊断单次预算、不允许重跑旧失败入口或改写旧缓存，也不自动扩大图像读取范围。解除的是项目脚本额外拦截，不能越过设备或执行环境的实际权限。

本文件下方保留原数字分支的设计与阶段记录；本周完整交付要求以新一周计划为准。历史“仅数字域”“不加入真实验证”等是原分支边界，不能当作用户最新交付要求；新增真实核查另立协议，不向封存目录追加产物。

## 一、目标、依据与固定要求

目标：完成《受控数字Mura扰动下的检测器响应表征与SEMU代理刻度适用性研究》的数字域最小研究，检验SEMU代理相对对比度、面积及预定背景／形态信息，能否预测背景留出上的冻结检测器响应。

权威设计：[研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)，用户已于2026年10月28日明确确认；其中数据、公式、候选、预算及停止条件是后续逐项执行依据。

私人设备、公开资源、个人学习与非商业研究，公司材料隔离。历史报告与源码不覆盖；旧tau不改。本轮不安排检测器训练、建库、下载或安装。GPU评分入口逐阶段明确，未来只有已实现并核验的命令才提供。不建立勾选追踪、过程表、软件测试套件或提交流程；必要协议、来源身份、输入、概率与统计输出属于最终科研材料。

本计划采用已确认题目、数字域范围与具体设计；批准参数不等于实验已运行。现有RGB重复排除不等于物理批次独立；历史已见的数据集不包装成全新独立验证。

## 二、总实验预算与文件安排

已固定新角色：Q200正常校准、F100正常误报检查、D_fit60统计拟合背景、D_hold60统计留出背景，共420个互斥RGB组；PCG64种子2026102801。元信息抽样与来源身份已封存，原图字节hash／解码及工作分辨率重复仍留待实际读入时核对。

每刺激背景9几何×2亮暗×7幅度＝126请求；拟合与留出各7560，合计15120请求上限。另420正常背景。两模型合计31080次输入引用是未去重预算，不是实际前向数；完全相同float图像复用同一份模型缓存，并保留所有请求引用。

主模型为既有PatchCore，补充为G1最终10000步。两模型各用新Q200的q0.95／linear／严格大于tau；F100检查误报，不回调tau。主响应是整图报警且热图首个峰在固定M_geo内的“空间联合报警”，不是完整分割准确率。主预测评价用按背景平均Brier，四种预定统计表示及D_fit冻结的常量参照全部报告；H按背景2000次bootstrap，不把刺激数当独立样本。CPU统计模型拟合由用户手动启动，助手只准备入口并核验。

独立代码根：`D:/华兴源创/实验/弱缺陷Demo/mura_response`。独立结果根：`D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1`。封存准备／generation_v2及旧失败保持不变；D_fit数字输入、normal_inputs、两模型正常工作点及新fit_scoring_preparation_v1均已核验。D_fit／hold实际评分与统计拟合尚未运行；独立fit_inference复用原生函数但不回改封存正常工作点双模块。

| 模块 | 职责 | 消费／输出 |
| --- | --- | --- |
| resources.py | CPU来源、资源、环境身份与显式CLI | 旧元信息→preparation_v1协议与来源 |
| inputs.py | 封存的纯数组几何、测量渲染及元信息工具 | preparation_v1的固定公式与字典；不增加读图入口 |
| generation.py（已实现，首次取得失败保留） | 独立D_fit CPU图像IO／生成与碰撞／覆盖核验 | 引用准备身份及inputs纯数组接口→fit_inputs_v1失败现场；无完整缓存，未实现hold入口 |
| generation_v2.py（已实现并完成A—D） | 分离历史组ID与新摘要，独立重新取得及完整输入核验 | 引用原固定角色／请求及失败历史→fit_inputs_identity_v2完整封存；不续接旧失败事务 |
| inference.py（正常工作点静态准备已完成） | CPU元信息／协议、进程守卫与显式CLI | 固定来源与两模型→normal_preparation_v1；不在准备入口读图或加载模型 |
| inference_runtime.py（CPU及两模型正常工作点已核验） | 共享正常输入CPU取得、单模型Q→F评分及缓存核验 | normal_inputs_v1及normal_scoring_v1/PatchCore、G1分别完整封存；不修改双模块扩展D_fit／hold入口 |
| fit_inference.py（已实现，CPU准备独立核验；尚未实际评分） | D_fit独立元信息／协议、显式单模型评分和缓存核验 | fit_scoring_preparation_v1七文件；7620引用／7305唯一float RGB／QF重合0；fit_scores_v1/PatchCore现有构造失败、无评分结果，CPU响应及统计拟合另行实施 |
| analysis.py | CPU统计拟合、冻结和留出评价 | 完整评分／测量→fit_analysis_v1、hold_analysis_v1 |
| figures.py | Python图件 | 冻结分析／固定样例→figures_v1 |

resources／inputs与generation_v2已实现并封存，旧失败入口不续接。正常CPU输入及PatchCore、G1正常工作点均已完成；新fit_inference的CPU准备和独立核验通过，单模型score／verify-scores入口已实现；PatchCore score已有构造失败，未取得评分，旧失败入口不重试。hold评分、analysis／figures仍未实现，不提供其命令。

## 三、执行权限、复用与验收

1. 先核身份，再做对应阶段；不调用旧整体入口冒充恢复，不改旧模块全局路径。
2. CPU准备与GPU运行分开。显式元信息／资源核验不得暗中导入Torch／模型、读图或前向。
3. 新图像实际读入时先核来源hash；Q、F、D_fit、D_hold各有独立用途，H在统计拟合及分析口径冻结后才解封。
   每个背景还核规范256 float RGB hash，角色内／跨角色或与旧训练背景精确工作内容碰撞则停止，不仅复用缓存后仍称独立。碰撞不删样、不补样；旧N／N_gen仅允许CPU来源与规范工作hash核对，不作为新增候选。
4. 全部请求都记录，D=0与SEMU未定义分开标注；不以M_eff>1/255、检测结果或外观好坏筛选／替换。
5. 完整同身份缓存只读复用；部分／未知运行保留停止，按systematic-debugging查因，不自动补推理或重生成。
6. 后续“执行第几阶段”仅完成该阶段声明范围，不自动推进新GPU评分或改协议。若用户明确授权连续执行，仍受设计准入、资源与未知事务停止规则约束。

实现身份只绑定本阶段`__init__.py/resources.py/inputs.py`三份源码，后续新增模块各自另绑，不让正常增加模块破坏准备缓存。resources的导入／载荷守卫永久作用于其进程；后续读图／评分进程不得直接导入resources，须在隔离元信息进程执行其verify入口，再消费封存JSON和inputs纯数组接口。

科研验收为协议与实际证据一致、覆盖和失败完整、留出没有参与选择，不要求SEMU必须胜出。本轮不输出标准a90/95、不声称物理人眼标定或真实屏幕收益。

## 四、按顺序执行的阶段

### 第1阶段：只读资源与设计准入〔已完成：供设计审阅，不是新实验通过〕

1. 阅读收束说明、文献评估、旧核查修订与元信息；只读追溯旧render、SEMU、评分与模型来源。
2. 按现有manifest的RGB组排除旧训练／校准／评价角色和参考审阅范围，计算候选数量；仅核代表路径存在，不抽样、不解码。
3. hash核对PatchCore库／通用骨干和G1最终文件；权重只作字节hash，不解析tensor或执行pickle。
4. 查询环境包元信息、RAM／磁盘与GPU资源，不导入模型包。
5. 完成独立研究设计和本执行计划，记录必须整体确认的参数与证据边界。

实际结果：[第1阶段资源与设计准入](D:/华兴源创/报告/Mura响应研究第1阶段资源与设计准入.md)。第2阶段必须基于审阅确认后的设计执行；本阶段资源可用不等于测量／模型兼容已验证。

### 第2阶段：冻结元信息与CPU测量实现〔已完成：元信息／人工数组核验及独立缓存核验〕

1. 核第1阶段来源hash和当前资源；另建resources.py／inputs.py及显式CLI，不修改旧源码或环境。
2. 按已确认排除范围与seed固定Q200／F100／D_fit60／D_hold60，只消费元信息，不读取任何RGB／GT／概率。保存完整角色与RGB／文件来源身份；不足420停止。
3. 冻结9几何、7幅度、亮暗、固定中心、M_geo、alpha与配对局部对比度／SEMU代理公式，写清k的工作坐标约定及Lb未定义规则。
4. 只用人工构造CPU数组核对零输入、均匀／梯度背景、弱于1/255、完全饱和、区域外零变化、未裁切配对能量及工作／原图坐标标量；不读真实图、不生成正式输入、不前向。
5. 保存preparation_v1的protocol／roles／requests规则／sources／verification与complete。冻结目录不再追加，后续模块另绑实现身份。

通过条件：抽样身份闭合，四角色互斥，所有拟定公式在CPU标量／人工数组下定义明确，实际请求计数与预算一致，旧材料未改。预处理／单位或数值条件不能落实时保存并停，不先推理再修定义。

实际产物：[第2阶段CPU准备结果](D:/华兴源创/报告/Mura响应研究第2阶段CPU准备结果.md)、[协议](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/protocol.json)、[固定角色](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/roles.json)、[数值核验](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/numerical_verification.json)、[完成清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/complete.json)。独立verify输出PREPARATION_CACHE_VERIFIED，8项输出身份闭合；人工数值不重算。实际几何面积为49／53／51、197／199／201、441／441／453像素，不宣称名义等面积已成为严格栅格等面积。

### 第3阶段：D_fit输入生成与CPU有效性核验〔已完成：独立身份适配及全量核验；v1失败保留〕

1. 先核准备缓存；仅为精确工作重复检查，对旧N／N_gen训练背景核来源hash并CPU建立规范工作hash，不展示、提取筛选统计或评分。再只读D_fit60原图并核原hash，按RGB bilinear256／float32处理，每背景仅解码一次；核角色内及旧训练工作内容不碰撞。Q／F／D_hold图像不打开。
2. 固定每背景9个几何的M_geo／alpha，先完成全部60背景的Lb及可定义性核验；至少一个几何可定义才计一个可定义背景，另报告9/9、部分、全无与各几何数量。同一背景几何的所有方向／幅度共用条件，不按弱档筛选；少于40背景时在正式刺激生成前停止。
3. 每请求生成一次，保存实际float输入／残差、裁切和零变化状态、代理特征及名义／实际分母；不重试、移位、换样或改幅度。
4. 保存内容去重清单，区分请求与唯一输入；D>0和D>1/255并列，不让旧训练标签门槛删掉弱探针。
5. 全量CPU核验后封存本阶段独立输入。原fit_inputs_v1因身份口径错误失败保留，实际独立适配输出为fit_inputs_identity_v2。少于40个背景具有可定义测量或有技术异常，停止设计审阅；不能依源图好坏补样。

通过条件：所有7560名义请求都有结果／状态；实际唯一输入、非零变化与可定义数量按运行结果报告。没有模型评分，不能据此声称有检测曲线或预测效果。

首轮失败历史：[第3阶段CPU生成与停止诊断](D:/华兴源创/报告/Mura响应研究第3阶段CPU生成与停止诊断.md)。隔离resources verify通过；generation独立入口、显式目录白名单及ROI代数／裁切核验已写入，但该次未执行完整真实输入语义核验。首次旧训练图`display/train/good/0002.png`原文件hash通过后，在不成立的“新原始RGB hash=旧组ID”断言处停止。该次实际旧训练解码1、D_fit解码0、render0；9份几何NPZ及四份事务JSON保留于[失败目录](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_v1)，无complete。失败实现与配置不回改，不在该目录续做。

独立补充已实施：[身份适配补充方案](D:/华兴源创/计划/Mura第3阶段独立身份适配与重新取得补充方案.md)、[第3阶段独立身份适配与CPU取得结果](D:/华兴源创/报告/Mura响应研究第3阶段独立身份适配与CPU取得结果.md)。另建generation_v2.py及fit_inputs_identity_v2，分开四类身份；60背景／540条件全部可定义且无工作碰撞，门槛前render0，随后7560请求各生成一次。7245非零、315零变化、2070弱非零且旧M_eff空均保留；7620引用对应7305唯一输入。新运行来源解码220，连同旧0002一次累计221事件／220路径。目录白名单、ROI代数／裁切、全部测量、总解码／逐路径计数及7620引用交叉核验通过。

最终产物：[完整输入](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2)、[覆盖](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/coverage.json)、[全部请求](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/manifest.json)、[实际生成核验](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/verification.json)、[完成清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/complete.json)。独立进程核14882份产物、全部数组及语义通过，新增原图打开／解码与render0；总目录含complete为14883文件，实际约5.376GiB。已完成的是数字输入，未取得检测器响应或代理预测结果，不自动进入第4阶段。

### 第4阶段：新正常工作点准备、评分与F检查〔已完成：两模型分别封存并核验〕

准备部分已完成：[第4阶段正常工作点准备结果](D:/华兴源创/报告/Mura响应研究第4阶段正常工作点准备.md)。独立双模块绑定两模型、Q/F原序、D_fit既有完整证据及新实现；[协议](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/protocol.json)、[来源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/sources.json)、[静态核验](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/verification.json)、[完成清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/complete.json)共4文件封存。准备期所有0的历史保留，固定原分数／预处理、eval／FP32／batch1／workers0、G1 stateless BN与逐图状态不变要求不改；两模型后续实际运行已分别核验，不能把原静态核验混同实际运行。

CPU输入已完成：[实际取得与独立核验结果](D:/华兴源创/报告/Mura响应研究第4阶段CPU正常输入取得与核验.md)、[正常输入](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1)、[来源与碰撞](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/records.json)、[实际计数](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/complete.json)。固定Q200后F100全部300来源各打开／解码1次，核文件hash／长度／尺寸／模式、新原始RGB与规范float工作摘要；旧组ID只作角色／排除。全300工作唯一且无与D_fit／旧训练碰撞，305文件完整封存；独立进程逐一核hash及300数组内容通过，新增原图解码0。该CPU步骤未重读D_fit或旧N原图／数组，未加载／前向或定tau；D_hold一直未解封。

PatchCore实际工作点已核验：[结果报告](D:/华兴源创/报告/Mura响应研究第4阶段PatchCore正常工作点与核验.md)、[新Q阈值](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/q_result.json)、[Q／F结果](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/result.json)、[状态／资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore/complete.json)。发现用户手动完整缓存后，本轮只读verify-normal通过，未再次调用score-normal；已有300次正常输入评分、1次构造ones探测分别记录，参数／159命名buffer和状态前后一致。新tau=1.852951288223265，Q同池FP10／TN190／200；F100 FP6／TN94，Wilson双侧95%约2.7786%—12.4768%。606文件完整，本轮新增前向／实际校准／tau重算0；Q不称独立验证，F不回调tau，不外推总体误报或缺陷检出能力。

G1实际工作点已完成：[结果报告](D:/华兴源创/报告/Mura响应研究第4阶段G1正常工作点与核验.md)、[新Q阈值](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/q_result.json)、[Q／F结果](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/result.json)、[状态／资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/G1/complete.json)。按同一Q200→F100各评分一次，G1自己的新tau=0.6587551027536391；Q同池FP10／TN190／200，F100 FP7／TN93，Wilson双侧95%约3.4319%—13.7495%。300真实输入、构造ones探测0、49个stateless BN／空buffer和最终参数逐图不变。完整606文件封存后另起进程verify-normal通过，新增前向／模型加载／实际校准0、tau_recomputed=false。没有重跑PatchCore、打开其热图或读D_fit数组，不以F调整tau。两模型的600次输入评分仅同300正常背景，不能累计独立分母或作能力排名；normal_scoring_v1仅作两子目录容器，各模型complete独立封存，不另写联合完成文件。

通过条件：来源与参数状态闭合，每模型200校准＋100检查分母完整；同池Q计数不称独立验证。Q实测速度与内存用于说明后续耗时，不用于改刺激或选模型。

### 第5阶段：D_fit评分与统计模型冻结〔PatchCore评分／CPU响应及用户人工拟合已完成并核验；G1刺激与拟合未执行〕

1. 逐模型新进程评分D_fit基线与全部冻结唯一刺激；RGB相同引用同一缓存。评分函数不读取区域；保存完整原生分数及float热图。
2. 只读核验缓存后，在CPU按固定M_geo计算H_alarm、H_joint及配对响应，并展示每个基线的9几何背景联合报警；零实际变化、不可定义代理和覆盖数全部展示。
3. 先只准备analysis.py及已实现人工入口，由用户显式手动启动CPU统计模型拟合。按设计拟合全部四候选、主k及两敏感性k，连续特征只用D_fit标准化，同时冻结背景等权p0常量参照；保留全部系数、数值条件和收敛状态，不选最佳拟合形式，助手不代为启动拟合。
4. 封存fit_analysis_v1。主链不可识别／未收敛则停止，不解封D_hold；补充G1不能拟合时明确NA，不借其结果调主模型。

通过条件：拟合与分析口径完全冻结，D_hold原图／分数未读取。高拟合分数不称留出预测好；统计拟合不更新检测器。

独立评分准备已完成：[第5阶段D_fit只读评分准备与核验](D:/华兴源创/报告/Mura响应研究第5阶段D_fit只读评分准备与核验.md)、[协议](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scoring_preparation_v1/protocol.json)、[唯一输入](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scoring_preparation_v1/unique_inputs.json)、[全部引用](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scoring_preparation_v1/references.json)、[虚构标量](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scoring_preparation_v1/scalar_verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scoring_preparation_v1/complete.json)。七文件封存，独立verify-preparation返回FIT_SCORING_PREPARATION_CACHE_VERIFIED，本轮各类数组／模型／前向及校准／拟合计数0。7620引用对应7305唯一输入，元信息与Q/F重合0；后续每模型新增7305次真实输入，315零变化引用背景、60背景不是7620独立样本。原float不再次除255或量化；预测不读区域，实际H_joint与配对差留给独立CPU后处理。未来只用各自正常工作点tau，不重取分位数。原CPU准备曾给出PatchCore单模型score入口；现已有构造失败，旧入口不重跑。新一周第1天来源准入与只读定位已完成，第2天A单次尝试已提前停止，启动身份补充方案已写，下一项仅其阶段1—2准备与CPU静态核验，不自动G1、响应后处理或拟合；构造适配、真实输入状态和长运行仍待明确范围核验。

### 第6阶段：D_hold生成、评分与CPU留出评价〔CPU输入已核验／评分准备完成；实际模型评分及预测评价未开始〕

1. 身份先核：确认D_fit模型／分析已封存，再读取固定D_hold60。先完成全部60背景的工作hash碰撞及9几何Lb覆盖核验；少于40个至少一个几何可定义的背景，或出现精确工作重复时，在正式刺激生成／GPU评分前停止。条件满足后才用相同CPU代码和全部预定请求生成一次，保存hold_inputs_v1；不为凑可定义分母补背景。
2. 按已实现人工入口逐模型评分全部唯一输入、基线各一次。完整核验后封存hold_scores_v1；未知前向保留，不自动补推。两模型不取成功样本交集。
3. CPU从完整缓存计算每背景Brier／log loss、四候选及p0参照的配对差、固定10个等宽概率可靠性组和2000次背景bootstrap；差定义为SEMU-only误差减参照误差，负值才表示较好。全部预定表示及k均报告，不能用H重拟合、选k／特征／阈值或删不利背景。
4. 保存hold_analysis_v1及全部逐背景预测误差／重复，明确名义15120总请求和本阶段7560请求、实际可定义／非零／唯一输入与背景数。

通过条件：完整呈现预测价值、代价及不能估计项；不是“SEMU赢了”才通过。两检测器结果是同一批背景，不能当两个独立实验；区间只是给定拟合下的背景重采样近似。

### 第7阶段：Python图件、专题报告与结论收束〔未开始〕

1. 只读冻结结果，画数据覆盖／零变化／裁切、完整预定网格的响应曲线、四表示留出预测误差和配对差、可靠性图；若没有实际跨越，不强行报a50／a90。
2. 图像规则在第2阶段即固定：D_fit第一个背景作为输入说明；D_hold不按好坏挑图，按固定请求序首个亮／暗圆、水平／斜椭圆及首个基线报警例去重，原因合并；若无相应事件写NA。同类热图统一范围，PatchCore与G1不同分数尺度不强行同色阶，原图不拉伸。
3. 报告写入`报告/数字Mura检测器响应与SEMU代理刻度研究报告.md`，分开当前数字域结果、历史审计和未来真实迁移。不把旧KSDD2收益当当前Mura证据。
4. 同步本计划唯一摘要、阶段结果与第六节一条下一步指令；按证据判断是否止于数字域技术报告、具备测量论文初稿条件，或需另立真实桥接方案。不自动开始训练／真实评价。

## 五、资源、停止与当前交付

CPU开始至少2GiB可用RAM；GPU开始至少4.5GiB RAM／4GiB空闲显存；全轮磁盘至少40GiB空闲。PatchCore已有运行Q段507.851043秒、每输入参考平均2.539255秒，所属事务839.295159秒，GPU allocated峰值366.806MiB／reserved410MiB，最小采样可用RAM8.225193GiB。G1第4阶段Q段192.137966秒、每输入参考平均0.960690秒，所属事务295.614997秒，GPU allocated峰值675.207520MiB／reserved756MiB，最小采样可用RAM6.638119GiB。Q时间均含状态核对／IO，排除实际分位数；所属事务不含先前CPU输入核验与最终hash封存，不能视为纯GPU延迟或严格速度对比。GPU峰值从CUDA准入后构造前重置起覆盖构造及300输入，不是全设备占用；最小可用RAM是系统余量，不是进程峰值。共享正常输入0.220GiB及CPU取得17.891841秒／最小采样RAM5.665802GiB等历史证据保留，不混作GPU数据。后续D_fit速度仍待实测，不能保证按正常输入线性外推。继续用D盘，不写E盘。 第5阶段准备实测RAM6.517578GiB／D113.305687GiB／空闲显存7899MiB；按旧Q含状态与IO的线性参考，7305输入约PatchCore5.15小时／G1 1.95小时，实际D_fit耗时与峰值仍未实测，两模型float热图净载荷约3.567GiB，额外元信息及事务余量另计。

科学与技术停止条件见设计第十节。中断／未知事务不删除、不自动恢复前向；来源异常或代码问题按systematic-debugging只查因，不用新数据／seed／规格救结果。旧阈值、指标及原审核结论保持不变。

当前实际交付：

- [独立研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)：用户已确认，已绑定冻结准备来源。
- 本分阶段执行计划：第1—4阶段完成；PatchCore第5阶段评分、CPU响应与用户手动统计拟合均完成核验，G1刺激／拟合未执行。第6阶段D_hold CPU输入完整核验与独立评分准备均完成，实际模型评分及留出预测评价未开始；第7阶段图件／报告未完成。首轮构造与诊断失败保留，本周真实链未执行，不将历史停止记录当当前PatchCore主链状态。
- [第1阶段资源与设计准入](D:/华兴源创/报告/Mura响应研究第1阶段资源与设计准入.md)：真实资源与元信息证据。
- [第2阶段CPU准备结果](D:/华兴源创/报告/Mura响应研究第2阶段CPU准备结果.md)：角色、数值核验、边界与实际入口。
- [第3阶段CPU生成与停止诊断](D:/华兴源创/报告/Mura响应研究第3阶段CPU生成与停止诊断.md)：实际1次旧训练图解码、0次D_fit解码／生成及hash口径错误证据；没有完整输入。
- [第3阶段独立身份适配与重新取得补充方案](D:/华兴源创/计划/Mura第3阶段独立身份适配与重新取得补充方案.md)：静态实施依据，已按A—D完成，不回改其封存正文。
- [第3阶段独立身份适配与CPU取得结果](D:/华兴源创/报告/Mura响应研究第3阶段独立身份适配与CPU取得结果.md)：真实覆盖／计数、全部强度／几何／方向分母、零变化及弱非零保留、资源、独立只读核验证据与已实现运行说明。
- [第4阶段正常工作点准备结果](D:/华兴源创/报告/Mura响应研究第4阶段正常工作点准备.md)：新协议／显式入口、两模型5资源字节身份、固定Q/F角色、只读与事务口径、虚构标量、资源快照和独立准备核验证据；该准备步骤的原图／数组／模型／新阈值操作均0。
- [第4阶段CPU正常输入取得与核验](D:/华兴源创/报告/Mura响应研究第4阶段CPU正常输入取得与核验.md)：该CPU步骤实际300次原图取得、工作碰撞0、305文件／300数组完整核验、各类计数分开、资源及首模型人工命令；该步骤时未启动模型评分。
- [第4阶段PatchCore正常工作点与核验](D:/华兴源创/报告/Mura响应研究第4阶段PatchCore正常工作点与核验.md)：该次发现已有手动完整运行后只读复用；新tau、Q10/200、F6/100／区间、300＋1构造计数、参数／buffer不变及计时／峰值保留。该次报告内G1未运行表述是历史，不回改旧报告。
- [第4阶段G1正常工作点与核验](D:/华兴源创/报告/Mura响应研究第4阶段G1正常工作点与核验.md)：本轮300次输入、新tau、Q10/200同池计数、F7/100／区间、49个stateless BN／空buffer、参数不变及实测计时／峰值，随后独立只读核验新增前向0。
- [第5阶段D_fit只读评分准备与核验](D:/华兴源创/报告/Mura响应研究第5阶段D_fit只读评分准备与核验.md)：独立实现／协议、7305唯一输入及7620引用、Q/F重合0、原float与状态口径、虚构标量、资源／停止规则、实际独立核验证据和首模型人工命令；未读取图像数组或启动模型。

当前已实现resources、generation_v2、inference及fit_inference各显式入口。正常CPU输入、两模型工作点和D_fit评分准备均完整核验；fit_inference score／verify-scores已实现且静态核验；实际PatchCore score尝试在构造阶段因临时写入被守卫阻断，目录有failure／inflight但无complete、记录及热图。正常完整缓存不再评分或取实际分位数；旧失败入口不重试。hold评分／CPU响应后处理／analysis未实现，不提供未实现命令。旧失败不续接，未知事务保留查因。

实施取舍：阶段各绑独立源码，resources永久守卫仅在隔离元信息进程作用；新增模块不回改封存准备。第3阶段旧失败及计数保留，第4阶段两正常工作点实际运行／独立核验与资源范围分开。第5阶段复用冻结静态来源检查器与原生构造／预测，不改其全局路径；CPU只核JSON/hash/stat及虚构标量，既有全量数组证据不是本次重新取得。一次新上下文审阅发现PatchCore运行时buffer元组与JSON列表直接比较会误失败，封存前仅新模块统一JSON域，虚构签名原比较不等／规范后相等，初始／逐图／最终统一。实际Torch／FAISS兼容、载荷及GPU状态／耗时仍待授权运行，不以静态通过冒充实际通过。没有过程表／软件测试／提交流程；旧冻结成果保留，所有训练与统计拟合由用户手动启动。



阶段1—2实际最终产物：[准备与静态核验报告](D:/华兴源创/报告/Mura第2天A启动身份适配阶段1—2准备与静态核验.md)、[独立模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_launch.py)、[三文件准备缓存](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_launch_v1)。静态30情形是准备阶段证据，阶段3—4实际状态见本页当前摘要；第2天B未执行。准备可用RAM7.9077GiB／D113.1988GiB，动态前重核≥2／40GiB。详见报告的真实独立核验证据、审阅修正、取舍与已实现命令。



阶段3—4实际结果：[一次启动握手与导入诊断报告](D:/华兴源创/报告/Mura第2天A一次启动握手与导入诊断结果.md)、[17文件封存现场](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_launch_v1)。仅本次实际一层启动器成立；首事件是NUL设备打开而非原Temp写入。运行内封存／另进程hash核对与原verify CLI失败分别表述，阶段4仍有独立核验入口障碍；单次预算已消费，保留全部源码／现场，不改环境或放宽后重试。后续仅第六节一条方案编写指令。



独立只读核验适配后续：[方案](D:/华兴源创/计划/Mura第2天A独立只读结果核验身份适配方案.md)已编写并自审，方案编写时未开发；现阶段A已准备import_diagnosis_result_verification独立标准库模块与结果根，核历史事务环境而不要求当前工具沙箱临时路径相同；完整hash／来源／token关联／有限进程关系／退出证据及原CLI失败状态保留。阶段A新模块／准备三文件已建立；实际适配验收仍未运行，不提前改写通过，当前仅第六节B—C指令。



独立只读核验适配阶段A已完成：[方案及实际准备证据](D:/华兴源创/计划/Mura第2天A独立只读结果核验身份适配方案.md)、[新三文件准备根](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_result_verification_v1)。732来源hash／9目录及48人工情形通过，另进程仅verify-preparation通过；尚未运行阶段B的实际17记录验收或新完整缓存。旧原verify失败不改，下一项仅第六节B—C指令。



### 阶段B—C最终结果（2026年11月3日）

[独立只读结果核验适配报告](D:/华兴源创/报告/Mura第2天A独立只读结果核验适配结果.md)已保存；[独立6文件结果根](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_result_verification_v1)完整，用户run保存passed=true，本次另进程verify返回INDEPENDENT_COMPLETE_CACHE_VERIFIED，未重复run或诊断。32项顶层必要检查通过，嵌套环境13项、身份20项及原16项复算一致，不累计为独立实验。原17文件／旧失败／模型／tau不变，原verify失败保留；NUL仍不是原Temp写入，依赖完整导入和模型评分尚未恢复。新结果完整后仅用已实现verify只读核验，不重跑run-once或修改环境；报告、缓存和当前下一步均已同步。



静态审阅最终意见：[成功与停止调用链审阅及方向建议](D:/华兴源创/报告/Mura成功与停止调用链静态审阅与方向建议.md)。已区分守卫mode／flags覆盖差异、原Temp调用者未知、NUL实际栈及导入期辅助对象求值；现有通用导入诊断不能当原构造等价复现。当前只提出一项固定离线运行边界设计建议，不开发或新增运行；下一项仅第六节方案编写指令。



固定WRN候选最终设计：[最小运行边界方案](D:/华兴源创/计划/固定WRN离线PatchCore最小运行边界方案.md)，C01=NO_GO。延迟通用注册表导入是唯一候选，不能解决其前面的Torch／NUL禁界；FAISS导入期辅助求值也保留。当前没有获准的实现或动态阶段，不创建候选模块／缓存，不重跑或继续扩展导入诊断；保持科研未完成项，第六节仅一条停止／保留指令。

### 运行边界变更（2026年11月4日，已完成文档同步）

用户已明确解除上述项目级启动限制，授权内容已写入[项目要求](D:/华兴源创/AGENTS.md)。本次实际产物是该要求与五份关联计划的摘要／边界说明／下一步同步；没有修改任何冻结源码、配置或缓存，没有导入模型或执行前向。下一步改为一次独立最小推理，不继续扩展旧导入诊断；下方仅保留一条当前指令。

### 固定PatchCore最小离线推理（2026年11月4日，已完成并独立核验）

[最终结果与运行说明](D:/华兴源创/报告/PatchCore解除启动限制后最小离线推理结果.md)及[独立完整缓存](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_minimal_runtime_v1)已保存。实际1次构造／1次V1加载／1次作者ones探测／2次输入评分，完整缓存独立verify新增前向0；81份绑定旧来源文件及5个历史目录清单未变。按原tau两例均未超过阈值，不据此评定整体能力；没有自动全量D_fit或真实评分。下方一条指令取代已完成的最小推理任务，旧NO_GO和禁止条款保留原条件历史含义。

### PatchCore D_fit独立全量评分（2026年11月4日，实际完成／独立CPU全量核验通过）

[最终评分与独立核验证据](D:/华兴源创/报告/PatchCore独立D_fit全量只读评分执行与核验.md)及[完整封存缓存](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_scoring_v1)已保存。原会话正常退出exit_code=0；7305唯一输出（7303新增＋2复用）和7620引用全量通过已实现verify，独立核验新增前向／来源RGB读取／tau重算均0。run complete=501b547812f6cb941e8f4f8ed7f92046c8451e330d5698302fb77d7578428f7e，原模型／BN／tau与来源身份保持。没有自动响应统计、G1、D_hold或真实评价；旧失败现场继续保留，评分完整性不等于科学效果结论。

### PatchCore D_fit独立CPU响应后处理（2026年11月4日，实际完成／独立全量复算通过）

[最终响应结果与运行说明](D:/华兴源创/报告/PatchCore独立D_fit缓存响应后处理结果.md)及[独立封存缓存](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_response_v1)已保存。run与独立verify分别解码7305热图／9掩码，完整7620引用；H_alarm3564、H_joint3107，315零变化／7245非零／2070弱非零／未定义0均保留，540视图仅对应60基线，全部预定分项输出。核验完整重算逐项一致，新增模型前向及拟合0；原评分和tau不改。主H_joint有3107个1／4453个0，仅满足后续拟合的初步双类条件，尚不证明候选收敛或SEMU有用。下一项只准备用户人工统计拟合入口，不自动拟合或解封D_hold。

### PatchCore D_fit独立CPU统计拟合（2026年11月4日，用户手动完成／独立核心数值核验通过）

[实际拟合与独立核验说明](D:/华兴源创/报告/PatchCore_D_fit统计拟合准备与人工运行说明.md)及[完整冻结结果](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_fit_analysis_v1/run)已保存。六个预定拟合均success，p0=0.4109788359788335，run complete=983524aab83fd1b99c641f103bf50adf6d93efe2c543a2883ab3c04bf534c649。助手只运行verify-fit且新增优化器0，核心数值及来源通过；梯度范数／rank等诊断字段仍按已记录非阻断范围限制保留。已有模型／tau／响应和源码不改，没有自动D_hold、G1或真实评价；D_fit惩罚目标不是留出预测成绩。下一项仅准备D_hold CPU输入取得／生成协议，实际后续拟合仍由用户手动启动。

### PatchCore D_hold60独立CPU输入与生成准备（2026年11月4日，准备完成／独立CPU核验通过）

[准备、实际边界与运行说明](D:/华兴源创/报告/PatchCore_D_hold60输入与生成准备说明.md)及[七文件独立准备](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/preparation)已保存。prepare和另进程verify-preparation通过，complete=4056569613a57ffb2308c545cdc45dd3193cecf6eff0eb0e78a0838a8e0c6d8c；绑定六个冻结拟合、p0=0.4109788359788335、原tau=1.852951288223265及原请求／几何／float口径。完整碰撞对照池520条旧背景摘要，40背景预生成停止门槛及零／弱／未定义保留规则已实现；756人工数组请求及39／40、碰撞边界核验通过，不是实际D_hold结果。原图／数组读取、正式render、前向、训练、校准与拟合均0，run尚不存在。新acquire／verify-inputs已实现但实际材料分支未运行；两项非阻断限制（守卫失败时可能仅留部分目录、资源无进程峰值）据实保留于说明。原源码／环境／输入／权重／tau／响应／拟合和旧失败不改，不自动进入取得或评分。

### PatchCore D_hold60 CPU输入取得与独立核验（2026年11月4日，完整缓存只读复用通过）

[实际结果与独立全量核验证据](D:/华兴源创/报告/PatchCore_D_hold60_CPU输入取得与独立核验结果.md)及[完整封存输入](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_inputs_v1/run)已保存。任务开始已有完整run，本次不再acquire，仅verify-preparation后另进程verify-inputs；全部7116唯一RGB、9几何、7560残差／请求与7620引用闭合，run complete=5cbc1351afe334cefd23486aa35efd2143a9e38def506f385876c3e48d31e355。原序60背景／540几何条件全可定义、无角色内或跨520条旧背景路径工作碰撞；504零变化／7056非零／2016弱非零／未定义0均保留。原取得记录各原图打开／解码一次、各请求render一次；本次新增上述事件均0，模型／训练／建库／校准／拟合0。原tau、六拟合／p0／变换、preparation和旧成果不改。数字输入完整不等于检测或SEMU预测有效；后续先准备D_hold评分入口、内容去重复用身份及新增前向预算，不自动评分或真实评价。

### PatchCore D_hold60独立只读评分准备（2026年11月4日，准备封存／另进程CPU核验通过）

[准备、预算与已实现命令](D:/华兴源创/报告/PatchCore_D_hold60独立只读评分准备与运行说明.md)及[六文件独立准备](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_hold_scoring_v1/preparation)已保存，complete=b496678690ed7394d54b3038f1e43676f5634892afb9c7aa34b25e1285b1a870。prepare与另进程verify-preparation通过；元信息核7116唯一RGB与旧正常300／D_fit7305重合0，原参数／BN状态一致不能替代内容条件，故旧输出复用0、预定新增7116输入前向＋1作者构造探测。原7620引用／7560请求／60基线、504零变化／7056非零／2016弱／未定义0及原tau保留。独立审阅无Critical／Important／Minor；预封存新消费者字段适配后静态通过，未修改旧schema／报告。实际数组／模型／前向／校准／拟合均0，仅preparation存在；run／verify实际评分缓存路径未实测。后续只按显式入口评分、完整后另进程核验，不自动响应整理、统计评价、重校准或训练。

## 六、当前唯一下一步指令

> 用 $superpowers:executing-plans 按原第6阶段执行PatchCore D_hold60独立只读评分。先用已实现patchcore_hold_scoring verify-preparation核独立准备（complete=b496678690ed7394d54b3038f1e43676f5634892afb9c7aa34b25e1285b1a870）、冻结输入与全量核验证据、六拟合／p0／D_fit变换、原V1／N80／预处理／评分／tau和资源；再仅用run按固定7116唯一float RGB原序各评分一次，7620引用／7560请求／60基线完整，旧正常及D_fit重合0所以新增输入前向7116、作者构造ones探测1另计。保留原最大patch分数／作者float32热图、eval／FP32／batch1／workers0、逐图参数／BN状态不变，不PNG回读／再除255，预测只消费RGB。完整后另起CPU进程verify全量只读核验；若已有完整同身份run只核验复用、不重复评分，未知／部分／failure／inflight保留停止，不重跑或自动补推。只完成评分，不CPU响应／H_joint统计、预测评价、bootstrap、生成、训练、建库、校准、改tau或统计拟合，不读其他角色图像／数组／旧热图或真实评价，不改冻结成果。按实际状态同步同一计划及一条下一步指令；全部训练及统计拟合仍由用户手动启动。
