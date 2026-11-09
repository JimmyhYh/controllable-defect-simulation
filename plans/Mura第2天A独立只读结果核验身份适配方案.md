# Mura第2天A独立只读结果核验身份适配方案

日期：2026年11月3日。使用superpowers:writing-plans。本次仅编写方案与同步计划，不开发、不创建下述新代码／结果目录、不导入依赖或启动诊断。本文件的技术条款为未来实施依据，旧方案与产物保持原样。

> 供执行者：采用superpowers:executing-plans按明确授权的阶段执行。用户要求优先于技能默认模板：不用勾选追踪、过程表、软件测试套件、Git／提交流程；人工字典与源码静态核验是必要准入证据，不启动动态试跑。

**唯一当前摘要（2026年11月4日，PatchCore D_hold60独立只读评分准备已完成／另进程核验通过）：新增patchcore_hold_scoring，六文件preparation封存，complete=b496678690ed7394d54b3038f1e43676f5634892afb9c7aa34b25e1285b1a870；绑定完整D_hold输入、独立证据、六拟合／p0／变换、原V1／N80／参数与BN／预处理／评分／tau。固定7620引用／7560请求／60基线对应7116唯一float RGB，与旧正常300及D_fit7305内容hash重合0，预定旧输出复用0、新增输入前向7116、作者ones探测1另计。7716项来源及人工标量核验通过，数组／热图／概率／原图读取、模型导入／加载／前向、生成／训练／校准／拟合均0；当前仅preparation，没有run。实际GPU评分／完整评分缓存分支未实测，SEMU留出预测结论尚无；下一项仅显式评分与只读核验，不自动响应统计。全部冻结成果不改，训练及统计拟合仍由用户手动启动。**

当前适用规则：本页下方关于NUL／必要临时文件／系统查询／辅助初始化的禁止和NO_GO，均是旧授权条件下的历史记录；今后独立运行以[AGENTS.md运行边界变更](D:/华兴源创/AGENTS.md)为准。该变更不重置旧诊断单次预算、不允许重跑旧失败入口或改写旧缓存，也不自动扩大图像读取范围。解除的是项目脚本额外拦截，不能越过设备或执行环境的实际权限。

**目标：** 独立核验既有诊断记录的来源、身份关联及有界结束，保留历史运行内部环境一致性要求，同时允许当前核验会话具有不同沙箱临时路径。

**架构：** 新增一份纯标准库核验模块，直接消费17文件及固定只读来源，不导入旧诊断模块或调用其verify／run。历史证据与当前核验观测分别保存；以固定hash、实际字段关系和原判定口径闭合记录，在全检查通过时只向新目录封存独立核验结果。

**技术栈：** 现有同一.venv／Python3.12.7，标准库pathlib、json、hashlib、ast、importlib.metadata、os及资源查询所需ctypes；无下载、安装、第三方导入或模型运行。

**规范依据：** [本次诊断结果报告](D:/华兴源创/报告/Mura第2天A一次启动握手与导入诊断结果.md)、[原启动身份方案](D:/华兴源创/计划/Mura第2天A独立启动身份适配与导入诊断补充方案.md)第二至第四节与封存协议；[一周计划](D:/华兴源创/计划/Mura检测器能力说明一周执行计划.md)。本方案只适配历史结果核验，不修改原启动／导入权限。

## 一、全局边界与已有事实

私人设备、公开资源、个人学习与非商业研究，公司材料隔离。训练与统计拟合始终由用户手动启动。本方案不授权任何训练、拟合或模型调用。

固定事实：控制PID29528／PPID26964，Popen PID21000，Python child PID31104／PPID21000；LAUNCHER_ONE_HOP，GO约0.234秒，进入导入约0.250秒，总约1.109秒。仅NumPy完成导入；Torch导入触发OUTSIDE_WRITE，open目标为Windows空设备NUL、flags130、mode=null，已拒绝。两个自有进程退出码0，原17文件包含complete，实际单次预算1已消费。

原complete SHA256固定为942429386ad21600c67f0c970c94c1dbd48318023376d9b7c2e5a9df5474887c；原模块import_diagnosis_launch.py为af4566d5100c698b59ccaff885b8454b76b80f7b35ac17a5ebff37bbace4a905；原准备清单为fdde56acd66795fab97649871495611e52fb12de498163c4608b7f13a18228b1。旧import_diagnosis.py为32e6a3972fba7c5b16a0495277d337873797783601b8fb05946269d15cbe59ed，包初始化为1adba9096b6adb0df8c1064ab74baf70825478009cefe3183c5d50ea3752a2fa。

本方案不重新证明这些历史行为发生过，不把hash当数字签名或外部真实性认证；它检查固定本地记录未变、来源符合、关系与结论一致。原会话缺失的PID／token分项、原Temp调用者仍未知。

NUL不是原Temp写入。本次栈是Torch → platform.machine／_syscmd_ver → subprocess._get_devnull，设备打开前已停止。后续platform系统命令／子进程路径仍在禁界内；不提议直接放行NUL、预热platform、变更TEMP／TMP、重跑或重置预算。核验通过也不授权第2天B。

## 二、文件与接口

以下代码／目录均为未来新增，本次不创建；不改任何旧入口、模块全局路径、vendor、环境或模型文件。

| 文件／目录 | 职责与权限 |
| --- | --- |
| D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_result_verification.py | 唯一新模块；独立纯比较、来源读取和本轮输出，不导入原import_diagnosis或import_diagnosis_launch |
| D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_result_verification_v1 | 唯一新缓存根；不认领未知目录，不向原17文件根追加文件 |
| D:/华兴源创/报告/Mura第2天A独立只读结果核验适配结果.md | 实施后的最终报告，阶段C才生成；不改本次诊断报告 |
| 原17文件根、旧6文件根、day1、原PatchCore失败及两正常工作点 | 只读来源；原complete和失败标记保持 |

拟定接口，不是已经实现的命令：

| 接口 | 输入与返回 |
| --- | --- |
| read_sources() -> dict | 固定根／hash；返回来源清单及规范JSON，不执行其中路径或代码 |
| compare_recorded_environment(bundle: dict, current: dict) -> dict | 返回运行内必要项、准备／历史差异、当前差异和各项原因 |
| compare_recorded_identity(bundle: dict, sources: dict) -> dict | 纯字典计算共同项、DIRECT／LAUNCHER_ONE_HOP与历史结论一致性 |
| verify_recorded_transcript(bundle: dict, sources: dict) -> dict | 计算来源／预算／许可／时限／事件／退出等分项；false／null保留原因 |
| prepare() -> dict | 绑定新实现及旧来源，人工字典静态核验；仅写新准备三文件 |
| run() -> dict | CPU读取完整固定记录；独占本轮核验事务，只写新结果 |
| verify_cache() -> dict | 完整同身份缓存只读复核，不动态运行／重新生成 |
| inspect／prepare／verify-preparation／run／verify | 拟定独立CLI子命令；只有实际实现并静态核验后才提供PowerShell命令 |

不通过AST提取／exec旧函数或猴子补丁绕过原env条件；新模块按本方案独立实现比较。旧源码仅作hash及文字依据。

新根成功白名单固定6文件：sources.json、protocol.json、preparation_complete.json、run_started.json、verification.json、complete.json。准备阶段恰3文件；实际核验前独占写run_started；失败时verification可保存完整拒绝分项，但不生成complete。崩溃／部分结果原样保留，不续写救现场、不删除或另开v2。新complete只能表示“本方案独立核验完成”，不能改写原verify失败或原诊断状态。

## 三、必须保留的核验规则

### 3.1 来源与完整hash

原目录严格17文件：sources.json、protocol.json、preparation_complete.json、started.json、popen_observation.json、child_boot.json、child_identity.json、parent_identity.json、decision.json、import_permission.json、child_result.json、stdout.txt、stderr.txt、result.json、verification.json、complete.json。文件不得缺失、增加、重复引用、重定向、跨目录或成为目录／链接；路径祖先也核重定向。固定complete hash后，核其16份输出实际hash及精确键集合，不能只相信清单自述，也不更新hash让变化过关。

JSON严格拒绝重复键、NaN／Infinity、错误类型、缺失必要字段；布尔不冒充整数／PID。stdout按已封存UTF-8消息读取，保留64KiB单行、4MiB单流边界；核HELLO／DONE各一次、会话／协议一致、无重复或未知控制消息，stderr按封存原样核身份。不得把任意依赖输出解释成许可。

从固定来源核原两诊断源码与包初始化、准备3文件、day1五文件及其独立证据、旧6文件、原PatchCore failure／inflight与空records／heatmaps、两正常工作点complete及原tau。沿既有固定清单全核695文件hash和9目录stat签名；数组目录只stat，不解析载荷。五份模型资源仅作不解释的文件字节hash，不执行pickle／解析tensor。对清单路径先核固定工作区／Python来源范围与既有锚定hash，不任意跟随JSON路径访问数据。

新来源另绑本方案固定技术正文、单个新模块与解释器／依赖元信息；不以可变摘要和第六节下一步作为冻结技术身份。现有.venv可执行文件、pyvenv.cfg、基础解释器和启动器、核心依赖元信息仍须符合封存身份；不因放开当前临时路径就放开源码或解释器匹配。只读取dist-info元信息，不导入对应包。

### 3.2 三种环境分别比较

| 环境层 | 实际来源与判定 |
| --- | --- |
| 历史诊断运行内 | started.environment、child_boot.environment、child_identity.boot.environment、parent_identity.observations.launch.environment及其boot、first_event.temporary_environment须完整一致，TEMP／TMP字符串原样比较、TMPDIR显式null。保留child_result.environment_unchanged=true和双方相应分项，并核各副本与原始来源一致 |
| 准备／更旧会话 | protocol.current_environment是准备进程环境，protocol.legacy_environment是更旧记录；仅关联来源和说明差异，不强制等于实际诊断运行。原准备与运行的沙箱GUID本来不同 |
| 当前独立核验进程 | 记录当前TEMP／TMP／TMPDIR及逐项差异；可以和历史运行不同，不使历史身份失败。不设置、不清空、不替换环境，不为“相同”重建沙箱 |

历史运行任一必要环境字段缺失、不同或记录分项矛盾，仍停止。当前环境不同只放宽“不同会话临时路径必须相等”这一要求，不能忽略历史运行内差异。不以当前PID／PPID等于旧PID为要求，也不查询旧PID当前是否存在。

### 3.3 token与来源／进程闭合

核started的token_sha256为64位十六进制；stdout HELLO／DONE的session与其一致，协议hash对应固定protocol文件；child boot的sources／protocol／started hash对应实际文件；相同boot／checks在child_identity、HELLO、parent_identity间一致，decision及result一致。子／父已保存token_match等必要分项必须true；保存为空／false则拒绝。

原token值仅经当时管道发送，没有保留原值。**只读核验无法再次计算原token原像，也不能重新执行当时challenge或内核句柄查询。** 输出须分开“可重新计算的hash／字段关系”与“只能复核其一致性的封存运行时比较”。后者标为历史观测，不编造新鲜nonce验证；不索取／生成新token。

独立纯比较按原规则重算：共同条件全真且DIRECT或LAUNCHER_ONE_HOP一个完整成立，未知拒绝。实际必须还与已保存LAUNCHER_ONE_HOP及GO一致。核控制PID／PPID来源一致，Popen返回PID=句柄PID，child PID／PPID及父侧child句柄记录一致，初始双方映像／hash与固定解释器集合、创建时间一致，启动器先于child。当前本地source与已保存source身份也核hash。

用保存的进入前alive／映像／创建时间及双方观测，不调用OpenProcess、GetProcessId、进程枚举或命令。退出证据按各自PID和创建时间关联到同一保存身份，要求alive=false、exit_code=0；退出后映像查询错误31及null字段可按本次明确已知结果保留，不能被拿来填补进入前缺失身份。必要PID／创建时间／退出状态缺失即失败。

### 3.4 许可、时限、事件与状态

从保存字段重算原16项一致性，不仅接受verification中16个true；核decision.action=GO的共同条件、消息检查、单HELLO、无未知队列与日志先保存标志，import_permission的own_checks及checks全真才permitted。实际entered_imports=true须有保存许可。

以started.submitted_monotonic_ns及entry_deadline_ns核实际保存进入时间，进入上限20秒，result.elapsed_seconds≤120秒且非负有限。本次未单独保存HELLO收取时刻：复核消息deadline分项，并以保存的decision.elapsed_seconds≤15秒作为HELLO截止的保守上界，明确这是记录次序下的上界而非重新测得的收取时间；若上界不能支持15秒，即停止，不只相信布尔。不以当前monotonic时钟、当前PID或文件mtime重造过去时序。必要时间字段缺失／矛盾即停止，不用报告约数补字段。文件hash／已保存“先落盘”标志只能证明记录一致性，不能认证真实I/O先后。

核父侧Popen观察时间≤HELLO／GO判定时间≤实际进入／结束；在时钟分辨率内允许相等，不能为符合报告四舍五入而改值。started预算与result提交数都1，actual_Popen_returned=true，hello_count=1，failure=null，管道closed=true、errors全null。原报告约数只作描述，不作替代阈值。

first_event须与child_result.first_event逐字段相同；kind=OUTSIDE_WRITE、during_import=torch、open目标为本次原始NUL设备路径、mode=null／flags130、保存栈和依赖元信息完整；child_status=FIRST_GUARD_EVENT，完成导入序列严格为[numpy]，DONE状态一致。禁止把捕获事件改写成原Temp写入或导入成功。不触碰目标设备、不执行栈内函数或验证命令。

若原verification、complete的identity_passed等结论与独立重算矛盾，则失败保留；不在新输出中替旧false改true。正常完成的独立输出至少分别报告：

- source_integrity／recorded_environment／recorded_identity／permission／bounded_exit／event_consistency的逐项结果与原因；
- token_preimage_revalidated=false、historical_kernel_queries_reexecuted=false；
- current_environment_difference_observed及字段，独立于历史事务通过与否；
- original_verify_cli_status=FAILED_ENVIRONMENT_COMPARISON_PRESERVED；
- recorded_diagnosis_status=FIRST_GUARD_EVENT、completed_imports=[numpy]；
- original_temp_caller_confirmed=false、model_ready=false、new_diagnostic_launches=0。

## 四、资源、缓存及停止规则

未来CPU准入沿原可用RAM≥2GiB、D空闲≥40GiB；只同一.venv／-B，不下载安装或探测GPU。hash流式读取，时长尚未实测，不将原1.109秒诊断耗时当核验耗时。当前核验环境差异可记录，资源不足仍停止；不降门槛或换环境。

准备不存在才创建新根；恰3文件且身份一致只核验复用。已有run_started而无完整complete、未知成员、部分／无效JSON或不明确事务一律停止，不自动续核、覆写、清标记、复制重算或新版本回避。

完整6文件同身份缓存只读复核：实际读前后hash与目录成员、原17文件和所有来源仍一致；本次新核验进程环境可以变化，不要求与新缓存“生成时当前环境”相同，也不改写旧输出。返回只读缓存核验的当前观测供报告说明，封存生成时观测保持历史含义，避免把同一问题搬到新缓存。

源17文件和保护目录在核验前后都核身份；中途变化停止，不封存complete。新verification若有false／null／无法按规定确认的必要项，保存失败结果和来源后停止、不生成complete；纯说明字段如“当前TEMP不同”“token原像未重验”不混作失败必要项。新complete引用新根全部5份产物hash并绑定原complete；后续只读验证完整性而不重新运行原诊断。

最终通过用词固定为“独立只读封存记录核验通过”，不称“原verify修复”“Torch／检测器准入”“原Temp根因确认”或“真实材料／屏幕效果通过”。

### 自审重点及其人工情形

| 风险 | 归属与未来必须核对的人工情形 |
| --- | --- |
| 环境放宽过度或新缓存重复旧问题 | 阶段A：历史相同／当前不同应通过环境比较；历史不同即拒绝；准备环境不同不误拒；完整新缓存换当前沙箱只读复用 |
| hash自证或坏JSON漏检 | 阶段A：内容hash变化、漏／增成员、重复键、NaN、重定向、路径越界分别拒绝，不能改清单救样本 |
| 当前PID被误当历史身份／退出映像缺失 | 阶段A：当前PID无关；真实进入前映像缺失拒绝；本次保存退出错误31且PID／创建时间完整可核；未知退出拒绝 |
| token断链／布尔假通过 | 阶段A：HELLO不同session或协议hash拒绝；无法重验原值必须明确，不伪造；缺失必要记录即停止 |
| 超时、事件矛盾与部分缓存 | 阶段A：迟进入、>120秒、NUL与child副本不符、完成导入多于NumPy、已开始缺complete分别拒绝且不复跑 |

## 五、分阶段执行与交付

### 阶段A：独立准备及CPU静态核验（准备与静态核验完成）

1. 只读核本方案与原报告、17文件、固定complete／来源／旧现场；检查新代码／输出不存在或属于可核同身份缓存，资源满足。遇到未知版本停止，不修改旧源码。
2. 在唯一新模块定义上述纯比较及入口，禁止模型／旧运行模块导入、子进程、写环境、动态内核进程查询、数据／tensor读取；只允许固定元信息和不解释文件hash。
3. 在新protocol明确本次专用状态、三类环境、所有必要项与说明项、原17清单、时限／错误31、复用／拒绝规则。新sources固定全部来源及实现hash，输出新准备三文件。
4. 只做源码编译／AST与上表人工字典核验。至少展示相同与不同环境、两个合法有限关系和未知关系拒绝、token关联失败、完整／部分缓存、事件／时限矛盾；不启动握手／模型子进程或实际阶段B核验。
5. 独立CPU元信息进程只读核准备身份与静态规则，保留全部人工情形结果。产物为新模块及三文件准备、阶段状态说明；不是独立历史结果验收完成。
6. 只有实际实现且静态通过后，提供阶段B已实现入口与资源条件，不自动运行。不提供本方案尚未实现的PowerShell命令。

接口衔接：阶段A产生sources／protocol／preparation_complete和纯比较返回结构，阶段B必须使用同一实现身份、明确区分必要检查与说明字段。

### 阶段B：实际独立只读封存记录核验（用户手动完成；独立核验通过）

1. 复核阶段A身份、资源、原17精确清单，独占run_started；当前TEMP差异只保存，不改环境。
2. 只CPU执行第三节来源／历史环境／摘要与进程／许可／时限／事件／退出检查，保存每项真／假／未知及可独立重算程度；不调用旧verify／run或外部命令。
3. 全检查通过才保存新verification和complete，失败保存可得结果后停止；保留原17文件及原CLI未通过状态。核验前后源身份不变。
4. 新独立进程运行已实现verify只读复核完整新缓存，当前沙箱差异不能再次使历史结果失败。若部分事务不明确，不自动补写或复跑；结果分别写清生成时与复用时观测。

产物：独立6文件完整核验缓存，或明确失败／部分现场；不取成功交集、不强行得到通过。

### 阶段C：最终说明及计划同步（已完成）

将来源完整性、历史比较／新观测、原CLI失败、NUL首事件与尚未知项写入第二节指定最终报告。注明证据仅限固定一次本地记录，没有新依赖准入、原Temp定位、模型构造或真实材料评价。保留旧报告，更新本方案和两个总计划及原启动方案的唯一摘要、对应产物与一条下一步；不自动选择下一次诊断或第2天B。

### 当前文档自审与交付

本次自审核对需求覆盖、接口类型、字段名、五类边界、6文件缓存与阶段交接。三种环境均有实际来源，尤其准备protocol环境不误归为诊断环境；token原值和历史内核观测不可重演的限制独立列出。固定17输入、完整来源hash与两旧失败保留，退出错误31不替代进入前身份，原verify状态不改。以上为原文档自审范围；阶段A现已有下方实际静态证据，阶段B验收不提前宣称。

原方案编写时仅文档工作；现已按阶段A新增独立模块与三文件准备，完成一次源码审阅及CPU静态核验。当前仅下方第六节B—C一条下一步指令，不自动执行。


### 阶段A实际产物与核验证据（2026年11月3日）

本段保留阶段A交付时状态；B—C当前结果见下方。阶段A完成时B／C未执行。仅新增[独立标准库模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_result_verification.py)与[新准备根](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_result_verification_v1)的sources.json、protocol.json、preparation_complete.json三文件；没有run_started、实际verification或最终complete。第二节“拟定／未创建”是方案编写时表述；当前以本段和顶部摘要为准，不回改已绑定的第二至第四节。

模块SHA256：f2c6dbf010a629dc865ba4810c63cecd4939175975a11b0c94c3e79f9386196e；新准备清单：caf71f3cfdc44e0d2aadcc0396c67c1f36630829e9e18729e258403bade11559。CPU来源核对732项文件hash（包含原695项、五份模型仅不解释字节hash）与9目录stat签名；另核41项旧元信息文件与保留身份一致，原17文件／complete、旧两失败、两tau及原源码均不改。来源准入不等同阶段B运行内字段验收，未把实际17记录送入完整验收函数。

首轮40人工情形通过；一次独立源码审阅提出两项Important：缓存长读取缺少最终前后比较、JSON溢出及布尔／环境类型可误接受。修正前人工输入证实1e999与布尔计数会被接受，且快照辅助接口缺失；封存前一轮修正后，48个人工情形及编译／AST全部通过。静态输入覆盖两关系、历史环境／当前环境区别、hash／成员变化、坏JSON／溢出／精确类型、token关联、许可／时限／退出、NUL副本及完整／部分缓存。只在虚构字典上调用完整比较函数，不验收实际17记录，不运行真实完整缓存路径。

取舍：审阅将协议缺少显式参数列为Minor，本轮按用户“协议明确时限／错误31／历史观测界限”的要求提升为必需项，已在同轮补齐15／20／120秒、错误31例外、可重算与只能复核的证据及停止规则；不留下已知协议缺项。代价是显式协议须与源码同时保持一致，现由来源／实现hash绑定。依用户要求不建技能默认ledger、测试套件或Git流程，以本最终阶段说明及准备协议保留实际证据。无延后Minor。实际阶段B验收和真实完整缓存复用尚未运行，静态通过不能证明这些路径已实测，也不改变旧CLI失败结论。

独立CPU元信息进程已运行verify-preparation，返回PREPARATION_VERIFIED，48／48人工情形通过，stage_B_executed=false、新诊断子进程0。prepare资源RAM5.3445GiB／D113.1575GiB；核验进程RAM5.4996GiB／D113.1573GiB。门槛仍≥2／40GiB；来源准入含流式hash与路径保护，inspect／prepare／verify-preparation各超过初始10秒等待后完成，未测阶段B耗时，不以原1.109秒诊断耗时推算。

以下是阶段A曾提供的阶段B入口，作为历史运行说明保留；目前已有完整6文件，请使用最终报告中的verify只读入口，不将run作为当前下一步：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.import_diagnosis_result_verification run
```

仅准备检查使用同模块verify-preparation，不读实际验收bundle；仅阶段B已有6文件完整结果才使用verify。部分／未知事务不补写、不自动改pin／样本／环境或另开版本。该句为阶段A历史边界；本轮B—C已按保存结果只读核验并形成指定最终报告，未新增过程文件。




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

