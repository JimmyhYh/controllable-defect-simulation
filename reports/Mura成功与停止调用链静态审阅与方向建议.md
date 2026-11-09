# Mura成功与停止调用链静态审阅与方向建议

日期：2026年11月3日。按superpowers:systematic-debugging仅执行现有证据回溯与模式比较；没有复现故障、运行人工事件、导入依赖、启动子进程或修改实现。本报告为最终审阅意见，不是新诊断、修复或模型准入。

**判断：Mura检测器能力说明仍值得继续；当前“不断扩展纯导入诊断再补核验”的路径应收束。先明确固定离线PatchCore运行边界，再决定是否存在符合现有禁界的最小适配。本次不批准新运行。**

## 四条调用链的实际差别

| 链 | 保存状态与可证结论 | 核心差异／限制 |
| --- | --- | --- |
| 正常工作点PatchCore／G1 | 保存验证均passed，分别300次输入前向；PatchCore另1次作者ones探测，G1无探测；原tau保留 | 文件守卫只看mode字符串，不检查整数flags；“blocked为空”不证明覆盖了所有事务外写入 |
| D_fit PatchCore构造停止 | model_loads=1、tensor_loads=0、作者探测0、输入前向0、完成记录0；inflight为model_construction，Temp路径被拒 | 仍调用同一build_model，但写入检测增加flags；没有原调用栈或mode／flags记录，确切Temp调用者未知 |
| 第一轮独立导入诊断 | 在token与child_ppid联合身份条件处退出，尚未记录child_started、安装child导入守卫或执行序列 | 是启动身份障碍；不能判断任何依赖的导入行为 |
| 第二轮独立导入诊断 | 一层启动器身份／GO成立，仅NumPy导入完成，Torch导入NUL打开被拒；原17文件完整，新6文件独立核验通过 | 冷启动导入有更严格的文件／子进程／函数调用禁界，省略原GPU／RNG等操作；不是原构造链的等价复现 |

两个正常模型测的是同一批Q200／F100，不是600独立正常图。这里引用已有计数和验证JSON，不读取概率、原图、GT或重算阈值／指标。model_loads在真正模型完成加载前递增，不能把D_fit的1当作成功构造一次。

## 已证实：不是简单的“正常链晚装守卫”

[正常入口](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference.py:408)先install_guards，后进入inference_runtime；[正常score_normal](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference_runtime.py:310)才调用build_model。因此正常评分并非在所有依赖导入完成后才安装文件守卫。

[D_fit入口](D:/华兴源创/实验/弱缺陷Demo/mura_response/fit_inference.py:643)也先装守卫，再执行score；[score的534行](D:/华兴源创/实验/弱缺陷Demo/mura_response/fit_inference.py:534)直接复用同一inference_runtime.build_model，没有在当前封存实现中另换骨干或预测方法。它导入旧inference模块作静态辅助，但未调用旧模块的main／install_guards，不能据此认定两套永久文件守卫都安装了。

真正明确的差别是写入判定：[正常file_guard](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference.py:75)仅用mode中的w／a／x／+；[D_fit file_guard](D:/华兴源创/实验/弱缺陷Demo/mura_response/fit_inference.py:101)同时看args[2]的O_WRONLY／O_RDWR／O_CREAT／O_TRUNC。正常链允许三个阶段4输出根，D_fit仅允许本模型事务根；独立诊断还会拒绝系统命令／新子进程及操作性构造调用。

所以，一个mode为空而flags具有写权限的open在正常守卫中不会因“写入”分支被拒，却可能被D_fit／独立诊断拒绝。这是源码可确定的覆盖差异，不需通过重新打开文件验证。旧blocked_attempts=[]只表示当时实现没有记录到拒绝，不能推断没有所有副作用，也不能据旧成功放宽现禁界。原正常评分／tau作为既有结果保持，不重写旧完成标记或历史指标。

## 原Temp写入：缩小范围，但仍缺直接调用者

[原failure](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_scores_v1/PatchCore/failure.json)记录事务外路径C:\Users\Huang Yuhang\AppData\Local\Temp\7stlrcex，仅有路径和错误，没有完整Python栈、mode／flags。模块尾迹最后包含timm.models._pretrained，模型权重加载、作者探测和输入评分均未到达；不能归因于7560请求的图像处理或真实缺陷内容。

[build_model](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference_runtime.py:182)顺序为：NumPy／Torch → GPU／RNG／线程设置 → model_loads递增 → faiss／torchvision → patchcore.common／patchcore.patchcore／sampler → 显式wide_resnet50_2(weights=None) → tensor_loads递增及本地V1加载 → PatchCore.load／作者探测／FAISS库。计数把原停止范围限定在权重加载计数前，但没有足够证据精确指定是哪一条导入或构造语句。

源码存在一条候选依赖支路：patchcore.patchcore顶层第12行导入[backbones](D:/华兴源创/实验/弱缺陷Demo/vendor/patchcore-inspection/src/patchcore/backbones.py:1)，backbones顶层导入timm；[timm._builder](D:/华兴源创/实验/弱缺陷Demo/.venv/Lib/site-packages/timm/models/_builder.py:14)导入_hub，[_hub](D:/华兴源创/实验/弱缺陷Demo/.venv/Lib/site-packages/timm/models/_hub.py:32)再导入huggingface_hub相关功能。这与导入尾迹相容，只是候选路线，不是被实际栈确认的Temp调用链。

[CPython tempfile._get_default_tempdir](D:/Python/Lib/tempfile.py:183)确实用os.open试写随机文件来探测临时目录；但“8字符路径看起来相似”不能证明旧事件就是它，更不能锁定其上层调用者。本次搜索和源码追踪没有补出原动态栈，可选依赖／原生扩展、历史缓存及实际调用上下文仍未知，不认定huggingface_hub或timm为已证实根因。

## NUL事件与导入期构造是另外两层边界

[第二轮first_event](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_launch_v1/first_event.json)直接证实Torch._load_dll_libraries → platform.machine／_syscmd_ver → subprocess._get_devnull → os.open(NUL)。mode=null、flags130；守卫在请求发生前拒绝。这是空设备打开，不是Temp文件创建，也没有证明系统命令成功启动。

新诊断在[main第979行](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_launch.py:979)装child_audit，在GO后第695行装调用级profile，再按固定序列导入；原build_model中的GPU、RNG、线程／后端调用均省略。原正常／D_fit与该诊断的进程环境、初始化历史和守卫范围不同，不能假设它们必然在相同的第一处事件停止，也不为重现去预热platform或改TEMP。

还有一个独立的源码事实：[patchcore.common.NearestNeighbourScorer](D:/华兴源创/实验/弱缺陷Demo/vendor/patchcore-inspection/src/patchcore/common.py:297)的函数默认参数创建FaissNN(False,4)；[PatchCore.load](D:/华兴源创/实验/弱缺陷Demo/vendor/patchcore-inspection/src/patchcore/patchcore.py:36)及load_from_path的注解也有求值构造。FaissNN.__init__会执行faiss.omp_set_num_threads。

这些是导入阶段求值的辅助对象初始化，与神经网络前向或权重加载不同；然而当前[forbidden_call](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_launch.py:244)对作者目录中的__init__作宽匹配。即使未来越过NUL，若推进到这些定义，也有后续已知禁界，不能把“完整导入但一切构造都不发生”当无条件可达到的验收目标。这是静态条件判断，本次没有真的走到或触发它们，也不改规则放行。

## 是否继续与唯一有界建议

继续研究有价值：已有固定模型与正常工作点、D_fit输入和清晰科研问题；当前缺少的是可在明确权限下执行的运行路径。现有阻断不证明PatchCore检测能力差，也没有检测效果或模型损坏的证据。记录核验适配已经闭环，再重复验证相同记录不会恢复评分。

**唯一建议：另写一份“固定WRN离线PatchCore最小运行边界”静态设计，先判定是否可行，不再扩展通用导入探针。** 目标只限现有显式WRN／本地V1／原N80库／原预处理与评分；把必需初始化、可避免的通用依赖和禁止行为逐项对应，形成是否能继续的准入结论，而不是直接放行一张白名单。

设计的具体出发点是：当前显式路径已直接传入torchvision WRN，通用backbones解析仅在[load_from_path](D:/华兴源创/实验/弱缺陷Demo/vendor/patchcore-inspection/src/patchcore/patchcore.py:266)使用，现路径不调用该pickle入口。可审阅独立派生实现能否延迟这条未使用的注册表依赖，保留现有模型计算；但这只排除一个不必要依赖候选，既不证明它就是Temp调用者，也不同时解决Torch冷导入或FAISS导入期初始化。

必要限制：保留所有旧vendor、环境、库、权重、tau和失败现场；保持flags检查，不放行NUL／platform系统命令，不预热或移后守卫掩盖事件，不偷偷借用已加载Torch。不同时提出改包版本、改临时目录或更多动态试跑。若静态设计不能在这些硬条件下容纳必需初始化，应明确“当前运行分支不可准入”并停止，不再连开诊断版本。

该建议只能识别显式推理是否不必加载通用骨干注册表、导入与显式初始化是否有可分边界；不能在本次静态审阅中证明无副作用、状态／数值等价、模型可构造或预测收益。未来任何代码、导入、构造、权重加载或前向都需另行明确方案和授权；没有本次可运行命令。若设计通过，后续也应先一个单因素、有界验证，不能直接重跑7305唯一输入或全量真实评分。

## 本次范围与保存

本次仅阅读上述源码、报告、已有日志／JSON／路径stat并核源码身份，没有调用任何项目运行／核验入口，也不导入NumPy／Torch／FAISS／模型。没有读RGB／GT／NPY／NPZ／概率或权重／pickle内容，没有模型前向、生成、训练、校准、统计拟合、环境修改、子进程或系统命令执行；系统shell仅用于授权的文件读写。

原17文件、新6文件、旧6文件、day1、原模型失败及正常工作点元信息保持，冻结源码不改；最终文件身份核对见计划中的本次审阅完成状态。本报告只新增最终意见文档并同步计划摘要、产物和一条下一步，不新增过程表、软件测试或提交流程，不自动第2天B。所有训练及统计拟合仍由用户手动启动。

