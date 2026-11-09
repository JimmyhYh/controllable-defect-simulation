# Mura能力说明第2天A：依赖导入诊断与启动停止结果

日期：2026年11月3日。依据：[一周计划](D:/华兴源创/计划/Mura检测器能力说明一周执行计划.md)第2天A。私人设备、公开资源、非商业学习，公司材料隔离。

**诊断代码准备及静态核验通过，唯一一次诊断子进程在新增启动身份校验处提前失败，尚未进入依赖导入循环。没有捕获原事务外临时写入调用栈；原故障未解决，本阶段目标未完成。已经保留现场，没有重试、放行临时写入或执行第2天B。**

这次提前失败来自助手新增的诊断前置校验；没有把实测子PID／PPID及两个比较分项先保存，是诊断设计的缺项。它不能归因于检测器效果、数据、学习率或用户设备性能。

## 一、准备与实际执行边界

先只读核第1天5文件、已有独立证明、695份保护文件hash和9目录stat、旧失败、两正常工作点及原tau，均一致。权重仅流式字节hash，未解析tensor／pickle；未读取源RGB／GT／数组或概率。

新增[独立诊断模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis.py)及无运行副作用的包初始化文件。未修改mura_response、旧vendor／环境、模型、阈值或旧失败目录。

模块SHA256为32e6a3972fba7c5b16a0495277d337873797783601b8fb05946269d15cbe59ed；初始化文件SHA256为1adba9096b6adb0df8c1064ab74baf70825478009cefe3183c5d50ea3752a2fa。准备清单SHA256为14216ceeaaa4495f3e3f61f3204351f140cac086bc5e61e5c68b93c2faa31433。

prepare返回IMPORT_DIAGNOSIS_PREPARED；另进程static-check返回IMPORT_DIAGNOSIS_PREPARATION_VERIFIED。源码编译／AST及虚构事件核对覆盖：整数flags型事务外写入、自有JSON写入、图像／权重／数组打开、操作型CUDA调用、模型模块构造的禁界。没有在这些检查中导入模型或实际执行虚构文件操作，也没有新建软件测试文件或测试流程。

准备时资源快照：可用RAM6.080002GiB、D盘113.204433GiB，满足CPU2GiB／D40GiB门槛。更早的本轮预核快照为RAM7.035454GiB，GPU元信息为RTX5060 Laptop／总显存8151MiB／空闲7512MiB／驱动621.01；仅查询元信息，不调用Torch CUDA接口，也不代表实际兼容。

协议预定的依赖顺序为NumPy → Torch → faiss → torchvision.models（仅取wide_resnet50_2符号，不调用）→ patchcore.common → patchcore.patchcore → patchcore.sampler。原构造前CUDA／RNG及线程／后端设置均省略并记录。原模型构造、加载和前向不属于诊断权限。

## 二、唯一一次实际尝试

仅提交一次run。父进程先写started，记录diagnostic_child_launch_attempts=1，再通过项目.venv解释器提交独立诊断子进程。父入口最终退出码1；保留的子进程stderr显示：

```text
RuntimeError: Child launch identity differs; preserve state; no automatic retry or substitution
```

停止于[child启动检查](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis.py:380)：

```python
started["token"] == token and started["parent_pid"] == os.getppid()
```

该检查先于child_started写入、诊断审计／profile安装和依赖导入循环。因此本次没有到达预定的模型包导入，也没有捕获首次事务外写入。这是由stderr停止行及固定源码顺序确定，不能把不存在的child_result当成已保存的包导入计数。

原计划允许“其他异常保留并停止”，单次诊断预算已经消费；没有改条件后再试，没有设置TEMP／TMP、预热tempfile或创建第二个诊断版本。

## 三、现场及核验的实际范围

[import_diagnosis_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_v1)保留6文件：

| 文件 | 状态／含义 |
| --- | --- |
| sources.json | 第1天／原失败／工作点／新实现／依赖及解释器身份 |
| protocol.json | 单次导入、禁界、环境继承及停止规则 |
| preparation_complete.json | 仅准备2文件的封存清单 |
| started.json | 父进程已提交1次诊断的身份与临时环境 |
| stdout.txt | 0字节 |
| stderr.txt | 890字节，子进程启动身份检查异常 |

没有child_started、first_event、child_result、最终result／verification或complete。**preparation_complete只证明准备完成，不代表导入诊断已完成。**不向该半成品现场补写complete，不把它改名为成功或在其中续跑。

原PatchCore fit_scores_v1失败仍是构造阶段、0评分／0热图／无complete；本次没有调用旧score或build_model。两故障必须分开：原问题是事务外临时写入；新增诊断现在卡在更早的启动身份检查，不能据此确认或排除原调用者。

实际子PID／PPID及token、PID比较的各自布尔结果没有被该版本保存。started中的期望parent_pid为28764，但不能凭此推断实测父PID或是哪一项不等。没有子进程导入／审计记录，不能声称已验证整个导入过程的副作用隔离。

## 四、只读追踪：Windows启动器是候选解释

本地.venv/pyvenv.cfg记录home及基础解释器为D:/Python；只读查询sys.executable为项目.venv/Scripts/python.exe，sys._base_executable为D:/Python/python.exe。

[本地venv源码](D:/Python/Lib/venv/__init__.py:281)明确说明Windows虚拟环境复制启动器。项目.venv/Scripts/python.exe与D:/Python/Lib/venv/scripts/nt/python.exe均为270104字节，SHA256同为137c474a645a9471ab3a369084e7cd573cd8b7ff8f81cb85363fffdf5904106c。

这些事实支持“虚拟环境启动器可能引入进程层级，因此不能无条件假定Python子进程的os.getppid等于原父Python PID”的候选解释。**本次没有保存实际进程链，不能把启动器已证实存在直接升级为已证实的失败根因。**当前沙箱临时环境也不同于旧模型失败时的普通Local/Temp，不能承诺以后导入会重现同一路径。

## 五、保留范围与有界后续建议

本次不修原临时写入问题，也不偷偷删掉新PID条件后重试。建议下一项只审阅独立的启动身份适配方案，明确：

1. 保持同一.venv与核心依赖，审阅可兼容启动器的身份闭合方式，例如实际Popen进程身份、一次性token／只读来源绑定及父子双方记录；不为绕开问题换基础解释器或注入旧site-packages。
2. 在拒绝前保存token比较结果、双方PID／PPID、Popen返回PID、解释器路径／hash、源码身份及相关临时环境；日志先于判定，未知分项不能被总异常遮蔽。
3. 保留独占事务及仅一次子进程预算，不只删除父PID检查。启动器允许关系须明确；不能说明时停止，不把任意祖先进程链都放行。
4. 旧模型失败及本次诊断源码／准备／6文件现场均保留。新配置和输出若有必要，须在后续明确方案中独立确定，本次不开发、不创建或运行新版本。
5. 启动身份被验证后，未来一次依赖导入仍须在首次事务外写入前捕获并拒绝，不扩系统Temp权限，不调用CUDA、模型构造／前向、权重加载或取图；这仍只是诊断，不是修复通过。

只有身份适配和后续明确权限落实，才能重新争取原写入调用栈。第2天B的真实CPU取得可与模型路径隔离安排，但本次没有执行，也不自动以其替代未完成的诊断目标。

## 六、入口状态与复用规则

实际已实现prepare、static-check、run和verify；内部_child入口需要已拥有的单次启动身份。当前状态如下：

- static-check仅只读核准备／身份／静态规则，不能表示导入或模型通过；其diagnostic_children_launched=0指本次元信息调用新增0，不否认started已有1次提交。可运行命令如下。
- run已尝试一次。当前存在非完整事务，源码会拒绝再次提交；本次没有再次调用它。
- verify只接受最终complete，目前不具备该条件；没有用它冒充失败恢复或完整结果核验。
- 没有修复后的新导入、评分或模型运行命令，不提供尚未实现的入口。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.import_diagnosis static-check
```

本次独立保存状态核验使用只读元信息入口和固定源码，不提交新的诊断子进程。没有执行软件测试套件、训练、生成、校准或拟合，不创建过程表、工作树或提交流程。

## 七、独立只读保存状态核验

实际在新元信息进程核准备绑定、新源码和6文件hash、原695文件hash和9目录stat，确认原成果未变；没有模型或数组包导入。该核验没有重新执行诊断，也没有创建最终complete。返回如下：

```json
{
  "status": "IMPORT_DIAGNOSIS_EARLY_STOP_PRESERVATION_VERIFIED",
  "source_preparation_status": "IMPORT_DIAGNOSIS_PREPARATION_VERIFIED",
  "diagnostic_child_launch_attempts_saved": 1,
  "one_child_returned_error": true,
  "failure_stage": "child launch identity check before import loop",
  "raw_error": "Child launch identity differs",
  "original_write_stack_captured": false,
  "dependency_import_loop_reached": false,
  "import_loop_evidence": "saved stderr line and bound source order; child runtime import log was not produced",
  "child_started_or_result_or_complete_exists": false,
  "strict_preserved_files": {
    "preparation_complete.json": {
      "bytes": 311,
      "sha256": "14216ceeaaa4495f3e3f61f3204351f140cac086bc5e61e5c68b93c2faa31433"
    },
    "protocol.json": {
      "bytes": 6338,
      "sha256": "a92623429e2fe8295cf26fd40e6ac2c832b8d539f7832fd84506698ef82a6655"
    },
    "sources.json": {
      "bytes": 4161,
      "sha256": "c8897856743260a7673b5aa36dd9e6de6474392f97324aa88d51e7f2cc179942"
    },
    "started.json": {
      "bytes": 504,
      "sha256": "3fbeabb030553ae0f067c255fbed70d070a48544b56a261927056347c5a5ce98"
    },
    "stderr.txt": {
      "bytes": 890,
      "sha256": "9ed91183a756955080bc280ec68ca50cbba7edb34a83120034c005ded7f92de2"
    },
    "stdout.txt": {
      "bytes": 0,
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
  },
  "new_module_sha256": {
    "D:\\华兴源创\\实验\\弱缺陷Demo\\mura_capability\\import_diagnosis.py": "32e6a3972fba7c5b16a0495277d337873797783601b8fb05946269d15cbe59ed",
    "D:\\华兴源创\\实验\\弱缺陷Demo\\mura_capability\\__init__.py": "1adba9096b6adb0df8c1064ab74baf70825478009cefe3183c5d50ea3752a2fa"
  },
  "protected_original_source_files_verified": 695,
  "protected_original_directory_stat_signatures_verified": 9,
  "old_diagnostic_failure_unchanged": true,
  "original_model_scoring_or_repair_ready": false,
  "expected_parent_pid_saved": 28764,
  "actual_child_pid": null,
  "actual_child_parent_pid": null,
  "token_match_subcondition_observed": null,
  "PID_match_subcondition_observed": null,
  "Windows_launcher_hypothesis": "local venv source and launcher identity support possibility, not a measured parent chain",
  "new_model_imports_for_this_preservation_verification": 0,
  "new_child_launched_for_verification": 0,
  "source_RGB_GT_or_input_probability_array_reads": 0,
  "scope": "read-only metadata/stat/byte hashes; no model, CUDA, generation, training, calibration or fitting",
  "runtime_preparation_resource_gate": "not a model compatibility pass"
}
```

这个结果只证明现场和来源可追溯，不证明首次事务外写入已捕获、实际调用者已找到、修复成功或模型可评分。下一步指令统一维护在执行计划第六节。

## 八、最终独立审阅与处置

完成一次新上下文整段只读审阅。Critical无；Important两项均是本报告已经披露、仍未修复的诊断缺项：联合启动身份条件阻断唯一尝试，拒绝前没有保存实际PID／PPID、Popen PID及token／PID分项。审阅确认报告没有将联合失败错误归因于PID或Windows启动器，也没有把静态准备通过写成诊断完成。

处置裁决：用户明确要求仅一次诊断子进程、其他异常保留且不自动修复，因此本次不改封存的新诊断实现、不补complete或追加运行。代价是第2天A目标仍未完成，原写入调用者继续未知；新代码不是可用于再次尝试的合格版本。后续只写独立身份适配方案，不以审阅意见冒充新运行授权。

Minor一项暂留：计划接口表的“prepare／static-check只读准备”措辞不精确。首次prepare实际建立并写3份准备文件，只有static-check是只读核验；本报告已准确列出副作用，本次不修改该表作额外润色。后续方案须使用准确名称，不据该短语扩大权限。

审阅仅读源码／报告／现场文字并核新2份源码与6文件hash，没有运行入口、复现或读权重／图像／数组。原695文件和9目录的保护结论由独立保存核验实际记录承担，不由审阅替代。实测进程链、具体失败子条件、导入兼容、CUDA／模型禁界的动态有效性、原写入调用者和模型能力均未得到本轮证据，继续记未验证，不改判通过。
