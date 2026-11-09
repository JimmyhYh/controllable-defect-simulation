# Mura第2天A：一次启动握手与导入诊断结果

日期：2026年11月3日。按superpowers:executing-plans执行已授权阶段3—4。只运行一次新诊断，不修改冻结实现／环境／准备／旧现场，不进入第2天B。

**结果：本次身份握手通过，Torch导入在首个事务外打开请求处被守卫拒绝；运行内记录已完整封存，另进程元信息核对通过，但既有verify CLI未通过。阶段4不能称全部完成，原Temp故障未定位或修复。**

## 实际执行与完整性

先运行既有static-check：30人工情形及全部冻结来源核验通过；可用RAM5.4219GiB、D空闲113.1592GiB，满足≥2／40GiB。新根当时仅3份准备文件，无started；随后仅调用一次run-once --allow-diagnostic-imports，没有修改TEMP／TMP、预热、换环境或重试。

诊断约1.109秒收束。控制PID29528／PPID26964，Popen返回和句柄PID21000，Python child PID31104／PPID21000。Popen实际映像是项目.venv启动器，child实际映像是已绑定基础解释器，路径、hash和创建时间及同一.venv运行身份均核实。共同条件全真，LAUNCHER_ONE_HOP完整成立；DIRECT不成立是未选分支，不当作共同失败。GO约0.234秒，实际进入导入约0.250秒，均在15秒HELLO／20秒进入许可与120秒总预算内。

双方先保存观测及逐项结果后GO，token仅保存摘要；未展示或重置旧token。两个自有进程均确认退出、退出码0，管道无溢出／读取错误，result.failure=null。退出后QueryFullProcessImageNameW返回错误31、映像字段为null，原样保留；进入许可前持有的句柄／映像／创建时间证据有效，不能用退出后空字段假称进入前未核身份。

[本次封存根](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_launch_v1)有16份被complete引用文件＋complete，共17文件；complete SHA256为942429386ad21600c67f0c970c94c1dbd48318023376d9b7c2e5a9df5474887c。运行内verification的16项一致性检查通过。完整标记表示该次诊断记录收束，**不表示依赖、模型或检测通过**。旧6文件／原PatchCore failure与inflight保持，不改旧预算。

## 捕获的事件与真正含义

[首事件](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_launch_v1/first_event.json)：

| 字段 | 实际值 |
| --- | --- |
| kind | OUTSIDE_WRITE |
| during_import | torch |
| audit_event | open |
| target | Windows空设备NUL（原始路径见first_event） |
| mode／flags | null／130 |
| 已完成导入 | 仅NumPy |
| 未完成 | Torch；faiss、torchvision.models和作者依赖均未进入 |

保存的Python调用链：torch.__init__._load_dll_libraries（237行）→ platform.machine → uname → win32_ver → _win32_ver → _syscmd_ver（283行）→ subprocess.check_output → run → Popen.__init__ → _get_handles → _get_devnull（1137行）→ os.open(os.devnull, os.O_RDWR) → 审计拒绝。

局部源码核对依据：[Torch初始化](D:/华兴源创/实验/弱缺陷Demo/.venv/Lib/site-packages/torch/__init__.py:237)、[platform](D:/Python/Lib/platform.py:283)、[subprocess](D:/Python/Lib/subprocess.py:1137)。flags130含读写访问及Windows二进制标志；守卫按事务外写入权限拒绝。**这是打开空设备的请求，不能写成已向磁盘或Temp写入文件。** 已进入依赖内部Popen初始化路径，但在获取空设备句柄时即停止；不能说已成功启动系统命令，也不能因只读系统查询的用途而直接放行后续子进程禁界。

本次事件先于原先待查Temp写入，原Temp文件调用者未被本次执行触达。新会话证实一层启动器，仍不能倒填旧会话实际PID／PPID／token比较或认定旧启动失败根因。没有模型构造、权重tensor／pickle解析、业务CUDA调用、模型前向、RGB／GT／数组／概率读取、生成、训练、校准或统计拟合；依赖原生内部行为不作全面认证。

## 阶段4的分项状态与只读排障

| 证据层 | 当前结果 |
| --- | --- |
| 运行内seal及verification | 通过，17文件封存 |
| 另进程原verify CLI | 未通过；两次只读调用同因停止，无新增诊断 |
| 另进程独立元信息／hash核对 | 通过，16输出hash、来源、保存字段／退出与运行内环境核对 |
| 依赖全部导入／原Temp调用者／模型准入 | 均未通过或未取得证据 |
| 本轮整体阶段4 | 未全部完成，保留障碍 |

既有verify在[第928行](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_launch.py:928)要求当前核验进程env()与历史started.environment完全相等。实际运行时控制／child／first_event中的TEMP／TMP相同，child.environment_unchanged=true；另一次工具进程的沙箱GUID不同，TEMP／TMP比较false，TMPDIR同为null。于是来源及内容未变，原CLI仍以Controller environment changed拒绝。

按systematic-debugging从错误栈、保存JSON及源码定位到跨会话环境比较条件；未更改环境来绕过，未修改第928行或源码／协议，也未再运行诊断。该问题与原Torch／PatchCore临时写入是不同层面的障碍。

独立CPU元信息核验不调用旧verify或模型包，不以成功交集缩减文件：核17文件完整清单、16输出hash、实现身份、旧6文件、全部695受保护文件及9目录签名，保存比较／原序／事件／退出状态和运行内环境。它不能冒充原verify CLI通过。证据如下：

```json
{
  "status": "INDEPENDENT_METADATA_HASH_AND_RECORDED_EVIDENCE_VERIFIED",
  "complete_sha256": "942429386ad21600c67f0c970c94c1dbd48318023376d9b7c2e5a9df5474887c",
  "output_count": 16,
  "old_695_hashes_unchanged": 695,
  "old_directory_signatures_unchanged": 9,
  "old_diagnostic_six_unchanged": true,
  "runtime_consistency_checks": 16,
  "controller_PID": 29528,
  "controller_PPID": 26964,
  "Popen_PID": 21000,
  "child_PID": 31104,
  "child_PPID": 21000,
  "actual_relation": "LAUNCHER_ONE_HOP",
  "completed_imports": [
    "numpy"
  ],
  "event": {
    "kind": "OUTSIDE_WRITE",
    "during": "torch",
    "details": {
      "audit_event": "open",
      "flags": 130,
      "mode": null,
      "target": "\\\\.\\NUL"
    }
  },
  "within_run_temporary_environment_unchanged": true,
  "current_process_temporary_environment": {
    "TEMP": "C:\\Users\\Huang Yuhang\\AppData\\Local\\Packages\\sandbox.{66794ab7-112a-4727-8790-c92cd7c33111}\\AC\\Temp",
    "TMP": "C:\\Users\\Huang Yuhang\\AppData\\Local\\Packages\\sandbox.{66794ab7-112a-4727-8790-c92cd7c33111}\\AC\\Temp",
    "TMPDIR": null
  },
  "run_temporary_environment": {
    "TEMP": "C:\\Users\\Huang Yuhang\\AppData\\Local\\Packages\\sandbox.{25689c7d-8912-4c85-9b8f-441c3029eb34}\\AC\\Temp",
    "TMP": "C:\\Users\\Huang Yuhang\\AppData\\Local\\Packages\\sandbox.{25689c7d-8912-4c85-9b8f-441c3029eb34}\\AC\\Temp",
    "TMPDIR": null
  },
  "current_vs_run_environment_match": {
    "TEMP": false,
    "TMP": false,
    "TMPDIR": true
  },
  "new_model_import_or_forward": 0,
  "new_diagnostic_children": 0,
  "source_data_array_reads": 0
}
```

## 最终审阅与取舍

一次新上下文只读审阅与实际证据一致，提出两项Important：跨会话环境比较不适合作历史事务唯一核验条件；NUL事件不能被写成原Temp调用者。没有放宽守卫或修改冻结源码来解决它们。技术根因的进一步验证、边界适配需要独立方案，当前不执行。

取舍：保留运行内complete并明确独立CLI失败，避免删除或补写冻结标记；代价是目前没有可跨会话通过的既有verify入口。独立hash与字段核对只作为单独证据，不能替代尚未完成的正式核验适配。原代码权限禁界及两旧失败保留，无新的软件测试／过程表／提交工作流。实际运行没有新增代码，最终审阅提出的适配留待独立方案，不对封存实现作修正轮或动态复跑。

## 已用入口与唯一后续

本次先static-check，再run-once，完整时调用verify；这些CLI均已实现。单次预算现已消费，**不要再次运行run-once，也不要将已开始现场交给static-check冒充准备态**。原verify仍有上述环境问题，不通过改TEMP／TMP重试。未来只准只读处理已保存证据；完整状态不意味着可以自动第2天B。

唯一后续建议：先编写独立只读核验身份适配方案，区分历史事务内部环境一致性与当前核验进程环境；固定17文件／来源／握手／退出及状态规则，旧源码和complete不动。NUL后的系统查询仍在禁界内，本次不批准新诊断、不放行设备／系统命令或推进真实取图。当前可发送的一条指令见[补充方案第六节](D:/华兴源创/计划/Mura第2天A独立启动身份适配与导入诊断补充方案.md)。所有训练与统计拟合仍由用户手动启动。

