# Mura第2天A：启动身份适配阶段1—2准备与静态核验

日期：2026年11月3日。按superpowers:executing-plans执行本轮明确授权的阶段1—2；仅CPU来源／hash／源码／人工字典，不执行阶段3。

已完成独立实现、三文件准备封存和另起元信息进程的只读静态核验。30个人工情形通过，实际诊断／握手子进程0、依赖导入0、模型／数据载荷读取0。原故障尚未修复，Windows实际进程关系、依赖导入及写入调用者仍未知。

## 最终产物与身份

- [独立模块](D:/华兴源创/实验/弱缺陷Demo/mura_capability/import_diagnosis_launch.py)，SHA256：af4566d5100c698b59ccaff885b8454b76b80f7b35ac17a5ebff37bbace4a905。
- [独立准备目录](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_launch_v1)：仅sources.json、protocol.json、preparation_complete.json。准备清单SHA256：fdde56acd66795fab97649871495611e52fb12de498163c4608b7f13a18228b1。
- 原import_diagnosis.py／__init__.py、旧import_diagnosis_v1六文件、原PatchCore failure／inflight、day1与两个正常工作点均保持。独立核对本轮前后32项文件hash；来源核验沿day1固定695文件hash和9目录stat签名全量核对，不解析数组载荷。
- PatchCore原tau=1.852951288223265，G1原tau=0.6587551027536391。不重取分位数、不改阈值。
- 新根没有started.json、first_event.json或最终complete.json；preparation_complete只证明准备，不证明动态诊断完成。旧诊断仍仅6文件／已提交1次／无最终complete，不重置其预算。

## 身份与权限实现

保持同一.venv、继承环境、不修改TEMP／TMP。分别绑定项目解释器、pyvenv.cfg、基础解释器、本地启动器、源码、固定来源与旧现场。基础解释器当前hash为ca60cf785b2314a6d6599eced15bdf094e6db171bec996b97a70b995942c3c37；这是本轮新观测，不冒充旧工作点已绑定基础二进制。

未来动态入口只准一份预先独占写入的启动记录和一次Popen；原token经stdin传递，落盘仅摘要。child对照启动记录独立计算，双方记录分项先保存后判定；Popen实际句柄PID／映像／创建时间与child内核身份共同约束。只允许共同条件全真且DIRECT或LAUNCHER_ONE_HOP完整一项成立。未知字段拒绝，未采用分支的false不误拒有效另一分支。

有界管道每消息64KiB、每流4MiB；父进程统一总时限120秒，HELLO15秒、进入许可20秒。先保存双方观测、decision和进入许可才准导入；核对同机Windows单调时钟截止值。禁止模型构造、权重tensor／pickle、CUDA、图像／数组／概率、前向、生成、训练、校准与拟合，首次事务外写入保存栈、mode／flags、当前导入与依赖身份后拒绝。Python守卫不等同于全部原生库副作用已经认证。

prepare只写新三文件；static-check只读。run-once与verify均已实现，但本轮未调用。已有完整事务只读核验，任何部分／未知事务拒绝再启动；未解释的进程关系不放宽。

## 实际静态核验与审阅

首轮21情形通过；一次独立源码审阅提出4项Important，未列Minor。封存前同一修正轮补齐：导入前的实时截止判定、启动字段逐项异常收集／null原因及非HELLO先记录、完整记录内容一致性、独立于启动器存活的已绑定Python进程收尾。修正前源码相应四标记均缺失；修正后编译／AST与30人工情形通过。该证据限静态比较、流程与源码审阅，不宣称实际超时／进程清理已经运行验证；未派第二次审阅或做动态试跑。

30情形含：两个有效有限关系，错误token／解释器／来源／环境，未知句柄／字段／祖先，日志未保存、缺失／重复／过期HELLO、ABORT／缺GO，已有启动拒绝，事务外flags写入、图像／数组／权重读取、CUDA／构造禁界，以及人工完整记录与晚进入／事件缺失／许可矛盾／总时限超限的拒绝。AST仅一处Popen调用点，静态执行实际调用0。

取舍：以“共同条件 AND (DIRECT OR LAUNCHER_ONE_HOP)”解释有限二选一，避免把未选分支的false误当共同失败；代价是如分支定义有误，仍须动态证据发现并停止，不能仅凭静态通过放行未知链。绑定方案第二至第四节固定技术正文，顶部摘要／第五节进度／第六节下一步不作为不可变身份；代价是进度文字不受该技术正文hash约束，准备三文件与源码仍严格hash绑定。用户明确不建过程表、软件测试／提交流程，因此用最终报告中的实际证据和人工字典核验，不引入相应工作流。无延后Minor。

另起进程实际静态核验证据：

```json
{
  "status": "STATIC_PREPARATION_VERIFIED",
  "complete_sha256": "fdde56acd66795fab97649871495611e52fb12de498163c4608b7f13a18228b1",
  "implementation": {
    "D:\\华兴源创\\实验\\弱缺陷Demo\\mura_capability\\__init__.py": "1adba9096b6adb0df8c1064ab74baf70825478009cefe3183c5d50ea3752a2fa",
    "D:\\华兴源创\\实验\\弱缺陷Demo\\mura_capability\\import_diagnosis_launch.py": "af4566d5100c698b59ccaff885b8454b76b80f7b35ac17a5ebff37bbace4a905"
  },
  "static": {
    "artificial_case_count": 30,
    "cases": {
      "DIRECT": "DIRECT",
      "LAUNCHER_ONE_HOP": "LAUNCHER_ONE_HOP",
      "wrong_token": "REJECT",
      "unknown_ancestor": "REJECT",
      "unknown_handle_identity": "REJECT",
      "wrong_interpreter": "REJECT",
      "permission_requires_saved_log": false,
      "permission_rejects_duplicate_HELLO": false,
      "permission_GO_valid": true,
      "malformed_HELLO": false,
      "wrong_session": false,
      "late_HELLO": false,
      "ABORT_is_not_GO": false,
      "missing_GO": false,
      "external_flags_write": "OUTSIDE_WRITE",
      "image_read": "FORBIDDEN_PAYLOAD_OPEN",
      "array_read": "FORBIDDEN_PAYLOAD_OPEN",
      "weight_read": "FORBIDDEN_PAYLOAD_OPEN",
      "CUDA_call": "FORBIDDEN_CUDA_CALL",
      "model_call": "FORBIDDEN_MODULE_CONSTRUCTION_OR_FORWARD",
      "legacy_PID_predicate_would_reject_launcher": true
    },
    "additional_cases": {
      "source_file_match": true,
      "temporary_environment_match": true,
      "unknown_field_refused": true,
      "closed_virtual_transcript": true,
      "reject_import_entry_monotonic_ns": true,
      "reject_first_event": true,
      "reject_permission_contradiction": true,
      "reject_excess_total_budget": true,
      "existing_launch_refused": true
    },
    "syntax_compiled": true,
    "AST_Popen_call_sites": [
      783
    ],
    "actual_process_API_calls": 0,
    "new_diagnostic_children": 0,
    "heavy_imports": 0,
    "real_file_operations_in_artificial_cases": 0,
    "dynamic_handshake_and_import_compatibility_verified": false
  },
  "resources": {
    "available_RAM_GiB": 7.833309173583984,
    "free_D_GiB": 113.1987533569336
  },
  "new_diagnostic_children": 0,
  "dependency_imports": 0,
  "actual_process_identity_observations": 0
}
```

保护与目录检查：

```json
{
  "status": "PRESERVATION_AND_PREPARATION_INVENTORY_VERIFIED",
  "old_files_unchanged": 32,
  "new_cache_files": 3,
  "started_exists": false,
  "final_complete_exists": false
}
```

prepare实测可用RAM7.9077GiB／D空闲113.1988GiB；另进程核验为RAM7.8333GiB／D空闲113.1988GiB。门槛RAM≥2GiB、D≥40GiB，动态前须重新检查。没有GPU探测、下载、安装或环境升级；prepare与static-check每次约5秒只作本机CPU参考，未来动态最多120秒是预算而非成功耗时保证。

## 已实现入口及下一步条件

只读准备核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.import_diagnosis_launch static-check
```

下列入口留待明确授权阶段3后运行，本轮没有执行。须准备身份不变、资源过门槛、无started／部分事务；如已有最终完整缓存则仅核验复用。含一次子进程和固定依赖导入，不构造模型：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.import_diagnosis_launch run-once --allow-diagnostic-imports
```

仅完整动态结果才使用已实现的只读verify；部分现场不能通过verify补完整。当前不提供／启动模型或第2天B命令。原始调用者、原token／PID子条件和进程链仍未知，新运行即使捕获写入也只直接说明当前环境，不能倒填旧根因。

当前唯一下一步指令见[补充方案第六节](D:/华兴源创/计划/Mura第2天A独立启动身份适配与导入诊断补充方案.md)。所有训练和统计拟合仍由用户手动启动。

