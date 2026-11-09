# Mura第2天A独立只读结果核验适配结果

日期：2026年11月3日。用户手动运行阶段B后告知“跑完了”；本次先检查保存结果，再仅调用已实现的verify进行独立CPU只读核验，没有重复run、原诊断或模型评分。按方案完成阶段C最终说明与进度同步。

**结论：独立只读封存记录核验通过，阶段B及本次阶段C收束。原verify入口仍保持失败记录；这不是Torch、PatchCore或真实Mura检测效果通过，原Temp调用者仍未知。**

## 实际结果和产物

[新独立结果根](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/import_diagnosis_result_verification_v1)严格6文件：sources.json、protocol.json、preparation_complete.json、run_started.json、verification.json、complete.json。用户此次run保存status=INDEPENDENT_READ_ONLY_SEALED_RECORD_VERIFIED、passed=true，source_integrity=true、admitted_preparation_unchanged=true；新complete状态为INDEPENDENT_READ_ONLY_RECORD_VERIFICATION_COMPLETE。

新complete SHA256：e4ee6b7bfdd51edea858b91b43eb9da37204c084dbc4a135cf17a692792063a7。
原诊断complete SHA256：942429386ad21600c67f0c970c94c1dbd48318023376d9b7c2e5a9df5474887c。
新独立实现仍为f2c6dbf010a629dc865ba4810c63cecd4939175975a11b0c94c3e79f9386196e；原17文件、原启动与导入实现、旧6文件、原PatchCore失败、两正常工作点及原tau保持不变。

来源闭合使用732项固定文件hash（包含原695项）与9目录stat签名，五份模型资源只流式读取不可解释字节hash，不加载／解析tensor或pickle。新6文件完整清单、hash与原来源在只读核验前后均检查，未写回缓存或覆写原结果。

| 核验层 | 保存及复核结果 |
| --- | --- |
| 顶层必要检查 | 32项均true |
| 历史运行环境必要检查 | 嵌套13项均true |
| 历史进程／来源身份检查 | 嵌套20项均true，LAUNCHER_ONE_HOP与原结果一致 |
| 原16项一致性 | 独立重算一致，全部true |
| 当前工具进程完整缓存核验 | INDEPENDENT_COMPLETE_CACHE_VERIFIED |
| 原verify CLI | FAILED_ENVIRONMENT_COMPARISON_PRESERVED |
| 新诊断启动数／模型ready | 0／false |

这些层级互有包含，不累计为独立样本、独立实验或额外科学结果。原一次诊断仍仅一次，新核验没有增加导入次数。

## 环境适配实际说明

历史诊断started、child_boot、child_identity、parent_identity及first_event的TEMP／TMP／TMPDIR在运行内部一致。准备／更旧会话的环境有其各自历史含义，不混为本次诊断运行环境。

用户实际核验进程使用常规本机Temp目录；随后助手只读核验处于另一工具沙箱，二者TEMP／TMP不同，TMPDIR均为null。新verify仍返回完整缓存核验通过，且没有修改任何环境变量或保存结果；生成时的current观测保持原样，新的current_observation只在返回值中报告。

这说明独立核验不再错误地要求“当前会话临时路径等于历史会话”，同时仍保留历史事务内部环境要求。原import_diagnosis_launch.py第928行和原verify从未修改；不能称原入口已修复。当前成功直接支持的是这份独立结果核验适配，不是所有环境／全部失败分支已经实际覆盖。

## 原诊断的状态和未解决项

已保存的一层启动器关系为：控制PID29528，Popen PID21000，Python child PID31104／PPID21000。独立核验使用保存的内核映像／创建时间和退出证据，没有重新查询这些历史PID。退出后映像错误31仍按既定例外保留，不能填补进入前缺失身份。

原诊断仅NumPy导入完成，Torch导入停在NUL空设备open请求（flags130、mode=null）。调用链为torch._load_dll_libraries → platform.machine／_syscmd_ver → subprocess._get_devnull；请求被拒绝，不是已向Temp写文件。后续platform系统命令／子进程路径没有放行依据，本次没有放行NUL或重新执行依赖导入。

原token原值未保存；只复核封存摘要、HELLO／DONE会话、来源与保存比较的一致性。token_preimage_revalidated=false、historical_kernel_queries_reexecuted=false。不把本地hash当数字签名／外部真实性认证，也不倒填旧会话缺失的token／PID子条件。

原Temp临时写入的调用者尚未被这条导入诊断链触达；不能认定NUL就是原故障。PatchCore D_fit评分尚未恢复，真实Mura材料取得／评分未推进，D_hold保持隔离。没有新增检测指标、模型改进或屏幕收益。

## 实际资源与执行边界

用户run_started保存可用RAM 5.0454 GiB、D盘空闲 113.1572 GiB，满足≥2／40GiB。没有记录该次CPU核验总时长或峰值，不按文件时间差猜测；原1.109秒是导入诊断时长，不是本次CPU核验时长。

助手本次仅运行verify；其内部复核冻结来源、准备与完整缓存，实际只读成功。没有启动诊断子进程、安装／下载、导入NumPy／Torch／模型、运行前向、生成、训练、校准、统计拟合、读取RGB／GT／数组／概率或参数解析。没有修改源实现／配置／权重／tau／原complete、失败标记或新6文件。

本次未改代码，复用阶段A已做的一次独立源码审阅及修正；没有新增软件测试、过程表、提交流程或后台任务。所有训练与统计拟合仍由用户手动启动。

独立进程实际verify返回：

```json
{
  "status": "INDEPENDENT_COMPLETE_CACHE_VERIFIED",
  "original_verify_cli_status": "FAILED_ENVIRONMENT_COMPARISON_PRESERVED",
  "current_observation": {
    "TEMP": "C:\\Users\\Huang Yuhang\\AppData\\Local\\Packages\\sandbox.{9c6b4f2e-68e2-493c-9dd7-da130ce601ce}\\AC\\Temp",
    "TMP": "C:\\Users\\Huang Yuhang\\AppData\\Local\\Packages\\sandbox.{9c6b4f2e-68e2-493c-9dd7-da130ce601ce}\\AC\\Temp",
    "TMPDIR": null
  },
  "generation_current_observation": {
    "TEMP": "C:\\Users\\HUANGY~1\\AppData\\Local\\Temp",
    "TMP": "C:\\Users\\HUANGY~1\\AppData\\Local\\Temp",
    "TMPDIR": null
  },
  "current_environment_is_not_identity_gate": true,
  "new_diagnostic_launches": 0
}
```

## 已实现运行说明与后续范围

完整新缓存只读核验入口如下，不重复run或原run-once：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.import_diagnosis_result_verification verify
```

返回INDEPENDENT_COMPLETE_CACHE_VERIFIED表示本方案记录核验通过；若身份／资源／目录／hash不明则保留停止，不补写、不换版本或修改环境。原verify仍保持其原行为。

记录核验适配可在此收束。真正的工程障碍仍是依赖初始化与原构造守卫边界，不能靠继续核验相同记录解决；后续若继续，应先对照既有正常工作点成功链与D_fit／独立导入停止链，审阅初始化和守卫时机的差异，仅提出有界建议。不自动新诊断、模型构造、评分或第2天B。四份相关计划已同步唯一摘要、当前产物和一条下一步指令。

