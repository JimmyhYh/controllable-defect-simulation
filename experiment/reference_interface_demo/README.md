# 独立最小参考条件接口Demo：阶段A

仅准备1949原move位置、参考019／024／020、种子20261016／20261017的6请求及3套参考几何。沿用v3原坐标、2倍保护区与原4倍模型上下文；不是神经生成或背景复合成果。

在 **PowerShell** 中核验已保存的输入：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.cli verify-inputs
```

CPU准备入口已实现：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.cli prepare-inputs
```

首次成功准备后，同一命令仅核验完整同身份缓存，不重新解码原图／重存结果。只读核验会从已保存掩码重建坐标数组来比较，不重新审阅背景或重算背景适用性。未知文件、目录、源码或来源身份变化、未完成事务均停止并保留查因；不要删除目录后强行重跑。

输出固定在 `D:/华兴源创/实验/弱缺陷Demo/results/reference_interface_demo/v1/`：requests.json为6请求，geometry为3套原图／local512／guard布尔数组；references保存原掩码、原生参考裁剪与RGB／mask512；background_original.npy为原始uint8 RGB；background512.png为共同模型图块。scope_protocol与source_guard分别记录范围／坐标和来源身份；CPU_position_overview.png是显示叠加，总览叠加不作模型输入；最终说明为“阶段A_CPU输入准备结果.md”。

原图与v2封存1949 RGB副本核验相等；只读取这一个背景和既定3参考。原v3仍3/24通过、ready_for_day2=false；本分支是筛选条件下的开发输入，不能代替原48请求验收或证明泛化。计划参考关闭／开启共12份神经输出，目前0份；参考掩码不同，跨参考差异含几何混杂。

没有下载、Torch、模型前向、异常合成、训练或旧检测评分；所有未来训练由用户手动启动。CPU输入准备不能证明参考接口响应或背景复合已成立。后续阶段须另行授权，另建独立资源／实现身份与结果位置，不往本阶段严格冻结结果目录追加文件，不修改本阶段已绑定的三份源码。当前没有模型加载、生成或资源下载命令。

独立只读审阅未发现Critical／Important。补充读取防护只覆盖当前代码使用的open、Torch导入和socket事件，不是通用I/O沙箱；本次未发现越界删除／改名或旧检测分数读取调用。它不能替代未来新模块的范围审阅。
