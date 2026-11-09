# 可控缺陷候选工具：第1阶段

当前只有CPU资源／元信息准备与完整缓存核验，尚无图片规范化、生成或界面入口。旧N_gen80按已有元信息全部RGBA，当前输入规则拒绝；background_admitted=false。不能把prepare命令成功当第2阶段准入。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m defect_candidate_tool.cli verify-preparation
```

首次准备命令为同解释器`-m defect_candidate_tool.cli prepare`；已有完整同身份缓存时仅核验复用，不重新查资源或改写结果。缺complete、未知文件或身份变化保留停止，不删除后重跑。

准备目录：`results/defect_candidate_tool/v1/preparation_v1`。environment为当次快照；verify不代表当前RAM仍相同。仅实现inspect_environment和validate_request等元信息功能，normalize_background待第2阶段，当前未实现也不提供调用命令。

无图像字节读取／解码、掩码生成、模型导入、训练或安装。Tk只核Tcl解释器与版本，无窗口，不能称GUI已可用。源PNG/JPEG的透明信息检查仍待实际输入阶段；当前只凭旧清单保守拒绝RGBA。
