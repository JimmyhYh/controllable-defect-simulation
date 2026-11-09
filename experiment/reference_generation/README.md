# 真实参考驱动生成：当前仅第1天CPU准备

当前状态：`CPU_PREPARED_DOMAIN_BLOCKED`。5候选、3实际参考、2固定背景及48个CPU几何请求已保存并可核验；背景域条件未通过，不能进入第2天或神经生成。本包目前没有生成、模型加载或训练入口。

PowerShell只读核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -X utf8 -m reference_generation.cli verify-inputs
```

预期输出包含 `requests_verified: 48`、`geometry_arrays_verified: 24`、`ready_for_day2: false`，以及0次模型前向／参数更新。核验会重新解码所选参考与两张固定背景、重新核对CPU几何；不打开S_bg／Q_cal图像，不调用模型。

已实现的准备入口：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m reference_generation.cli prepare-inputs
```

当前完整同身份产物存在时，该入口只核验复用，不重新选图或改写输出。没有completion时遇到未知产物，或存在未完成准备标记时，停止并保留；不能删除标记后盲目重跑。完整缓存的核验会检查清单内全部文件hash，但目前不拒绝额外的未列入清单文件；这是待v2加强的目录清点限制，不能据此开启下游运行。

产物在 `D:/华兴源创/实验/弱缺陷Demo/results/reference_generation/v1/`。结果与障碍详见[第1天结果](D:/华兴源创/实验/弱缺陷Demo/results/reference_generation/v1/第1天CPU输入准备与域条件核验.md)。下一步先审阅同域背景抽样规则，不能把这次亮底正常图悄悄替换为暗底。
