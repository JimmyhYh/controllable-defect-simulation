# 第1天v3：编辑局部适用性CPU审阅

状态：LOCAL_REVIEW_UNCERTAIN，ready_for_day2=false。24位置中3个通过，其余受确定结构／不确定色点的保护区交集影响；没有正式48请求或24套几何数组。不能进入第2天，不能据3个通过位置缩减原主方案。

PowerShell只读核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -X utf8 -m reference_generation.local_cli_v3 verify-local
```

已有完整缓存时，以下已实现准备命令也只核验复用：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m reference_generation.local_cli_v3 prepare-local
```

初次准备会先冻结空间／显示／标注规则，再生成两张既定背景的审阅材料；review-local要求显式结构与不确定矩形JSON，先保存再算交集。当前一次审阅已经完成，不允许重画矩形或覆写；再次调用已完成入口只返回原缓存核验。

核验重建CPU几何和结构相交，检查封存数值统计／身份，不重新解码正常RGB重算像素统计；会读取既定参考掩码及必要旧元信息／文件hash，不读取其他正常源图像素。严格目录清单拒绝未知文件／目录、.tmp或不完整事务。

[冻结结果](D:/华兴源创/实验/弱缺陷Demo/results/reference_generation/v3_local_context/第1天v3局部背景适用性结果.md)与[主计划](D:/华兴源创/计划/真实参考驱动可控缺陷生成执行计划.md)记录完整24分母、全部矩形和下一步。S/U是开发审阅标记，不是缺陷语义真值；不确定色点不认定为真实制造缺陷。远处结构仍可进入完整模型画布，影响尚未知。v1/v2失败结论保持不变。

没有模型加载、异常合成或训练入口。所有未来训练仍由用户手动启动。本说明在结果根之外，不修改冻结缓存清单。
