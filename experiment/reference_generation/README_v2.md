# 第1天补充v2：同域背景CPU准备

实际结果：固定64组中28组数值入池，PCG64(20261018)抽中1949／1540；视觉审阅分别发现字符／刻度状结构和规则点阵，判为failed。机器状态为VISUAL_DOMAIN_UNCERTAIN，ready_for_day2=false。没有顺延替补，没有创建v2正式48请求、24几何数组或位置总览；已有visual目录仅为正常图及参考的审阅裁剪。

PowerShell只读核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -X utf8 -m reference_generation.domain_cli_v2 verify-domain
```

实际可用准备入口：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m reference_generation.domain_cli_v2 prepare-domain
```

当前完整缓存存在时仅核验复用，不再读取正常源图像素、筛池、抽样或改写成果。verify-domain核验会读取固定64源文件字节hash及元信息，当前阻塞模式不解码其RGB。未知／缺失文件、额外目录、.tmp或未完成事务停止，不自动覆盖。没有模型、合成或训练命令。

CPU过程分为prepare-domain与显式finalize-domain。后者记录了助手对两张既定正常图及所有预定局部的开发视觉审阅，不是盲评；当前已完成的决策不可改写，重调用只核验原缓存。

[冻结结果](D:/华兴源创/实验/弱缺陷Demo/results/reference_generation/v2_domain_matched/第1天v2同域背景准备结果.md)与[主计划](D:/华兴源创/计划/真实参考驱动可控缺陷生成执行计划.md)记录完整分母、来源限制及下一步。所有未来参数更新仍由用户手动启动。本说明在结果根之外，未向严格缓存目录追加文件。
