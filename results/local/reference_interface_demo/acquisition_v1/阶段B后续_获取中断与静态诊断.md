# 阶段B后续：获取中断与静态诊断

2026年10月16日。状态：STOPPED_BY_USER_INCOMPLETE。用户要求停止下载，之后手动提供资源；对应pip进程已停止、父任务以失败结束，再次进程查询无匹配项。没有继续下载、重试、安装或进入阶段C。

## 实际取得情况

固定20文件中4个小文件大小及Git blob身份核验通过；第5个safety_checker权重仅收到282977983／608018440字节，保留.download.part，未作为完整权重或解析张量。其余15文件未开始。不存在完整权重集合或完整获取completion，acquisition.incomplete保持。

独立环境已建立。当前分发包元信息为：pip 24.2。冻结10核心依赖及完整传递依赖尚未安装／核验，不能称环境就绪。PyTorch轮子中断副本保存在package_partials，500695040字节；这只是下载前缀，不是有效完整wheel，不用于安装。

## 原因与证据边界

补绘权重流在冻结长度之前结束，download_one因长度不足停止。对同revision URL的一次HEAD返回200、Content-Length608018440、Accept-Ranges bytes；没有取响应体或续传。证据表明服务仍公布完整对象，但未定位哪个网络环节提前断流，不认定hash冲突或模型仓库失效。软件包下载则按用户明确要求终止，与前述权重流提前结束分开解释。

已有inspect-local-resources实际调用因目录中的未完成.download.part拒绝，属于当前未完成现场，未删除残片使其通过。6个完整权重header、全20文件身份、pip安装报告、完整传递依赖及解释器验收尚未完成。

## 保留与后续

阶段A／B冻结身份核验通过；旧环境解释器、pyvenv.cfg及METADATA／RECORD共102项与获取前一致，未全量重hash旧包二进制。没有导入Torch／模型包、读取背景／参考图、前向、合成或训练。原v3为3/24、ready=false不变。

本次只完成部分资源取得与中断诊断，未完成阶段B后续验收，不能进入模型运行。未来保守资源门槛仍为可用RAM6GiB／显存7168MiB／磁盘20GiB；历史RAM4.958GiB不足，未做运行峰值实测。手动资源送达后先核验来源、完整hash与事务归属，再决定独立恢复方案；当前acquire／verify-acquisition不得作为恢复命令重跑。

精确文件身份、残片hash、环境包和检查器拒绝信息见同目录interruption_diagnosis.json。手动补齐详见[手动资源补齐说明](./手动资源补齐说明.md)。
