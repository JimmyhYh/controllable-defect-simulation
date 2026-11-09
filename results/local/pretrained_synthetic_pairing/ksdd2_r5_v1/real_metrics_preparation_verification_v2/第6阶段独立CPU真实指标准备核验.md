# 第6阶段独立CPU真实指标准备核验（v2）

当前完成的是独立CPU指标准备，实际六模型指标为0/6，未启动。六组E1004完整评分、最终10000步权重、各自原Q200阈值、固定E原序、规范GT路径及既有hash、每模型146,411,528原生像素与精确AP源码已绑定。1004唯一图没有按六模型重复计为独立数据。

v1的独立审阅指出实际CPU依赖身份缺失、未来概率重复全量I/O两项Important。v1代码与准备封存保持不变；v2另建real_metrics_v2.py与real_metrics_preparation_v2，绑定解释器hash和Python、NumPy、scikit-learn、SciPy版本；实际概率路径改为每份文件一次read_bytes，随后在内存中hash、解码、校验并供精确AP使用。没有调用旧real_cache的预遍历。完整结果复用不重算AP或AUROC，计数、率、FP/FN ID及selected_examples从保存分数和规范标签核一致性。AUROC只核封存hash与范围，未声称独立重算验证。

本次新进程prepare与verify均实际退出0。永久审计钩子禁止准备进程打开任何标签/概率.npy、图像、模型包或实际指标包；GT数组与概率数组读取均0，Torch导入0、模型操作0，实际AP/AUROC/阈值计数/配对结论均未计算。旧110异常/894正常与486461阳性像素未在本阶段重新核实；未来实际入口必须全量解码规范标签后核二值、尺寸、来源与实际分母，不读取原始GT。

v2修正由作者完成静态、源码身份、AST单读取点及受保护CPU元信息/虚构标量核验；第二次独立复核因审阅者使用额度不足未完成，不称二次独立审阅通过。实际数组、第三方指标计算兼容与六组CPU耗时留待明确运行，准备完成不能称实际运行通过。

运行门槛为可用RAM至少3GiB，D/E盘各至少80GiB。六模型精确表共51,136,954,416字节（约47.63GiB），全部保留；冻结exact_ap扫描内部没有连续RAM监控。部分/未知事务保存停止，不重累计、不取成功交集、不换指标或阈值。E历史已见，只作固定单类别、固定R5/划分与三初始化下的探索性组成证据。

只读准备核验：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v2 verify
```

已实现但本阶段未运行的实际CPU指标入口：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_metrics_v2 run
```
