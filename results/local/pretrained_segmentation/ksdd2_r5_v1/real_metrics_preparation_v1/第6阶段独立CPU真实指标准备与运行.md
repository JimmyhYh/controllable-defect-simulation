# 第6阶段独立CPU真实指标：准备与运行

本次只做CPU元信息／来源hash／整数分母与虚构标量核验，未读取任何GT文件字节、Q／E／U原图、解析概率载荷、计算真实AP／AUROC／计数或结论，未导入Torch／SMP／timm／模型、加载参数、前向、训练、生成或重校准。完整E评分以保存NPY文件hash和既有全量核验证据绑定；哈希读取文件字节不等于解析数组，未调用旧verify-results解析概率。

一次独立静态审阅发现，原新增复用入口只逐项核已登记hash，不能拒绝上游后来新增的未知文件或空目录；已按systematic-debugging以虚构清单确认原因，补齐E准备／E评分／E全量证据／Q校准证据的完整文件和目录集合及hash复核。虚构failure文件／未知空锁目录均被拒绝，实际上游目录未写入任何模拟异常；未改封存评分实现或数据。新增3个Python模块AST、显式CLI、CPU来源／标量与资源核验通过，未新增软件测试或模型运行。审阅后的单次修正由执行方核验，未重复派发审阅。

## 冻结指标口径

单个最终10000步模型，原E原序1004张／146,411,528原生像素、4270窗评分缓存和原τ=7.264203168233501e-06。规范GT路径／来源hash／shape来自冻结metric_sources；GT类别与阳性分母当前未知，历史110／894或486461仅为历史，不构成本次实际GT核验。未来显式compute先完整核验评分概率，再对1004规范GT各解码一次，校验字节hash、mode1/L、尺寸与逐图二值{0,1}或{0,255}，非零映射1，不resize／膨胀／软化；另存uint8原生GT缓存及实际分母，RGB始终不打开。

每个原生像素一次，拼回图已剔除右补边，禁止重复计算重叠窗。全局AP只读复用fewshot_synthetic.metrics.exact_ap，不抽样／平均逐图AP；其两个表共8,522,825,736字节（约7.94GiB），为全部非负float32值精确分组，并非近似分箱。输入逐图有界分块；相同值共同处理、-0规范为+0。复用旧21窗实际sklearn差3.33e-16与643077不同分数／265563重复分数值的封存证据，不重算旧AP；它是算法依据，不是E效果。未来E原生N／实际P、精确扫描计数和独立磁盘表计数／ties不变量均核对，不在E全量拼接作高内存sklearn替代。

整图AUROC用1004个保存原生max和GT是否非空计算；固定阈值strict>τ分别给TP／FN／FP／TN、异常召回／正常误报及真实分母。无阳性像素的AP、单类别AUROC、零分母率写NA/null并注明原因，不减分母。按E原ID升序取首个TP／FN／FP／TN，空组NA，去重且保留理由；不按最好定位选图、不打开RGB。单seed无三对均值／差值／显著性判断。E历史已见，只作探索性评价；不能据与旧架构差异证明预训练因果收益或屏幕效果。

## 已实现入口、缓存与停止规则

CPU准备核验（不解码GT／概率或计算指标）：
```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.real_metrics_cli verify
```

未来实际CPU指标入口已实现，本次未执行；需用户手动运行或另行明确授权：
```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.real_metrics_cli compute
```

准备独立real_metrics_preparation_v1；结果独立real_metrics_v1，内含labels／exact及最终metrics。完整同身份结果仅verify-results只读核验保存值和规范缓存，不解码原GT或重算AP／AUROC；任何未知文件／目录／锁、hash／依赖冲突、部分事务、failure或只有exact完整而总事务未完成均停止保留，不自动续接、重新累计、取交集、近似替代或删现场。GT读入与概率读取使用同一字节快照hash后解析，防止两次打开间身份替换。

准备／实际计算准入RAM≥3GiB／磁盘≥80GiB，CPU只读完整复核RAM≥2GiB／磁盘≥1GiB。逐GT和AP输入块同原3／80门槛；独立表计数核验2／1。冻结exact_ap内部排序表扫描只在入口检查资源，不是连续监控；运行中OOM／磁盘错误保留，不改算法。准备快照RAM5.438GiB／磁盘128.007GiB，仅快照，未来重新准入。算法有8.52GB表写入和多轮hash／扫描，实际本轮耗时与峰值待测，不把旧22.110秒小试或E评分172.437秒当指标耗时承诺。全部训练仍由用户手动启动；本次不进入实际计算、图件或新研究。
