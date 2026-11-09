# Mura响应研究第4阶段CPU正常输入取得与核验

日期：2026年10月29日。范围：固定Q200／F100的CPU实际取得与独立完整缓存核验。

**已按固定顺序完成Q200＋F100共300张正常图的取得，每原图只读取字节／解码1次；300份规范工作RGB唯一，无角色内、Q/F互相或与D_fit60／旧训练160的精确工作碰撞。normal_inputs_v1已完整封存，另起进程全量核验通过，新增原图打开／解码0。尚未加载模型、前向、生成扰动、训练、实际校准、计算新tau或拟合。**

依据：[冻结研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[总执行计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)、[第4阶段封存准备说明](D:/华兴源创/报告/Mura响应研究第4阶段正常工作点准备.md)。本轮沿用私人设备、公开材料、个人学习与非商业研究，公司材料隔离。

## 一、封存身份与实施范围

先在独立进程运行已实现verify-preparation，返回NORMAL_PREPARATION_CACHE_VERIFIED、退出码0。冻结双模块、固定角色、原两模型最终资源字节hash和环境身份闭合；没有修改或重新封存准备配置。

- 正常准备protocol：`ea64281c4ca714f30d557dc7215a57805329b61f6d8f9ba5e95cbe0903bfe4d7`。
- 正常准备complete：`b27ab761fc0af094d98d221546e64ca053efb7216e5ac6862b306c947e46dc77`。
- 本次正常输入complete：`bcccbbf1c2fa1001d3459bf9944bdd5a6ad934f57104ac05babe330b591de9da`。

本次只调用既有acquire-normal-inputs实际取得，再用新进程verify-normal-inputs完整核验。没有开发、改代码、改参数或自动推进GPU。D_fit及旧N只消费已有source_audit工作hash元信息与独立证据，不打开其原图或数组；原完整输入14883文件、失败13文件和正常准备4文件保持原状。既有38项源码／JSON／设计／报告hash及这些目录stat在交付前核对不变；旧数组仅stat核对，不叫重新证明其内容。

## 二、实际来源与工作域核验

顺序严格为Q200原元信息200行，随后F100原元信息100行，未换样、换seed、补数或按外观筛选。实际取得核原PNG全部字节hash、长度、尺寸1024×1024和原模式，保留原始对比度。

四类身份分别保存：原文件SHA256；只作角色／排除的旧RGB组ID；本次convert('RGB')后1024×1024×3 uint8 HWC C字节SHA256（无头）；规范工作RGB摘要。两种新摘要既不强求等于旧组ID，也不要求不同，没有猜历史序列化。

规范处理仍为Pillow RGB uint8先bilinear256，再float32/float32(255)，不量化后再读回。工作摘要沿用冻结inputs的dtype／shape／C头及数组字节定义。缓存核验逐张核float32、256×256×3、有限性、[0,1]及完整内容摘要；原始RGB摘要只在取得时计算，核验不重读原图。

| 项目 | 本次实际结果 |
| --- | ---: |
| Q角色来源 | 200/200 |
| F角色来源 | 100/100 |
| 原图字节打开／解码 | 300次／300条路径，各1次 |
| 规范工作输入 | 300份，各256×256×3 float32 |
| 唯一工作RGB | 300 |
| Q/F角色内部／互相重复 | 0 |
| 与已保存D_fit60工作内容重复 | 0 |
| 与已保存旧训练160工作内容重复 | 0 |
| D_hold工作域碰撞检查 | 未进行，原图未解封 |
| Torch／模型包导入、参数加载、前向 | 全0 |
| 扰动生成、训练、实际校准、新tau、统计拟合 | 全0／未取得 |

全300完成后才核碰撞并封存，未取有利子集。300条唯一工作内容不等于300个独立物理面板，采集分组仍未知。第3阶段的旧失败1次及新取得220次解码计数保持其历史含义；本次只新增固定Q/F的300次，未把缓存数组读取当作原图解码或模型前向。

## 三、最终输入及独立核验证据

[normal_inputs_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1)严格305文件：300份RGB NPY、started／sources／records／verification四份JSON、最后1份complete。实际总占用约0.220029GiB，无failure或不明确事务；所有既定来源都有记录。

- [全部来源／顺序／摘要及碰撞结果](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/records.json)。
- [实际取得期计数、逐路径访问和资源](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/verification.json)。
- [准备与输入身份绑定](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/sources.json)。
- [完整文件hash清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_inputs_v1/complete.json)。

另起进程verify-normal-inputs核305文件严格目录、全部封存hash、300份缓存的shape／dtype／数值域／内容摘要、来源顺序与角色、碰撞和保存的逐路径计数，返回NORMAL_INPUTS_CACHE_VERIFIED、退出码0。原图打开／解码均新增0，没有对未明事务补取得。

取得命令最终boundary的array_opens=300，来自封存时对300份新NPY作字节hash；保存于verification.json的边界在这一封存步骤之前记录。独立核验的array_opens=600，包括300次文件hash读取及300次NPY内容加载。它们均为正常RGB缓存访问，不是额外原图解码、概率读取或模型评分，不能要求该计数也为0。

以下为独立核验实际返回的完整元信息：

```json
{
  "status": "NORMAL_INPUTS_CACHE_VERIFIED",
  "normal_complete_sha256": "bcccbbf1c2fa1001d3459bf9944bdd5a6ad934f57104ac05babe330b591de9da",
  "identity": {
    "protocol_sha256": "ea64281c4ca714f30d557dc7215a57805329b61f6d8f9ba5e95cbe0903bfe4d7",
    "preparation_complete_sha256": "b27ab761fc0af094d98d221546e64ca053efb7216e5ac6862b306c947e46dc77"
  },
  "collision_check": {
    "Q200": 200,
    "F100": 100,
    "unique_working_RGB": 300,
    "collisions": [],
    "D_hold_working_collisions_not_checked": true
  },
  "boundary": {
    "source_decodes": 0,
    "array_opens": 600,
    "model_loads": 0,
    "tensor_loads": 0,
    "model_input_forwards": 0,
    "author_placeholder_forwards": 0,
    "parameter_updates": 0,
    "render_calls": 0,
    "real_calibrations": 0,
    "statistical_fits": 0,
    "original_file_opens": 0,
    "original_unique_paths": 0,
    "blocked_attempts": [],
    "model_packages_loaded": [],
    "Q_F_acquisition_only_if_explicit": true,
    "D_hold_S_bg_Q_cal_old_test_GT_reads": 0,
    "unrestricted_pickle_deserializations": 0,
    "forward_counter_semantics": "attempts before call; only committed rows are completed outputs"
  }
}
```

## 四、实际资源与尚未实测项

取得期保存到验收字段的elapsed_seconds=17.891841秒；该计时包括原图核对、解码／缩放、保存及源记录检查，不含其后再次准备身份核验和全文件hash封存，也不含另起进程的完整核验。未把它称GPU速度，未测本轮CPU进程峰值。

300次输入边界采样的最小可用RAM约5.665802GiB，满足2GiBCPU门槛。输出后资源快照：RAM7.352886GiB、D盘空闲113.458626GiB；NVIDIA GeForce RTX 5060 Laptop GPU总8151MiB／空闲7899MiB（约7.714GiB），驱动621.01。当时满足首模型启动4.5GiB RAM／4GiB空闲显存／40GiB磁盘的条件，启动入口仍会再次检查；这些是元信息快照，没有加载模型，不是GPU实测峰值或兼容保证。

运行环境沿用已封存Python3.12.7及既有NumPy／Pillow／SciPy，没有安装／下载。实际Torch／CUDA／FAISS加载兼容、作者初始化探测、BN／参数状态与Q评分耗时仍待首模型运行确认；不套用旧训练或第3阶段CPU耗时。

## 五、首模型已实现人工命令

本次只交付首模型PatchCore，未实际启动。以下命令会使用已核验正常缓存，保持冻结PatchCore正常库／ImageNet V1与预处理，先Q200评分、以q0.95／linear／严格大于冻结新tau，再F100检查。它不训练、不建库，不使用旧tau或D_fit刺激，不自动运行G1。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.inference score-normal --model PatchCore
```

输出拟在`D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_scoring_v1/PatchCore`，当前不存在。每个Q/F唯一正常输入只评分一次；原作者构造阶段单张ones特征维度探测另计，不伪装成第301张正常图。保留作者最大patch分数与原热图，状态快照在构造／加载和eval之后，逐输入核参数／BN不变。Q同池FP/200不称独立验证；F100报告FP／TN及Wilson区间，不回调tau。

任何来源／资源／依赖错误、OOM、状态变化、部分目录或残留inflight均停止并保留，不重复命令试图补推或删标记；需先查因。完整同身份评分缓存未来只核验复用，不再前向或重取tau。当前正常输入已就绪仅表示数据侧门槛通过，不意味着模型效果、SEMU预测或真实屏幕收益已经得到验证。

只读核验正常输入可使用已实际实现并运行的`-m mura_response.inference verify-normal-inputs`；它不重新打开原图。唯一可直接发送的下一步指令维护在总计划第六节，GPU评分须另行明确启动；训练及统计拟合继续由用户手动操作。

交付前完成一次独立只读结果审阅，未发现Critical／Important／需延期的Minor项。审阅核对协议、来源顺序、逐路径计数、碰撞、JSON封存身份、计时范围和人工命令；没有另开原图／数组／模型载荷，数组内容采用本次已完成的独立CLI核验证据。旧成果的前后不变由执行者记录的38项hash与目录stat负责，范围如第一节；实际GPU兼容／峰值／速度、物理面板独立性和检测收益继续保持未验证，不因审阅可交付而改成已证实。

