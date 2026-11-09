# PatchCore解除启动限制后最小离线推理结果

日期：2026年11月4日。个人学习与非商业研究，私人设备、公开材料，公司材料隔离。

## 结论与实际完成范围

固定PatchCore最小离线推理已完成，并在另一个新进程中通过只读完整缓存核验。直接复用原作者显式WRN构造及旧inference_runtime，不修改vendor、旧守卫或环境，不实施C01延迟导入改写。

实际完成1次模型构造、1次V1 tensor加载、1次作者ones构造探测，以及2次指定输入评分。模型构造后的参数／完整state_dict／BN缓冲／eval和梯度状态与原正常工作点完全一致，两次评分后均保持。作者构造探测仍沿原行为执行；这里的“状态不变”比较从构造结束开始，不意味着探测前后的BN状态未发生任何变化。

本结果证明按新边界原路径可以完成这两份输入的推理和保存，不是整体检出能力、SEMU刻度适用性、真实Mura或屏幕收益的证据。原Temp失败的具体调用者仍未知，旧失败和NO_GO原条件结论不改为通过。D_fit全量和真实Mura评价仍未完成，没有自动推进。

## 输入规则、实际扰动与模型口径

选择完全依据冻结D_fit原顺序：roles中的首个背景fit_b000，以及该背景requests原序首个非零实际变化请求fit_b000_r04_circle_bright_a0。没有根据分数选图，没有读取原RGB、GT或其他角色输入数组。两个输入是同一背景的基线／变体，不能称两个独立背景或未见真实验证样本。

所用float32/HWC/256×256工作RGB直接来自fit_inputs_identity_v2保存的NPY，各读取一次，逐一核文件和规范工作RGB摘要。未PNG回读、再次除255、重新渲染或量化。

| 项目 | 冻结条件／实际值 |
| --- | --- |
| 骨干与库 | 原torchvision wide_resnet50_2；已有ImageNet V1；已有N80库819×1024 |
| 特征／近邻 | layer2／layer3，嵌入1024／1024，patchsize3，NN1 |
| 扰动 | r04_circle，亮，原幅度0.001；固定中心，不重新定位 |
| 实际变化 | 非零49像素，实际RGB残差最大0.0010000020265579224 |
| 诊断有效掩码 | 按1/255阈值计为0像素；不意味着无变化，不用它删除本请求 |
| 图像预处理 | 保存float32工作RGB；原ImageNet mean／std，不拉伸对比度 |
| 运行 | eval／FP32／batch1／workers0／threads4，seed42，原TF32／benchmark关闭 |
| 整图分数 | 原最大patch平方L2最近邻分数；不是热图最大值 |
| 热图 | 原作者插值及Gaussian sigma4，保存原始float32，不额外处理 |
| 阈值 | 原tau=1.852951288223265；严格大于；本次未校准 |

## 两份实际输出

| 输入 | 原始整图分数 | 是否严格超过原tau | 原生热图范围 |
| --- | ---: | --- | --- |
| fit_b000背景 | 1.1345081329345703 | 否 | [0.10924215614795685, 0.9606893658638] |
| fit_b000_r04_circle_bright_a0扰动 | 1.134429931640625 | 否 | [0.10924215614795685, 0.9606890678405762] |

分数差（扰动−背景）为−0.0000782012939453125。该局部微弱变化未使本例整图最大patch分数越过阈值；不能据一个位置／强度／背景判断整体敏感度或要求响应单调。整图最大值可能由其他区域主导，本次没有读取其他刺激、计算联合定位命中或归因峰位置。热图最大值也不能替代原整图分数。

本轮只核有限性、形状、来源、模型状态及保存完整性，没有拟合曲线、统计检验、训练或修改阈值，不报告AUROC、AP或整体召回。

## 资源、冷启动与边界证据

准备时可用RAM3.7167587280273438GiB，未达到4.5GiB模型准入；当时未创建运行事务或启动模型。用户释放内存后再次核查9.978862762451172GiB，正式事务开始时9.830307006835938GiB，D盘113.50581359863281GiB，随后才启动唯一一次run。未降低门槛或替用户关闭应用。

实际设备NVIDIA GeForce RTX 5060 Laptop GPU；既有.venv Python3.12.7、Torch2.8.0+cu128、torchvision0.23.0+cu128、faiss-cpu1.12.0，未安装／下载／升级。原runtime另核CUDA空闲显存至少4GiB。

构造耗时40.836646秒；自构造前事务计时到两次输出完成约45.459366秒。这个时间不含前置来源核验、末尾再次hash封存或独立核验，不能当全流程时间或纯GPU延迟。构造前重置后的GPU allocated峰值366.806152MiB，reserved峰值410MiB；结束采样可用RAM8.339805603027344GiB，不是进程内存峰值。

本次新审计实际记录：NUL打开1次，platform系统查询1次（C:\\WINDOWS\\system32\\cmd.exe /c "ver"），正常临时文件写入打开2次；未记录被拒绝事件。未改变TEMP／TMP，父级启动环境保持继承。此次允许必要启动行为后路径运行成功，但未保存这些临时写入的完整调用栈，不据此认定原Temp失败调用者。

审计仍拒绝网络／不受限pickle、非列定原图及数组、非列定模型资产和结果／正常临时目录之外的写入。params.pkl仅作不透明字节hash，不反序列化；实际模型加载沿原weights_only=True。Python审计不是完整操作系统或原生扩展沙箱，因此原生库全部I/O隔离没有被本次核验独立证明。

## 实现、结果与独立核验

唯一新增实现：[patchcore_minimal.py](D:/华兴源创/实验/弱缺陷Demo/mura_capability/patchcore_minimal.py)。根目录：[patchcore_minimal_runtime_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_capability/week1_v1/patchcore_minimal_runtime_v1)。

preparation封存sources／protocol；run封存started／state／result及两张float32热图。unknown／部分／failure／inflight事务不会补跑；完整同身份缓存只读核验复用。没有向任何旧冻结目录追加文件。

- 实现SHA256：479fd19ac5f2694d5ab81dd9b0722a095241288c985d2c0834fc74f7acf62924。
- preparation complete SHA256：d7813fabeb7b706193e24ce4df965a2d59f1bac01e5f6c8ab7d42ee6938a7ced。
- run complete SHA256：51472f150d69d2d206fd26990a56ea95d35d6a9440ce1e7212c0e4bdffd038b6。
- 状态参数SHA256：6372d614e547b82ad3683dab5f7ab453897c13d765c73871fb26305f1a4e4587。
- 完整state_dict SHA256：1a33eb6df3f1ea53b04d7913bf4039dbc8f13db5f5ed7c9263a39afc7f9c25f9。

运行后再次核验81份绑定旧来源文件与5个历史目录文件清单未变；新增实现的身份另计。权重、库、vendor、原工作点／tau及旧失败包含在相应绑定范围，不是扫描全部项目文件后的“所有文件未变”声明。

CPU人工字典核对了固定原序选择、零变化跳过规则和顺序错位拒绝；没有建立软件测试套件或过程表。依executing-plans进行一次独立只读代码审阅，无Critical／Important／Minor修改项；没有因审阅修改已封存源码。审阅不替代运行证据，原生I/O边界和检测效果仍不作外推。

另一个进程的已实现verify返回：

```json
{
  "status": "MINIMAL_COMPLETE_CACHE_VERIFIED",
  "complete_sha256": "51472f150d69d2d206fd26990a56ea95d35d6a9440ce1e7212c0e4bdffd038b6",
  "new_model_forwards": 0,
  "new_source_RGB_reads": 0,
  "tau_recomputed": false
}
```

该核验只从自有热图缓存核有限性／dtype／尺寸／hash及保存分数与tau；没有导入Torch、构造模型、新前向、打开来源RGB或重取分位数。

## 已实现运行说明

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_capability.patchcore_minimal verify
```

准备及其核验入口分别是同一模块的prepare、verify-preparation；实际运行入口是run。三者均已实现且本轮已调用；当前完整结果存在时run只核验复用，不再次推理。必须带-B以避免普通字节码缓存写入。不要删除、改名目录或标记以触发重跑。

本轮授权只到这两个输入。后续若恢复D_fit全量评分，需在独立新入口／目录绑定本次完整结果与相同RGB／模型状态，明确复用及新增前向分母，不直接重跑旧fit_scores_v1入口；本报告不自动授权全量评分或真实材料取得。

