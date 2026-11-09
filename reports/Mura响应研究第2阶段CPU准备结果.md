# Mura响应研究第2阶段：CPU协议与输入准备结果

日期：2026年10月28日。依据用户明确确认的[研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)及[执行计划第2阶段](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)。私人设备、公开材料、个人学习与非商业研究；公司材料隔离。

**结论：第2阶段已实际完成并独立核验。420个背景角色、15120条名义请求字典、9个几何与测量公式及来源身份已封存；人工数组数值核验通过。真实背景尚未解码，正式刺激尚未生成，没有模型加载／前向、校准或拟合，不能把准备通过当检测或刻度预测效果。**

## 一、本次实际范围

新增独立`mura_response/__init__.py`、`resources.py`、`inputs.py`。只读旧JSON元信息、源码、已确认设计、解释器／包元信息，并对模型／库文件作hash；不执行pickle或解析tensor。未安装、下载、建库、修改旧环境或入口。

CPU入口的模型／图像解码器导入守卫和图片／NPY／NPZ文件访问守卫均未触发，进程未加载Torch、torchvision、timm、PatchCore、PIL、cv2或旧生成模块。实际新增：真实RGB／GT／概率载荷读取0、正式刺激生成0、模型前向0、参数更新0、校准0、统计响应拟合0。

## 二、已固定的新数据角色

使用旧manifest的RGB内容组和来源元信息，排除旧N／Vnew／Tfinal／excluded_train、N_gen／S_bg／Q_cal、参考分支固定64组及已知RGB重复。1629候选RGB组／1634路径按组hash排序，组内取最小路径，PCG64种子2026102801单次排列后按以下顺序切片：

| 新角色 | 实际固定数量 | 首个路径（仅元信息） | 用途 |
| --- | ---: | --- | --- |
| Q200 | 200 | display/train/good/0603.png | 未来各模型本轮正常工作点 |
| F100 | 100 | display/train/good/1390.png | 阈值冻结后检查正常误报 |
| D_fit60 | 60 | display/train/good/1420.png | 未来统计响应拟合背景 |
| D_hold60 | 60 | display/train/good/0881.png | 统计拟合冻结后的背景留出 |

420组在原始已知RGB身份上互斥。当前仅核元信息和路径存在，未重新读取原图字节／解码，也未验证256工作输入是否发生碰撞；物理面板／采集分组未知。这些限制在协议和角色文件中显式保留，不能称420个独立工件。

完整角色见[roles.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/roles.json)。后续按固定角色读入时核原文件hash、工作RGB重复与覆盖；不能因碰撞或背景不合适换样。

## 三、几何、请求与坐标

固定256网格，中心(128,128)。圆、水平椭圆、45°斜椭圆三种几何，各有r=4／8／12。实际栅格面积如下，全部几何质心为(128,128)：

| r | 圆面积／像素 | 水平椭圆面积／像素 | 斜椭圆面积／像素 |
| --- | ---: | ---: | ---: |
| 4 | 49 | 53 | 51 |
| 8 | 197 | 199 | 201 |
| 12 | 441 | 441 | 453 |

它们只具有相同名义连续面积，不能称实际栅格严格等面积。固定M_geo和alpha各自内容hash已保存，面积不随幅度／亮暗变化。

请求按D_fit后D_hold，各自背景原序→9几何→亮／暗→幅度0.001／0.002／0.004／0.008／0.016／0.032／0.064。每背景126条，拟合／留出各7560，共15120条唯一请求ID；它们现在只是元信息，实际生成数量为0。未来两模型31080次输入引用是未去重预算，不是实际前向数或独立图数。

坐标为x,y像素中心，256到1024使用`(p+0.5)*4-0.5`，而非直接乘4；故工作中心(128,128)对应原图连续像素中心(513.5,513.5)，不是(512,512)。人工标量往返最大误差0。对应[geometry.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/geometry.json)、[requests.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/requests.json)。

## 四、人工数组的实际数值核验

人工条件为中灰、局部亮度隆起、空间梯度、全0、全1、接近1，共6背景×9几何×2亮暗×7幅度＝756条件。先在内存检查一次，封存时同协议再检查一次；每次都是756个人工条件，不累计成1512个真实刺激或独立背景。封存后的独立verify没有重新生成这些人工数组。

| 核验项 | 封存结果 | 准确含义 |
| --- | ---: | --- |
| 人工条件闭合 | 756/756 | 只针对预定人工数组 |
| 非零变化但旧M_eff为空仍保留 | 225 | 1/255不是float刺激准入下限 |
| 零实际变化请求保留 | 126 | 不改成检出成功或伪造缺陷FN |
| 背景近零导致代理未定义保留 | 126 | C／SEMU写null，不给分母加小数救结果 |
| 未裁切能量配对 | 486 | 不把裁切后的输入强行套幅度平方关系 |
| 未裁切能量比最大相对误差 | 0.00005489602433561558 | 小于预定float32数值余量0.0005 |
| 标量代理公式复算最大绝对差 | 0 | 是表达式一致性复算，不是独立测量链验证 |
| 坐标往返最大误差 | 0 | 仅人工坐标标量 |

全部预定人工条件的固定几何／区域外零变化要求通过。局部亮度隆起背景在零注入时，新配对对比度为0；未配对全局背景取法仍可出现非零偏移，说明配对定义确实隔离了这个人工空间偏差。Lb=0、1e-6附近及1.1e-6的分界也按`Lb>1e-6`核对。

当前人工条件以灰阶同通道输入为主，没有追加单独RGB通道的已知解析参照；Y权重正确性的依据仍包括批准公式和源码审阅，不称其已通过独立端到端测量链验证。更不代表真实背景覆盖、物理亮度或人眼阈值已验证。

完整证据：[numerical_verification.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/numerical_verification.json)。

## 五、来源、封存与实现审阅

[sources.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/sources.json)绑定25项来源（包含已确认设计）、三份新源码、解释器和包元信息。原PatchCore库、V1骨干、G1两最终权重及第1阶段关键来源在写入前后核对一致。

准备目录严格包含started、sources、protocol、roles、requests、geometry、numerical_verification、verification及complete九份JSON；complete绑定前八份文件。已有完整同身份目录只核验复用，未知／部分目录保留停止，不删标记、覆盖或自动补算。

独立只读代码审阅发现：若按整个包动态绑定源码，后续增加文件会破坏本准备缓存。封存前已改为显式绑定`__init__.py/resources.py/inputs.py`三文件，未来新增模块另绑自身身份。resources的守卫是进程永久状态，因此将隔离核验作为必要接口约束：后续读图／评分进程不得直接导入resources，先用独立元信息进程执行verify，再消费封存JSON与inputs纯数组接口。后续新增generation.py，不向封存inputs添加图像IO。

上述修正没有改动任何实验参数或旧成果。审阅中的“公式复算不是独立链验证”已作为本报告限制保留，没有为了消除措辞问题再追加验证项目。模型兼容、真实图像预处理及预测表现不属于本阶段完成项。

本次独立核验实际输出**PREPARATION_CACHE_VERIFIED**，8项输出身份、角色重建和请求字典核对闭合，模型／载荷守卫状态为空；没有重跑人工数值或正式刺激。见[verification.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/verification.json)、[complete.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/preparation_v1/complete.json)。

## 六、实际资源与已实现入口

封存时记录可用RAM5.23596GiB、D盘空闲119.50199GiB，满足2GiB CPU RAM和40GiB全轮磁盘预算。GPU查询为RTX5060 Laptop，8151MiB总显存、7899MiB当次空闲；没有初始化Torch／CUDA或运行GPU负载。资源是快照，不保证后续持续余量；GPU耗时仍未实测。

本阶段已由助手完成，无需用户再手动运行。需要只读复核时，实际已实现命令为：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.resources verify
```

输出应为PREPARATION_CACHE_VERIFIED。`inspect`只做当前元信息／资源观察，不写结果；其`frozen_roles_written=false`表示该次inspect不写入，不代表项目尚未封存。`prepare`在本完整同身份目录存在时也只核验，不重新抽样或重跑人工数组。没有正式生成／模型评分命令，本次不提供尚未实现入口。

## 七、当前剩余问题与下一阶段

尚待第3阶段实际CPU读入验证：固定D_fit60的来源文件当前字节、256工作内容碰撞、9几何的局部Lb与可定义背景覆盖，以及真实背景上弱变化、裁切和重复输入的实际数量。它们不因人工数组通过而自动满足。

下一项只按计划第3阶段另建独立generation.py，先做旧训练背景工作hash与D_fit60全部覆盖，再决定是否生成7560固定请求。Q／F／D_hold图像、GT、模型和概率保持未读取；不自动进入第3阶段，所有训练及统计模型拟合仍由用户手动启动。
