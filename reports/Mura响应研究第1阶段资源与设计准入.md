# Mura响应研究第1阶段：资源与设计准入

日期：2026年10月28日。范围：只读文献／既有报告、源码与JSON元信息、模型／库文件字节hash、路径存在与设备资源。用户已选择先完成数字域刻度验证，不加入本轮真实Mura迁移。

**结果：资源与元信息足够编写具体研究设计；可供新角色选择的候选为1629个RGB组／1634路径。尚未抽样、读新图、生成或评分；研究设计与执行计划仍待整体确认，不能把资源核对成功称为测量／预测研究已通过。**

对应：[独立设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[分阶段计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)。原材料和旧计划不修改。

## 一、实际核对及未执行边界

读取旧探针核查结论及原实现，确认旧流程将1/255训练有效标签门槛用作弱刺激准入、SEMU读数混入背景空间差异；新轮需要独立测量实现，不能直接复跑旧入口。

只对权重／FAISS／pkl文件作字节hash，没有执行pickle或解析tensor；没有导入Torch、torchvision、timm、PatchCore、旧render或探针运行包，没有模型构造、加载、前向、建库、训练、生成或统计拟合。`nvidia-smi`只查询设备。没有打开任何RGB／GT文件字节、图像解码、刺激NPZ或概率载荷；旧JSON中的阈值和身份属于元信息。

这不是对2806份旧探针文件的再次全量审计；本次只核列出的关键来源和文件。旧核查仍以原封存结果为准，不重新计算其统计。

## 二、候选数据与角色排除

使用旧`dataset_manifest.json`中的路径、文件hash、RGB内容hash和尺寸。2400条train/good记录均为1024×1024元信息。排除下列角色及匹配RGB组：

| 已有角色 | 元信息路径数 |
| --- | ---: |
| 原N正常库 | 80 |
| 原Vnew校准 | 479 |
| 原Tfinal | 1100 |
| 原excluded_train | 1 |
| N_gen训练背景 | 80 |
| S_bg旧共享评价背景 | 20 |
| Q_cal旧正常校准 | 40 |
| 真实参考分支旧固定审阅范围 | 64 |

这些数量按来源列出，集合可能重合，不能相加当作不同正常图总数；Tfinal也包含异常路径。原参考v1两背景均在64组审阅范围内，已覆盖。

按RGB组排除后，候选为**1629组、1634路径**；每组选字典序最小路径时，1629条代表路径全部存在。未执行PCG64抽样，也未将420角色写入数据文件。候选RGB身份来自旧已解码manifest，当前没有重新读图核验；下阶段实际读入时仍须核原文件hash。

候选足够支撑拟定Q200／F100／D_fit60／D_hold60，但数量充足不代表局部亮度条件必然满足。物理面板／批次分组未知，新角色只是路径与已知RGB互斥，不称全新工件独立验证。

## 三、冻结模型与实际文件身份

PatchCore元信息：80正常图、输入256、coreset0.01、819个1024维向量、ImageNet V1骨干。原tau=1.8928574562072753，保留不改；新轮Q200工作点尚未计算。

G1原核验记录为最终10000步、10保存点、stateless BN；本次核最终两文件hash与训练记录相同。原Q_cal阈值0.7155899673700332保留不改；不加载中间权重。它在参数化模拟上训练的偏好仍须报告。

| 文件 | 字节数／SHA256 | 本次核对 |
| --- | --- | --- |
| PatchCore nnscorer_search_index.faiss | 3354669；`4e452bc6ac0fdab8bbb2a869ab7009ac98b4c8ac63168295af98e96c5a71b7b1` | 与bank记录一致 |
| PatchCore patchcore_params.pkl | 232；`f6ccc3e6dc10816392b80c3de4f00f0cb56f66c2d96f0a256eaa917aa2c86667` | 与bank记录一致，仅hash |
| wide_resnet50_2-95faca4d.pth | 138223492；`95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8` | 本地资源存在，通用V1身份 |
| G1 reconstruction.pth | 276225877；`eb5fd05c44317f36f367ab89d734a380361d9a53df4b69a8c14579613f4edb0a` | 与训练／第8天核验记录一致 |
| G1 segmentation.pth | 113536931；`6e1b991203df330f1687b88dfd565077650372f0f1e8bc3fa50c9725d7b64fba` | 与训练／第8天核验记录一致 |

文件存在且hash一致，不证明未来新包装已兼容。评分时仍须独立核对初始化／预处理及参数／BN前后不变，本次不进行前向验证。

## 四、实际资源快照

| 项目 | 当次观察 |
| --- | --- |
| 系统总RAM | 15.62198GiB |
| 系统可用RAM | 6.78590GiB |
| D盘空闲 | 119.50214GiB |
| E盘空闲 | 724.99006GiB；本轮未写入 |
| GPU | RTX5060 Laptop，8151MiB总显存／7899MiB当次空闲 |
| 驱动 | 595.97 |
| 项目Python环境 | 现有`.venv`，本次未安装或升级 |
| 包元信息 | NumPy1.26.4／SciPy1.16.2／Pillow12.3.0／OpenCV4.11.0.86／scikit-learn1.7.2／Torch2.8.0+cu128／torchvision0.23.0+cu128 |

资源是查询时快照，不保证长时间运行余量；模型包版本来自安装元信息，不等于本次导入／兼容验证。拟定40GiB全轮磁盘预算可在D盘安排；当前无须下载、安装、换盘或启动新训练。GPU总耗时待实际Q评分估计。

## 五、关键来源绑定

以下hash用于下一阶段重新核对观察来源，不将本次读取冒充早期运行时完整封存证明。

| 来源（相对弱缺陷Demo根） | SHA256 |
| --- | --- |
| data/weak_localization/split.json | `feb206cd18eeb9594c64b30afead64577f34473bda57bcaef5a58b3294124ddb` |
| data/controlled_generation/split.json | `765aa3f0be86f36f2c5370b0729d18daaa2b63e16d7df8271c89073d27b2750b` |
| results/weak_localization/dataset_manifest.json | `9d0bf06efd17660e2a0ab5e3e3d45337021b7305490f89f1c32512a4f479a270` |
| results/weak_localization/patchcore_bank.json | `582172a55e87de5e4a924f5d90c489dc910acb477ede199257c06e27112b2c42` |
| results/weak_localization/frozen_config.json | `93c29798fdb301363e97059ac02a5e3a389698794a21cd3b991fb1fdccb35ff9` |
| results/controlled_generation/formal_preparation/common_config.json | `51ad58ea35d76e10d10e545e51b10b91ee56b942e6e3630101ddd77680800353` |
| results/controlled_generation/calibration/calibration_verification.json | `5d1e4bd432b0b1baae41114157b82beaa8b81fe7acaaff39740fbbe8f157fb72` |
| results/reference_generation/v2_domain_matched/scope_protocol.json | `5a9b467f718ad71f60da066711d5faebcb623af2f9c3e8802442545a77044b73` |
| results/pod_probe/audit_20261028_v1/verification.json | `628a73188da9c7e1635b1ce4012811207e68c8bcc93757ac5d52466f964df705` |

本次新产物只有研究设计、执行计划与本资源报告。独立只读设计审阅实际完成，指出三个重要缺项：工作分辨率跨角色重复、可定义背景口径及H停止时机、常量概率参照。已在设计／计划中逐项补清，并明确CPU统计模型拟合也由用户手动启动；这是设计修订，不是运行验证。没有重新分配背景、读图或开发模块。

下一项是审阅确认具体设计，然后实施第2阶段CPU元信息与人工数组准备；没有尚可运行的新实验命令。
