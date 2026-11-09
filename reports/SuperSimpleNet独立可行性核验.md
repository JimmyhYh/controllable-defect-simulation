# SuperSimpleNet独立可行性核验

2026年10月21日。范围：作者论文／公开源码静态审阅、本地环境元信息、资源快照与通用权重文件hash。未安装、下载权重、开发适配代码、导入Torch、实例化模型、前向、生成或训练，未打开原图／GT或读取旧评价分数选型。

## 一、结论与主候选

**结论：有条件可行，支持进入独立执行计划编写；尚未通过运行准入，也未证明性能提升。** 可复用本地KSDD2固定角色与通用ImageNet权重。主要工程工作在数据隔离、移除旧框架依赖、独立训练／评分包装以及资源预算，而非再训练一个生成大模型。

**主候选：作者固定提交中的ICPR2024式SuperSimpleNet配置，适配严格R5像素监督。** 固定源码参照为`blaz-r/SuperSimpleNet@98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d`，提交页显示为更新权重链接的提交，固定地址可访问。本报告不声称它一定是此刻远端最新HEAD。原论文为[SuperSimpleNet, arXiv:2408.03143v1，2024](https://arxiv.org/abs/2408.03143v1)，仓库另有2025混合监督扩展。[固定提交](https://github.com/blaz-r/SuperSimpleNet/commit/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d)、[版本与参数说明](https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/README.md)

首选2024式而非直接套用main默认混合监督：现有R5全部有像素标注，没有批准使用U241图像级标签；首轮不需要引入混合监督。该提交通过`adapt_cls_feat`兼容两个分类特征分支，README说明`dt`、`dilate`、`adapt_cls_feat`是两版本主要差别。**建议候选为adapt_cls_feat=True、dt=None、dilate=None，保留作者特征异常生成与训练损失**；这些是供计划采用的候选，不是已经运行验证的冻结配置。由于数据划分、依赖、batch和评价分辨率需要适配，应称“2024式严格R5适配基线”，不称作者论文原样复现。

当前研究问题先收束为：成熟预训练检测器在严格R5条件下是否能形成可用、可重复的真实定位基线。首轮不叠加旧RGB可控合成，也不先开展原C组曝光对照。作者的特征合成本身保留，因此基线应称“未加外部RGB合成”，不能称“完全没有合成”。以后若比较外部合成，必须固定检测器再作对照。

### 候选比较

| 候选 | 适配判断 | 本轮决定 |
| --- | --- | --- |
| 固定提交的ICPR2024式分支 | 支持标注异常，方法组成较明确，严格R5需新数据包装 | 主候选 |
| JIMS2025默认混合监督 | 有额外距离加权／训练标签处理，默认KSDD2入口使用246标注条件 | 保留为后续方法候选，不与首轮同时运行 |
| Anomalib内置无监督SSN | 作者说明其中提供的是无监督版本，不自动等价于R5监督版本 | 不作为主候选的无声替代 |

## 二、可复用资源与许可证据

### 2.1 通用骨干权重已具备

作者`FeatureExtractor`明确请求torchvision的`IMAGENET1K_V1`，使用WideResNet50-2的layer2／layer3。[固定feature_extractor.py](https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/model/feature_extractor.py)，定位：构造函数及forward。

本地文件：`D:/华兴源创/实验/弱缺陷Demo/models/checkpoints/wide_resnet50_2-95faca4d.pth`，**138223492字节**，本次重新按文件字节计算SHA256为：

`95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`

与既有资源记录一致；旧环境torchvision0.23.0的`models/resnet.py`文本将该文件名指向ImageNet V1官方URL，hash也匹配文件名公布的前8位。因此它是身份明确的本地复用候选，无需为了规划再下载。未反序列化或验证实际参数键／dtype；完整加载匹配留到获准模型阶段。该hash不是从官方独立全长校验清单取得的签名，不夸大为官方全长认证。[官方同版本定义](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/models/resnet.py)

只允许通用ImageNet骨干作为预训练来源。禁止直接用作者KSDD2任务权重，它可能包含R5以外监督；旧PatchCore正常库、旧GN／DRAEM权重也不作SSN初始化。未来显式指定本地骨干路径并核对hash，不能让构造器自动联网下载其他权重。

### 2.2 许可及来源保留

作者该提交的仓库LICENSE为MIT，保留版权与许可文本。[固定LICENSE](https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/LICENSE)。但其helper另有来源：focal loss注明来自Apple DeSTSeg，Perlin代码注明来自DSR。已读取[Apple来源许可](https://github.com/apple/ml-destseg/blob/main/LICENSE)及[DSR Apache-2.0许可](https://github.com/VitjanZ/DSR_anomaly_detection/blob/main/LICENSE)；未来保存vendor时需连同来源注释／许可分开保留，不把所有文件简单改标为本项目原创MIT。

Torchvision代码许可与预训练权重条件不是同一件事；其[官方预训练权重声明](https://github.com/pytorch/vision#pre-trained-model-license)提示可能有训练数据衍生条款。本次没有找到该V1文件单独、完整的权利授予文本，不据仓库代码许可声称权重可任意商用。本项目继续限已确认的个人学习／非商业研究，资源准备阶段需保存可获得的权重来源／使用说明，若出现与该用途冲突的明确条款则停止，不自行换权重。

KSDD2数据来源、CC BY-NC-SA4.0及本地原包核验沿用[既有可行性报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练_数据与实验可行性核验.md)。公司材料隔离。此次未重新读取归档、数据图像或GT，也不新增数据使用用途。

## 三、必须做的适配

### 3.1 数据：严格R5不等于只改标注数量

作者KSDD2枚举支持0／16／53／126／246，默认入口为N246混合监督，划分通过pickle读取。[固定ksdd2.py](https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/datamodules/ksdd2.py)，定位NumSegmented、read_split；[固定train.py](https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/train.py)，定位main_ksdd2、run_sup。

必须独立使用旧split的路径／hash元信息建立JSON请求清单，只允许R5和N1885进入训练数据集。R5为11670／11884／12070／12245／12303；Q200只作未来新模型正常校准，U241不得以“未分割异常”或图像级标签混入训练。E1004不实例化进训练侧loader。禁止调用作者pickle或直接调用会同时setup train／test的原数据模块。原data.py及其准备结果保持不变。

作者每epoch的正常／异常采样、翻转扩展及频率调整与旧固定四槽序列不同。新计划必须写清独立图数、每轮采样及实际曝光次数；不能将只标5张但仍用其他异常图的配置称严格R5，也不能把小batch下的“300 epochs”直接当旧10000步的同等预算。

### 3.2 方法：保留核心，移除不必要的训练框架耦合

保留通用骨干、特征适配器、特征噪声／Perlin生成、定位头和分类头，以及作者分割／分类损失与优化器分组。损失不沿用旧CE＋Dice；不能擅自用旧GN结构替换作者模型，或把旧训练成功当作SSN可学习性已通过。核心源文件静态依赖主要是Torch／torchvision及本地helpers，因而**移植核心、独立包装数据／训练／指标**比整体安装旧anomalib训练栈更值得优先准备。这是可行性推断，尚无实际导入或运行证据。

未来建议新目录`ssn_baseline/`、`vendor/SuperSimpleNet_<commit>/`、独立结果／模型根`ssn_baseline/ksdd2_r5_v1/`；不覆盖旧fewshot_synthetic、作者DRAEM或render源码。必须先取得该固定commit源文件和各文件hash，当前只完成公开固定地址审阅，尚未取得本地封存源码。依赖应安装在独立`.venv-ssn`，不修改旧`.venv`。

### 3.3 图像、标注与输出：不生搬旧评分口径

建议首候选采用作者整图640×232与ImageNet均值／标准差归一化，保留作者图像上下文。旧实验是原生256窗，两者不同；新结果可作方法整体比较，不能将差异只归因于预训练。旧角色划分不变，不代表所有旧预处理都应原样沿用。[作者数据转换实现](https://github.com/blaz-r/SuperSimpleNet/blob/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/datamodules/base/datamodule.py)

原生GT始终原样保留。训练GT缩放采用离散标签规则；作者另将GT双线性缩到特征网格再以0.5二值化，可能使细缺陷消失，未来CPU训练侧核验必须审阅全部R5。发现某图监督丢失，先保存并审阅，不换R5、不借U、不开E挑配置，也不暗中加膨胀救场。是否保留该特征标签处理及必要替代须在计划中明确。

作者输出先上采样并对logit作Gaussian sigma4平滑，再sigmoid；图像级主分数来自分类头，另报告定位图最大值。新计划应固定作者平滑与返回原生分辨率的插值／sigmoid顺序，逐图保存原生float32定位图和分类分数；像素评价GT维持原生尺寸与二值，不用缩放后的GT替代。主指标仍建议全局精确像素AP，整图AUROC与正常阈值计数补充；分类头与定位图最大值分别标明，不复用旧模型τ或把它们称同一评分。[固定模型实现](https://github.com/blaz-r/SuperSimpleNet/blob/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/model/supersimplenet.py)

作者test把整个评价池拼起来做min-max归一化，并周期调用test；未来应独立只读评分，关闭测试池驱动的归一化及训练期E评价。保存非池依赖的原始分数以支持Q阈值；不把这个适配称作者原评价流程，不把原有阈值照搬过去。旧exact_ap可作为只读CPU算法参照，但不得改旧实现或重算旧指标。

### 3.4 小试和评价角色

先静态／CPU准备，再由用户显式手动启动训练侧资源与可学习性小试，之后才审阅正式三初始化预算。不预先承诺若干分钟或沿用旧probe AP门槛／1000步作为已证实SSN验收条件。小试只能用明确固定的R5和少量N训练材料，不能据同池高AP选“泛化最好”的候选。

默认不再新增随机验证角色：在没有独立异常验证材料时，首轮以文献预设与资源关口固定配置，不对E调参。Q仍正常校准而非模型选择。现有E1004已经被旧研究评价并展示，SSN未来即便仅一次最终评分，也须标为旧基准上的新方法探索比较，不能再称未使用的新独立最终证据。若需要确认性结论，先核验另一个未参与选择的真实评价来源；本次不自动挪用U或重划N作为新验证，也不下载新数据。

## 四、设备、依赖与资源条件

### 4.1 本次只读实测快照

| 项目 | 实际记录 | 含义 |
| --- | --- | --- |
| GPU | RTX5060 Laptop，8151MiB总显存，7899MiB空闲；驱动595.97 | nvidia-smi静态快照，非SSN峰值 |
| RAM | 总15.621983GiB，当次可用4.290112GiB | Windows GlobalMemoryStatusEx，不是连续监测 |
| D盘 | 空闲155.125420GiB | 当次快照，不代表后续完整缓存后余量 |
| 现有核心包 | Torch2.8.0+cu128、torchvision0.23.0+cu128、NumPy1.26.4、SciPy1.16.2、Pillow12.3.0 | 从包元信息读取，未导入Torch |
| 未安装的框架依赖 | anomalib、albumentations、pytorch-lightning、torchmetrics | 仅旧.venv检查范围，不声称整台机器均不存在 |

作者requirements是Torch2.1.0+cu118／torchvision0.16.0+cu118／anomalib0.7。[固定依赖清单](https://raw.githubusercontent.com/blaz-r/SuperSimpleNet/98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d/requirements.txt)。优先准备与已成功运行本机GPU的2.8／0.23／cu128一致的隔离核心环境，避免为复现旧依赖破坏当前环境。能否用核心移植消除anomalib等依赖须开发时核验，不能标为已兼容。

### 4.2 内存与时间：只给工程准入，不伪造实测

640×232输入下，根据层步幅推算，拼接特征约1536×160×58；单张FP32这一份张量约54.375MiB。batch32仅此一份就约1.699GiB，尚不含适配特征、成对噪声、激活／梯度、优化器与卷积工作区。作者batch32不能作为本8GiB设备的默认可运行配置；以上是形状静态估算，不是显存峰值证明。

供执行计划采用的保守起点：**FP32、batch1、workers0、整图640×232，单模型串行**；先手动小试测量再固定正式预算，不自动降精度、减分辨率或换骨干。训练／前向阶段建议启动可用RAM至少6GiB、空闲显存至少6GiB、磁盘至少80GiB，运行采样RAM不得低于3GiB。这些是新分支工程门槛候选，不是作者要求或充分性保证；当前4.29GiB快照不满足建议模型运行起点。静态准备不需要提前加载模型来证明资源够。

epoch数必须重算预算。仅作算术说明：假设每epoch遍历1890张训练图且batch1，300epochs将有567000次更新；这不是作者频率采样器的实际步数预测，却说明不能只缩batch而忽视时间。正式步数、曝光组成、LR单位与检查点间隔需在计划阶段明确；完整耗时只能在用户小试后估算，不套用旧GN模型的11—24分钟。

若复用旧精确AP口径，6套旧大表已经占约47.6GiB，新分支还需独立空间；新三模型表约23.8GiB只是容量估算，概率／环境／检查点另计。不得清理旧结果腾空间。是否首轮三模型或分阶段完成由新计划固定，不能暗中只留最佳seed。

### 4.3 静态构造也可能触发前向

`FeatureExtractor.__init__`调用get_feature_dim，其中用torch.rand作一次真实前向；默认骨干构造还可能触发自动下载。因此“只实例化、不训练”也超出本次静态核验范围。未来入口必须区分元信息准备与获准模型操作，资源小试中若含参数更新，一律由用户手动启动。

## 五、停止规则与必要备选

1. 固定commit文件取不到、来源／许可说明冲突、骨干hash变化或任务权重监督范围不明：保留证据，停止资源准入，不随意跟随main或换权重。
2. 任意U／Q进入训练、作者pickle被调用、训练期访问E或按E选模型：停止并查因，不以混合监督默认行为作为例外。
3. R5坐标或缩放不符、特征网格标签丢失、有效域／原生GT处理不明：保存并审阅，不换样本、救标签或据E观感调整。
4. 环境依赖需要覆盖旧.venv、核心移植不能保持作者计算、独立导入产生未批准下载或前向：停止；先提出单项适配，不顺带换损失、归一化或模型。
5. 资源不足、OOM、非有限损失／梯度、预测全常量等异常：保留自己的明确事务，用systematic-debugging定位，不自动加步数、挑中间权重或改阈值。
6. 所有运行使用独立配置／结果／模型目录与源码身份；partial、未知锁或不明确前向保留停止，不自动重跑。是否支持完整训练恢复须真实核验，未实测就披露。
7. 若全图batch1仍不可承受，必要备选是先另立原生256窗口适配提案，重新定义分类头跨窗聚合与边缘行为；它不是本次自动批准的降规格，也不是与整图同一方法配置。环境问题不自动切到只用正常图的Anomalib版本冒充严格R5。

## 六、核验边界与后续交付

本次已明确主候选与固定源码参照、需要的适配及资源／停止规则；已有骨干身份核验通过，且无需新的生成预训练权重。**可进入执行计划编写，不能进入训练或直接报告检测成绩。** 尚未完成：本地vendor封存／逐文件hash、完整权重条件说明归档、隔离环境安装与核心导入、R5缩放／特征标签适配的CPU内容核验、GPU小试、完整恢复及新的独立最终评价来源。

遇到的工具问题未扩大任务：本地git ls-remote因getaddrinfo线程错误未取得远端HEAD，GitHub API入口亦不可用，改用作者公开提交页及固定raw地址取得可引用的具体commit，不声称完成本地clone；CIM内存查询不可访问，改用只读Windows内存API成功得到快照。未安装软件修复网络或权限，也未删除任何文件。

已据本核验编写[独立SSN分阶段执行计划](D:/华兴源创/计划/SuperSimpleNet严格R5基线执行计划.md)，未实施其中任何阶段。下一项仅第1阶段静态来源／资源准备；计划先完成来源／环境／严格数据包装准备，再提供真正实现后才能使用的人工小试命令，待小试核验后才推进正式预算与校准／评分。所有训练均由用户手动启动；计划不设过程表、勾选追踪、软件测试套件或提交流程。旧一周报告、所有源码／配置／权重／阈值／评分与图件以及历史失败保留。

附：本核验仅静态审阅公开论文、固定源码地址和本地元信息，未打开新图、GT、概率数组或旧指标作选型；未执行项目CLI、Torch导入、模型构造、优化器、前向、反向、生成、训练或校准。源码、环境及运行兼容性的剩余未知如上列明，未将静态可行当成运行成功。
