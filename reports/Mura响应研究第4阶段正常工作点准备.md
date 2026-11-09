# Mura响应研究第4阶段正常工作点准备

日期：2026年10月29日。范围：第4阶段独立准备，尚未取得Q/F实际输入或执行评分／校准。

**独立inference控制模块、显式运行模块与normal_preparation_v1已准备并封存；独立元信息进程核验通过。当前原图／输入数组／概率打开、Torch／模型导入、参数加载、模型前向、实际校准、训练与统计拟合均为0；尚无新tau或实际Q／F误报结果。**

依据：[冻结研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[总执行计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)、[第3阶段独立CPU输入结果](D:/华兴源创/报告/Mura响应研究第3阶段独立身份适配与CPU取得结果.md)。私人设备、公开材料、个人学习与非商业研究，公司材料隔离；未下载、安装、改环境或改旧成果。

## 一、本次真正核验的范围

在独立进程运行resources verify，preparation_v1的8项输出、420角色和15120请求身份通过；没有重算人工数组。准备complete SHA256仍为`0fd936ea58f8c8a7d16c209c96d116daa0f34c6044d1b809781bd4b01f7c7ba5`。

fit_inputs_identity_v2引用此前独立全量核验证据，complete SHA256仍为`3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593`。本次核该清单、8份输出JSON的hash、固定请求／唯一输入所推导的14883文件严格目录清单、文件stat及原独立结果报告绑定。**未打开或重新hash／解析任何NPY／NPZ载荷，不能称本次又做了全数组内容核验。** 之前实际证据与这次元信息核对分开保留，旧v1失败不改为通过。

两冻结模型的5份资源文件实际作了完整字节hash，与既有封存来源一致；没有解析tensor、反序列化FAISS或执行pickle。

| 模型资源文件 | 实际字节SHA256 |
| --- | --- |
| checkpoints/wide_resnet50_2-95faca4d.pth | `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8` |
| G1_20261030/reconstruction.pth | `eb5fd05c44317f36f367ab89d734a380361d9a53df4b69a8c14579613f4edb0a` |
| G1_20261030/segmentation.pth | `6e1b991203df330f1687b88dfd565077650372f0f1e8bc3fa50c9725d7b64fba` |
| patchcore/nnscorer_search_index.faiss | `4e452bc6ac0fdab8bbb2a869ab7009ac98b4c8ac63168295af98e96c5a71b7b1` |
| patchcore/patchcore_params.pkl | `f6ccc3e6dc10816392b80c3de4f00f0cb56f66c2d96f0a256eaa917aa2c86667` |

PatchCore正常库元信息仍为N80、819个1024维向量、256输入、coreset0.01；G1仍为20261030种子最终10000步，原10保存点的既有训练核验证据通过。本次只核最终资源与旧核验证据，没有读取完整训练检查点tensor或重新验证训练。PatchCore固定来源沿用本地原记录commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`；本次绑定当前vendor逐源码hash及LICENSE，没有重新取得上游压缩包来证明其历史逐文件身份。DRAEM作者model_unet.py沿用原封存hash。

## 二、独立实现和协议

新增[inference.py](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference.py)承担纯元信息准备、显式CLI、永久进程守卫、虚构标量核验与协议封存；新增[inference_runtime.py](D:/华兴源创/实验/弱缺陷Demo/mura_response/inference_runtime.py)承担未来明确调用的CPU正常输入取得、单模型评分和缓存核验。后者顶层只有标准库导入；NumPy／Pillow／模型导入均置于相应显式运行函数中，本次没有调用它们。

准备输出为[normal_preparation_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1)，严格4文件：[sources](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/sources.json)、[protocol](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/protocol.json)、[verification](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/verification.json)、[complete](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/normal_preparation_v1/complete.json)。没有向preparation_v1或fit_inputs_identity_v2追加文件，没有修改任何已封存模块／全局路径、模型、旧阈值或报告。

新实现绑定如下：

- inference.py：`32eb1d45800fd56030a19d45a3108ec29316bcb1bbd805ab55d9cb8e1a09895b`。
- inference_runtime.py：`59343c5b8c0d5a819b962cc89fcc4afd2939f99a24bca17470c932eaadf2c240`。

protocol SHA256：`ea64281c4ca714f30d557dc7215a57805329b61f6d8f9ba5e95cbe0903bfe4d7`。complete SHA256：`b27ab761fc0af094d98d221546e64ca053efb7216e5ac6862b306c947e46dc77`。动态总计划及本准备报告不作新协议整份hash，避免正常同步进度破坏缓存身份。

本次交付前比对31项既有源码／JSON／设计／报告hash，以及原14883份完整输入和13份失败文件的目录stat摘要，均与执行前一致；本轮对旧数组始终只核stat，不把stat一致当作重新证明数组字节。

## 三、冻结的数据、评分及阈值口径

Q200／F100路径、ID、ordinal及旧文件／RGB组身份按preparation_v1原序绑定，不重新抽样。角色元信息互斥不等于本次已经核过这些原文件或工作域重复；实际来源／工作碰撞留到显式CPU取得。

未来运行接口拆为共享CPU输入与逐模型GPU事务：

1. CPU按固定Q200后F100实际取得300张，每原图只打开字节、解码一次，保存规范RGB256 float32。核原文件hash、原尺寸／模式、独立新原始RGB摘要和规范工作hash；旧组ID只作角色／排除，绝不猜旧序列化。完成全300后核角色内、互相、与D_fit60及旧训练160的精确工作重复；任何碰撞保留停止，不删样／补数，不进入GPU。
2. 正常输入独立完整核验后，逐模型新进程，PatchCore与G1分别只评分Q200，再冻结自己的新tau，才评分F100。全池工作内容互斥门槛满足时，每模型300次唯一输入评分预算；两模型600次引用仍只有300个正常背景，不称600独立图。当前这300／600都是预算，实际取得／前向仍为0。
3. Q新阈值固定q=0.95、linear、严格score>tau。只首次完成Q200时取一次分位数并保存；完整缓存复用只读原tau，不重新取分位数。旧tau、旧Q分数及旧测试效果不用来定本轮工作点。
4. Q报告FP/200、TN/200，是定阈值同池计数，不是独立验证；F报告FP/100、TN/100及z=1.959963984540054的双侧95% Wilson区间，不回调tau。分数全常量或阈值在端点时明确限制，不改评分规则或分位数，不预填实际FP必为10/200。

两模型共同FP32、batch1、workers0、eval、4线程，关闭TF32和cudnn benchmark，各自预处理保留：

| 模型 | 输入与分数／热图 |
| --- | --- |
| PatchCore | RGB uint8双线性256再float32/255，另作固定ImageNet mean/std归一化；保留作者最大平方L2最近邻patch分数，热图保留作者双线性回256和sigma4 Gaussian语义；不再平滑、拉伸或跨池归一化，整图分数不改成热图max |
| G1 | 同一规范RGB float32/255，不额外ImageNet归一化；最终10000步重建／判别网络，22＋27共49个stateless BN、空buffer；保留原始异常通道softmax图和其最大值，不池化／平滑／拉伸 |

PatchCore构造保留作者单张ones特征维度探测及其BN初始化语义；它在模型构造时发生，和300次真实输入评分分开计数。**本次没有构造模型，ones探测也为0。** 状态快照在模型完整加载和eval之后建立；评分前后逐图核完整参数／buffer／training／grad状态。G1另核原最终参数hash、49个stateless BN及空buffer。只有未来实际加载才能证明这些运行时条件，本次不称已测兼容。

预测函数只接收RGB和模型，不接收GT、区域、角色、阈值或测量标签。通用probabilities缓存名下，PatchCore存的是作者连续分数热图，不能因目录名把它当作[0,1]概率；G1才是[0,1]异常softmax。

## 四、事务、完整复用与停止规则

CPU正常输入拟输出normal_inputs_v1，单模型输出拟在normal_scoring_v1/PatchCore及G1。本次这些目录都不存在。

新目录使用排他创建；模型构造及每次前向之前保存inflight，输出和状态检查提交后才清除自己的标记。任何已有未知／部分事务、残留inflight或failure均保留停止；不自动续推、补推、删除标记、改来源或另开版本。已完整且来源、实现、环境一致的缓存只读核验复用，不实例化模型、不重复前向或重取tau。

冻结模型文件、作者源码、解释器／依赖元信息、旧准备与D_fit证据、新正常输入及对应协议身份分别绑定；strict目录白名单由固定Q/F字典和预定文件构造，不遍历收编未知文件。异常不取模型成功交集缩分母。前向计数在尝试前增加；部分事务必须同时看committed记录数与inflight，不能把未明尝试叫成功评分。

当前准备核验严格禁止一切原图／NPY／NPZ和模型导入，模型文件仅明确白名单字节hash。未来实际CPU取得、GPU评分和缓存数组核验各需显式入口；不导入resources或generation_v2来绕过原永久守卫，不修改旧模块路径。

## 五、实际资源、标量与审阅

准备时可用RAM8.096230GiB、D盘空闲113.686581GiB；显卡NVIDIA GeForce RTX 5060 Laptop GPU，驱动621.01，总8151MiB／空闲7899MiB（约7.714GiB）。这些是元信息快照，未加载模型，不能当实际峰值或运行充分性的保证。

沿用CPU至少2GiB RAM、GPU启动至少4.5GiB RAM／4GiB空闲显存、D盘至少40GiB；实际运行启动和输入边界检查，资源不足停止而不改batch、精度或样本。Q/F取得时间、模型兼容、Q评分速度及GPU实际峰值均未实测，不从第3阶段CPU生成或旧训练时间推算。

环境只核软件包元信息：Python3.12.7、NumPy1.26.4、SciPy1.16.2、Pillow12.3.0、Torch2.8.0+cu128、torchvision0.23.0+cu128、faiss-cpu1.12.0、timm1.0.20、tqdm4.67.1；没有导入这些模型包验证运行兼容。

虚构标量核验使用0—199共200个数：linear q.95为189.04999999999998，严格大于计数10；200个相同虚构值计数0。Wilson端点核验仅是假设0/100和100/100，未评价实际F。最初fp=0端点出现3.47e-18的浮点消减残值，已按解析端点显式置0（fp=n置1），未改z、区间公式或放宽验收；修正后标量通过，未创建软件测试文件。

执行方式按用户要求在现有非Git项目中使用独立模块和结果目录，没有worktree、过程ledger、软件测试套件或提交／合并流程。一次独立整段源码审阅确认静态边界、模型口径与Q→F逻辑；两项原评为资源报告Minor的意见，在执行判读中因影响后续预算／峰值解释定为封存前需修复项，并在同一修正轮补齐：
- Q200从模型加载后到最后Q缓存提交单独计时，包含逐图状态核对和IO，排除分位数计算；虚构时间10→60秒核每引用0.25秒，仅验证标量关系，未测GPU。
- GPU峰值在CUDA准入后、模型创建前重置，覆盖构造和全部300输入，保存明确计时／峰值范围。静态核对重置位置早于构造，没有实际运行CUDA。

两项只补资源证据，不改数据、模型、分数或阈值口径；无延期审阅项。审阅未判断的实际图像／模型兼容、速度、资源峰值及完整运行结果均保留未验证；旧D_fit载荷的新鲜内容有效性沿用本次不读取的边界，不能用静态通过代替实际兼容或效果。

## 六、独立准备核验证据及已实现运行说明

独立元信息进程实际返回NORMAL_PREPARATION_CACHE_VERIFIED，配置hash与已封存protocol一致，退出码0；保存准备期verification同样显示所有数据／模型操作0。以下为本次独立核验结果及其封存绑定：

```json
{
  "status": "NORMAL_PREPARATION_CACHE_VERIFIED",
  "configuration_sha256": "ea64281c4ca714f30d557dc7215a57805329b61f6d8f9ba5e95cbe0903bfe4d7",
  "output": "D:\\华兴源创\\实验\\弱缺陷Demo\\results\\mura_response\\semu_applicability_v1\\normal_preparation_v1",
  "boundary": {
    "source_decodes": 0,
    "array_opens": 0,
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
  },
  "preparation_complete_sha256": "b27ab761fc0af094d98d221546e64ca053efb7216e5ac6862b306c947e46dc77"
}
```

可复制的只读准备核验命令已经实际实现并运行过：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.inference verify-preparation
```

inspect与prepare也已实现；prepare遇完整同身份缓存只读复用。acquire-normal-inputs／verify-normal-inputs、单模型score-normal／verify-normal入口已实现并经源码／语法审阅，但实际数据和网络兼容尚未执行；normal_inputs_v1未完成，所以当前不交付GPU正式启动命令或声称评分准入已通过。

本轮只完成准备；唯一下一项为显式CPU取得并核验Q200／F100，不自动执行。当前唯一可发送指令见总执行计划第六节。实际校准／F检查、D_fit评分和CPU统计模型拟合都尚未开始，所有训练及统计拟合仍由用户手动启动。

