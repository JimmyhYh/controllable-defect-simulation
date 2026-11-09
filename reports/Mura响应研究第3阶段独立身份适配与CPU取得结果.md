# Mura响应研究第3阶段独立身份适配与CPU取得结果

日期：2026年10月29日。题目：《受控数字Mura扰动下的检测器响应表征与SEMU代理刻度适用性研究》。

**第3阶段独立身份适配A—D已完成：60/60背景的540/540项几何测量可定义，无本阶段工作内容碰撞；7560个固定请求各生成一次并完整封存，独立进程全量只读核验通过。当前成果是数字刺激与测量输入，尚未取得本轮检测器响应、正常工作点或SEMU留出预测结果。**

依据：[冻结研究设计](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度研究设计.md)、[身份适配补充方案](D:/华兴源创/计划/Mura第3阶段独立身份适配与重新取得补充方案.md)、[原失败诊断](D:/华兴源创/报告/Mura响应研究第3阶段CPU生成与停止诊断.md)。本报告不修改它们；实时进度统一见[总执行计划](D:/华兴源创/计划/Mura检测器响应与SEMU代理刻度执行计划.md)。

## 一、身份适配与旧现场保留

独立实现为[generation_v2.py](D:/华兴源创/实验/弱缺陷Demo/mura_response/generation_v2.py)，结果在[fit_inputs_identity_v2](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2)。不导入旧generation.py、不覆写旧模块全局路径；resources verify在隔离进程核验，父进程只消费封存JSON和inputs纯数组接口。

身份字段分别使用：

| 身份 | 本次语义 |
| --- | --- |
| 原文件file_sha256 | 原PNG完整字节SHA256，逐文件严格核旧manifest；D_fit另核固定角色文件hash |
| legacy_rgb_group_id | 原样继承旧组ID，仅作角色／分组／排除；历史序列化仍未知 |
| decoded_RGB_u8_sha256 | 本次convert('RGB')后的1024×1024×3 uint8 HWC C字节SHA256，无头，另存域定义 |
| working_RGB_sha256 | RGB uint8先Pillow bilinear到256，再float32/float32(255)；沿用冻结inputs带dtype／shape／C头摘要 |

新摘要与旧组ID既不要求相等，也不要求不同，没有试算法去凑旧值。原始RGB只在本次取得时计算摘要，不另存1024数组；缓存核验核保存记录及完整性，不从256反推原始摘要，也不重读原图来验证其当前像素。

固定0002在新来源审计中的记录如下，仅引用保存的元信息，没有为报告再次解码：

- 原文件hash：`8324bd7b6562bb635647e592977525f98452903c9dd76b6c5288cf5a8c28d7c1`。
- 旧组ID：`14de204cf65295d225e29c695d8e07128ebd81c523b38bc2d4335d208beb760d`。
- 新原始RGB摘要：`7c04556d02b197e4ea95e2f66b8b57f4ca93539756cf6f88cb2f8ec97e3d8753`。
- 新规范工作摘要：`4809e3fb10e753f670519abdf32da5f84abc890fbb54d0aebb5fb9166556e22c`。

新sources绑定准备complete／8项输出、封存三源码、固定来源元信息、设计、补充方案、停止报告、旧失败JSON／实现、独立新实现、解释器及依赖身份。总执行计划用于动态进度同步，其整份hash不作运行身份。

原generation.py与fit_inputs_v1失败现场保留。交付前19项既有源码／设计／诊断／准备与失败JSON的hash和旧目录清单／stat均与执行前一致；旧9份几何NPZ只核stat，未打开其字节或数组，不把该检查称为重新证明NPZ内容。旧目录仍为13文件、无complete，images／residuals为空。

## 二、实际来源、覆盖与计数

来源与生成沿用原D_fit60、原顺序／字典、9几何、亮暗及7幅度。未重抽样、换背景、改位置／幅度／seed或降低门槛。旧N80与N_gen80只用于来源和规范工作hash核对，没有展示、外观筛选或模型评分。

| 项目 | 本次实际结果 |
| --- | ---: |
| 旧训练来源路径 | 160，各读取原字节并解码1次 |
| D_fit来源背景 | 60，各读取原字节并解码1次 |
| 本次来源打开／解码 | 220次／220条路径 |
| 全部背景×几何条件 | 540/540可定义 |
| 至少1/9几何可定义背景 | 60/60，门槛40 |
| 9/9、部分、全不可定义背景 | 60／0／0 |
| 各预定几何可定义背景 | 九几何均60/60 |
| D_fit唯一工作内容 | 60 |
| 旧训练唯一工作内容 | 160，无内部精确重复 |
| D_fit内部／与旧训练精确工作碰撞 | 0 |
| 正式render调用 | 7560，每固定请求1次 |
| 基线＋请求引用 | 60＋7560＝7620 |
| 包含基线的唯一模型输入 | 7305 |
| 仅请求的唯一RGB内容 | 7250 |

全60来源和540条件先完成，coverage记录的门槛前render为0；随后才生成。Q200／F100／D_hold60未打开，其跨角色工作内容碰撞仍待对应阶段核验，不能说全420背景已完成工作域去重。

旧尝试在0002解码1次即停。本次新版本重新取得220条路径，因此跨两次尝试累计**221次解码事件、220条唯一源路径**；0002累计2次，其余来源各1次。累计render为旧0＋新7560，不把缓存NPY／NPZ读取算作原图解码。新目录没有续接或补齐旧失败事务。

## 三、全部请求的实际数字变化

| 总体项目 | 实际数量／分母 |
| --- | ---: |
| 有结果且保留的名义请求 | 7560/7560 |
| 对比度与SEMU代理可定义 | 7560/7560 |
| 未定义 | 0/7560 |
| 非零实际变化 | 7245/7560 |
| 零实际变化 | 315/7560 |
| 存在通道裁切的请求 | 503/7560 |
| 非零但旧诊断M_eff为空 | 2070/7560 |
| 诊断M_eff非空 | 5175/7560 |

零变化请求照常保留并指向相同B内容，不救样本，也不能把未来这类请求的未报警称为异常漏检。2070个弱非零变化仍保留float输入和D，没有被旧1/255诊断门槛删去。裁切计数与零变化、弱变化允许交叉，不能将各列相加当互斥样本数。

全部幅度如下；每行1080请求，均可定义。这里的幅度是请求数字幅度，不能直接叫实际局部对比度。

| 请求幅度 | 请求 | 可定义 | 非零 | 零变化 | 有裁切 | 弱非零且M_eff空 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 1080 | 1080 | 1035 | 45 | 45 | 1035 |
| 0.002 | 1080 | 1080 | 1035 | 45 | 45 | 1035 |
| 0.004 | 1080 | 1080 | 1035 | 45 | 45 | 0 |
| 0.008 | 1080 | 1080 | 1035 | 45 | 45 | 0 |
| 0.016 | 1080 | 1080 | 1035 | 45 | 45 | 0 |
| 0.032 | 1080 | 1080 | 1035 | 45 | 122 | 0 |
| 0.064 | 1080 | 1080 | 1035 | 45 | 156 | 0 |

全部九几何如下，实际栅格面积仍按冻结几何元信息记录，未改成严格等面积。

| 几何 | 可定义背景 | 请求 | 非零 | 零变化 | 有裁切 | 弱非零且M_eff空 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r04_circle | 60 | 840 | 805 | 35 | 55 | 230 |
| r04_ellipse_d | 60 | 840 | 805 | 35 | 55 | 230 |
| r04_ellipse_h | 60 | 840 | 805 | 35 | 55 | 230 |
| r08_circle | 60 | 840 | 805 | 35 | 55 | 230 |
| r08_ellipse_d | 60 | 840 | 805 | 35 | 57 | 230 |
| r08_ellipse_h | 60 | 840 | 805 | 35 | 56 | 230 |
| r12_circle | 60 | 840 | 805 | 35 | 56 | 230 |
| r12_ellipse_d | 60 | 840 | 805 | 35 | 57 | 230 |
| r12_ellipse_h | 60 | 840 | 805 | 35 | 57 | 230 |

| 方向 | 请求 | 非零 | 零变化 | 有裁切 | 弱非零且M_eff空 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 亮 | 3780 | 3465 | 315 | 341 | 990 |
| 暗 | 3780 | 3780 | 0 | 162 | 1080 |

## 四、全量输入核验与独立只读证据

全部float输入、固定M_geo／alpha、D=max_RGB(abs(I−B))及诊断M_eff已逐项核shape、dtype、有限性、摘要和实际数值。保存输入未经PNG量化；区域外变化为0。核验没有调用render，没有另造或替换完整I。

ROI独立核验要求未裁切通道等于冻结float32加和值，裁切通道等于相应边界；裁切比例从ROI越界条件独立核对。Lb／std及实际DeltaY、配对对比度、全部k=0.4／0.2／0.8的代理测量也从保存B／I核对。6022个满足条件的相邻幅度配对用于未裁切能量比检查，最大相对误差0.00005489602433561558，小于固定容差0.0005；未对裁切配对强求幅度平方关系。

两项冗余检查均已实际执行：

1. 保存的original_decodes=160＋60=220；原图打开记录恰为白名单220路径、各1次，禁止导入／访问记录为空，全部禁止操作计数为0。取得阶段render0与最终render7560分别核对。
2. unique_inputs顶层references=7620，与逐输入引用数之和、全部60基线ID和7560冻结请求ID一致，每引用恰1次、逐内容引用顺序一致。唯一数组实际7305，未强制做成7620份不同图。

成功清单含14883文件：8份JSON、9份几何NPZ、7560份残差NPZ、7305份唯一RGB NPY及1份complete。期望目录由固定几何／请求和唯一内容索引推导，未收编未知文件；全文件hash和ROI／测量语义均核验，未取成功子集。

独立进程的已实现verify-fit从完整缓存核验，原图打开／解码、render、前向、参数更新、校准和拟合均新增0。其当前0计数与原生成220／7560分开保存；以下为本次独立核验的最终证据：

```json
{
  "status": "FIT_INPUTS_IDENTITY_V2_CACHE_VERIFIED",
  "complete_sha256": "3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593",
  "frozen_files_verified": 14882,
  "independent_verification_elapsed_seconds": 258.50172980000025,
  "cumulative_generation_evidence": {
    "historical_original_decodes": 1,
    "new_original_decodes": 220,
    "cumulative_original_decode_events": 221,
    "cumulative_unique_source_paths": 220,
    "historical_render_calls": 0,
    "new_render_calls": 7560,
    "cumulative_render_calls": 7560,
    "source_paths_are_not_proof_of_physical_independence": true
  },
  "current_invocation_boundary": {
    "forbidden_modules_loaded": [],
    "blocked_import_attempts": [],
    "blocked_payload_attempts": [],
    "original_decodes": 0,
    "old_training_decodes": 0,
    "fit_decodes": 0,
    "render_calls": 0,
    "original_paths_opened": 0,
    "original_file_opens": 0,
    "Q200_F100_D_hold60_decodes": 0,
    "GT_or_probability_reads": 0,
    "model_forwards": 0,
    "parameter_updates": 0,
    "calibration": 0,
    "statistical_fits": 0,
    "old_training_usage": "source and canonical working hash only"
  }
}
```

完整数据依据：[来源审计](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/source_audit.json)、[覆盖](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/coverage.json)、[请求与全部分组分母](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/manifest.json)、[唯一输入与引用](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/unique_inputs.json)、[生成期全量核验](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/verification.json)、[完成清单](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/complete.json)、[来源与实现绑定](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_identity_v2/sources.json)。

## 五、实际环境、资源与源码审阅

使用现有私人CPU环境：Python3.12.7、NumPy1.26.4、SciPy1.16.2、Pillow12.3.0，未安装／下载。实际解释器为`D:/华兴源创/实验/弱缺陷Demo/.venv/Scripts/python.exe`。没有导入Torch／模型、加载权重或pickle；没有检测器评分、训练、校准或统计拟合。

生成事务记录到全量语义核验完成为344.633139秒（约5.74分钟）；该计时不包含之后全文件hash封存及另起进程的D核验。启动可用RAM9.833782GiB／D盘空闲119.118729GiB，最小采样可用RAM8.919014GiB，生成进程峰值工作集156.648MiB。输出实际约5.376081GiB；封存后读取的D盘空闲约113.686653GiB。这些是当前CPU输入工作实测，不是GPU评分速度估计；系统RAM余量与本进程峰值工作集含义不同。

独立新源码SHA256：`17ac328a351aebe6053bdce02a9525147f4346426e875a7a6f35e423bb3e3d2a`。complete SHA256：`3db6a4c13ae5eee423149acc96b37ce27e7917cb9e346508c0a10a36ac76a593`。

按executing-plans完成一次独立整段源码审阅，五个预定失败模式均覆盖；发现1项Important、无Critical或Minor。已在首次实际读图前修复：失败处理原来调用会再抛异常的boundary，可能遮蔽首次禁止访问并漏存failure；新实现将成功路径的严格断言与失败路径的非抛错边界捕获分开。只用隔离命名空间中的虚构标量名单确认非严格捕获可序列化、严格成功仍拒绝异常，没有实际尝试导入模型或打开受保护材料；随后静态inspect通过，再开始本次取得。未改旧generation.py。

实施与判读取舍：项目无Git仓库，依用户要求只用独立文件／目录及科研验收，不建立worktree、过程ledger、软件测试套件或提交流程。独立审阅未判断的实际覆盖、残差和资源已由本次B—D补证；检测器兼容、预测价值、未来角色碰撞和物理独立性仍未知。程序守卫约束既定执行路径，沿用原诊断的解释边界，不当任意恶意代码安全沙箱；没有延期Minor项。

## 六、已实现运行说明与下一项边界

本次已实际执行一次取得和生成。以下是已实现入口；完整同身份结果存在时generate-fit也只核验复用，不能用于修复或补齐部分事务。已有未知／失败目录须保留查因，不删除后重跑、不自动再开版本。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.generation_v2 inspect
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.generation_v2 verify-fit
```

首次取得所用的已实现命令为`-m mura_response.generation_v2 generate-fit`；本报告不要求再次运行。verify-fit仅核新完整缓存，失败v1不由此改为通过。

本成果证明本阶段固定数字输入的来源身份、覆盖、实际变化与计数闭合，能够交给下一阶段按既定模型评分；还不能证明检测能力、SEMU代理用途、人眼可见性、外观逼真或真实屏幕收益。60背景及其变体不是7560个独立物理面板，采集分组仍未知。Q／F／D_hold工作去重、正常工作点、模型响应及背景留出预测均未进行。

下一项仅为第4阶段**独立新正常工作点的准备**，须另行授权，不自动读Q／F图像或启动模型评分。唯一可直接发送的下一步指令维护在总计划第六节；训练和CPU统计拟合仍由用户手动启动。

