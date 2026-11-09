# 第6天：独立E1004真实评分

当前状态：本轮一周任务已完成，图件及[独立研究报告](D:/华兴源创/报告/少样本真实工业缺陷合成训练研究报告.md)齐备。平均AP增益为正但不稳定，全部冻结成果及历史失败保留；下一项仅审阅独立曝光匹配提案，以一周计划第六节为准。

## 已实现范围

`real_evaluation.py`提供`prepare`、`verify-preparation`、`run-model --model`和`verify-model --model`，不接入或修改旧CLI。`real_evaluation_cache.py`提供未来CPU指标阶段的`resolve_evaluation_caches()`和`verify-all`：必须六组完整，返回原始缓存和阈值来源引用；没有GT读取器或指标算法，不调用模型。当前六组E概率缓存完整；六缓存解析接口已在CPU只读边界下逐张核验通过，未计算检测指标或读取GT。

独立准备目录为`results/fewshot_synthetic/ksdd2_v1/real_evaluation_preparation_v1/`，严格只有protocol.json、source_manifest.json、E_requests.json、CPU_validation.json、complete.json。实际评分目录为`results/fewshot_synthetic/ksdd2_v1/real_evaluation_v1/`，准备不创建。两份新源码、冻结assembly及原六份正常校准／最终10000步权重身份均被绑定；旧源码／配置／权重／结果不改。

准备只读原六组校准缓存的全文件hash、NPY头及scores／summary元信息，其中包括保存的最大值、阈值及FP／TN标量；这属于本次允许的元信息核验，不能写成“未读取任何Q分数”。不解析Q的NPY概率数组载荷，不重新计算分位数。为此没有调用会重新解析Q概率载荷的旧assembly resolver；以冻结assembly的三文件封存和原六份来源清单逐项匹配。实际原Q阈值仅保留在各自原summary，本次配置与E缓存保存来源路径／hash，不复制成新的可编辑阈值。

E仅从固定split和已封存inventory提取元信息，保存原1004图顺序、ID、RGB路径／hash、尺寸和窗口起点；剥离标签和GT路径。准备阶段没有读取E图像字节；A03实际运行现已按冻结源码逐图核对来源hash／模式并评分，助手本次只核验保存缓存，未重新打开原图。上述为评分准备时的范围；当前独立CPU指标已核实本地GT110异常／894正常及阳性486461，见顶部实际结果。评分器本身仍不读取GT或按标签改变输入。

## 六组评分已完成：不再启动模型

[A03及其余五组最终核验](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/real_evaluation_verifications_v1/all_six_v1/第6天六组真实评分缓存最终核验.md)均通过，原六模型顺序不变。用户此前手动启动的串行批量命令保持每模型独立Python进程，全部完成；当前没有下一组评分任务。实际评分缓存／权重／阈值不改，不因准备指标再次运行模型。

需要只读复查全部六组时，已有入口如下；这只是复查缓存，不是下一项指标命令，也不会读取GT或执行模型：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.real_evaluation_cache verify-all
```

准备／只读准备核验入口（不运行模型）：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.real_evaluation prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m fewshot_synthetic.real_evaluation verify-preparation
```

`verify-preparation`仅CPU元信息／hash／NPY头／虚构标量核验，不解析E概率。未来`verify-model`会读取已保存的指定完整E概率核对范围与最大值，只能在该模型实际评分完成后使用；不会隐式调用run-model。

## 冻结评分口径与分母

各模型只加载自己的最终10000步权重；作者3→2／base16结构、GN8、FP32、batch1、workers0、无AMP，eval＋inference_mode，空buffer。加载及评分后检查参数hash与最终训练记录一致，不实例化优化器，不调用任何参数更新。

原图L复制RGB，float32除以255，不拉伸对比度、缩放或增强。256窗、stride128、尾窗H−256、x=0，右侧reflect补齐256；纵向按预定起点顺序评分，重叠异常softmax等权平均，剔除补边后保存原生H×W float32概率及其最大值。不平滑／拉伸；不读取GT或根据标签改变输入、窗口或评分规则。

每模型1004图、4270窗、146411528原生像素；六模型6024次整图评分、25620窗，但只有同一批1004张唯一真实图。每模型概率净载荷585646112字节，六模型3513876672字节（约3.273GiB），另加NPY头与必要元信息。不能将窗口／种子／模型评分次数当独立真实图数。

阈值沿用对应原Q200的q0.95／linear／严格大于，原阈值只作路径＋hash引用。评分不计算TP／FN／FP／TN、AUROC或AP；这些必须在六组缓存齐备后，另行授权的CPU阶段读取GT计算，不能把Q同池误报称独立验证或真实训练收益。

## 资源、时间与停止规则

CPU准备至少2GiB可用RAM、80GiB空闲磁盘；实际评分在Torch导入前及模型创建前要求RAM≥4.5GiB、磁盘≥80GiB，空闲显存≥6GiB；每图边界RAM≥3GiB且磁盘≥80GiB。实际运行重新检查，不能用准备快照代替运行准入；不自动降低精度／窗口／门槛或改用CPU推理。

六组评分段已实测74.938—84.500秒，合计471.814秒，不含来源hash／模型加载／人工等待，不能当整批墙钟时间或未来耗时保证。每模型需完成4270个256窗口。记录的elapsed_scoring_seconds仅包括逐图评分和保存，不含来源hash／模型加载／人工等待；图像边界最小RAM是采样值，不是连续峰值。Torch显存峰值仅本模型评分会话，不能当整机总占用。

单次人工命令最多运行一模型。每模型独占目录、独立token，解码前／每个窗口前向前落盘cursor，逐图保存概率；完整后保存scores、summary、complete。缓存身份完整时，同一命令在Torch导入之前仅核验复用。未知文件、缺失seal、active／failure、临时文件、部分目录或不明确前向均保留停止；没有断点续推／自动重做功能。Ctrl+C也不自动补推理，不能因“尚未保存图”推断之前没有前向，更不能删除标记再跑。

处理器／CUDA、磁盘或来源错误须先查因，不能取六模型成功交集缩小分母。正常校准原失败事务保持失败；E新目录也不会冒充修复原事务。公司材料隔离，所有训练始终由用户手动启动，本模块没有训练入口。

## 本次准备完成证据与边界

prepare及独立verify-preparation已实际通过；完整准备再次prepare在禁止全部写入及np.load的CPU边界下仅核验复用，五文件字节身份不变。资源快照RAM7.097214GiB／磁盘206.218906GiB，prepare段26.828秒；不是E实际评分时间。准备封存complete SHA256为`2bc7462ca53f48e16a6243e8488776f5033b7f7709696bf3930b93684492ef03`；该准备时刻未创建E运行目录；当前已存在全部六组完整结果。CPU准备在禁止np.load、模型运行和解码调用的进程边界下完成，原图GT、Q／E数组载荷加载及模型操作均0；继承检查只对arange(200)／ones(200)虚构标量使用quantile，不重算Q阈值。保存Q最大值／阈值／计数属于获准元信息；此解释未扩大真实数据访问范围。

一次独立静态审阅覆盖两新模块、说明和必要冻结接口，Critical0／Important0／Minor0；未执行CUDA、模型加载、真实评分、概率载荷验证或异常注入／恢复。本轮按用户要求只保留科学配置与最终运行说明，不建立ledger、软件测试或提交流程。上述为准备时的证据边界。当前六组已实际运行并核验，GPU／审计钩子有六次完整运行证据；其他资源条件和中断恢复未被证明，run-model完整评分缓存复用分支尚未单独实测。

## 六组实际完成与核验

[最终核验](D:/华兴源创/实验/弱缺陷Demo/results/fewshot_synthetic/ksdd2_v1/real_evaluation_verifications_v1/all_six_v1/第6天六组真实评分缓存最终核验.md)绑定六份原序评分缓存、最终10000权重hash与原Q阈值来源，每模型1004图／4270窗／146411528原生像素；共6024次模型整图评分，只有1004唯一真实图。保存参数前后一致／buffer0。评分段依序82.172／79.235／74.938／75.000／84.500／75.969秒；全组图像边界最小RAM7.230610GiB。

助手只核保存概率与身份，无新前向、Torch、GT或实际检测指标；实际GT正常／异常和阳性像素仍待核验，不能宣称合成收益。六组齐备，当前不再提供模型运行命令；独立CPU指标入口与配置现已在real_metrics.py实现并核验，命令见[CPU指标运行说明](D:/华兴源创/实验/弱缺陷Demo/fewshot_synthetic/README_real_metrics.md)；六组CPU真实指标现已完整计算并核验，当前只整理第7天图件。原A03核验及旧失败状态保留，六组严格评分目录不追加文件。
