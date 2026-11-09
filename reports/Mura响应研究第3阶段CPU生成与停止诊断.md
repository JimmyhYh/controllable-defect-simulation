# Mura响应研究第3阶段CPU生成与停止诊断

整理日期：2026年10月29日。研究题目：《受控数字Mura扰动下的检测器响应表征与SEMU代理刻度适用性研究》。

**实际状态：第3阶段已启动，在第一张旧训练背景的RGB身份口径检查处停止，未完成输入生成。D_fit解码0/60，正式扰动生成0/7560；可定义背景覆盖、工作图碰撞、实际残差与去重数量均尚未取得。原失败目录及实现保持原状，没有自动重跑或进入第4阶段。**

## 一、已完成范围与实际分母

本次先在隔离元信息进程执行已实现的`mura_response.resources verify`，返回PREPARATION_CACHE_VERIFIED。第2阶段的8项输出、420角色及15120条请求字典身份闭合；未重算人工数组、读取其对应图像或运行模型。停止后再次进行同入口核验，准备身份仍通过。

独立新增[generation.py](D:/华兴源创/实验/弱缺陷Demo/mura_response/generation.py)，没有修改封存的`__init__.py/resources.py/inputs.py`。生成进程不导入resources；只消费隔离核验后的JSON及inputs纯数组接口。静态语法、固定60背景／7560请求／160旧训练路径和资源核验通过，随后实际启动一次CPU来源取得。

实际运行记录见[failure.json](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_v1/failure.json)：

- 旧N80／N_gen80并集：160条路径；实际只打开并解码首张`display/train/good/0002.png`一次，未完成其规范256工作hash。
- D_fit60：原图解码0/60；540项背景×几何Lb尚未核验，至少40可定义背景的准入尚未判断。
- 正式请求：render调用0/7560。没有B／I／D／M_eff、有效／未定义请求统计或内容去重结果。
- Q200／F100／D_hold60、S_bg／Q_cal、旧测试图、GT和概率均未读取；没有Torch／模型导入、模型前向、训练、校准或统计拟合。

已经保存9份固定几何NPZ及started／sources／protocol／failure四份JSON，共13个文件。九几何是独立于真实背景的CPU定义，不能算作9个完成的刺激。`images`和`residuals`目录为空；不存在complete、coverage、source_audit、manifest或unique_inputs，不能把该目录称为完整输入缓存。

## 二、停止原因及证据链

运行依次完成首张原文件的长度、文件SHA256、尺寸与模式核对，随后在`decode_original`的解码RGB比较处退出。因此该次读入的文件字节与冻结manifest一致。其原文件SHA256为：

`8324bd7b6562bb635647e592977525f98452903c9dd76b6c5288cf5a8c28d7c1`

错误来自新模块的一个未经证实的假设：将旧manifest中的`rgb_content_sha256`直接解释为`sha256(decoded uint8 HWC RGB C字节)`，并要求两者相等。

只读追溯已有实现发现，历史工程早已明确两者不能直接互认：

1. [reference.py](D:/华兴源创/实验/弱缺陷Demo/reference_generation/reference.py:25)注明：旧RGB组摘要的序列化在保存schema中未定义；用原文件字节hash绑定来源，用不可变旧组ID执行排除，另存定义明确的新RGB hash。
2. [preparation.py](D:/华兴源创/实验/弱缺陷Demo/reference_generation/preparation.py:181)保存了同样的RGB组身份约定。
3. [历史v2结果](D:/华兴源创/实验/弱缺陷Demo/results/reference_generation/v2_domain_matched/第1天v2同域背景准备结果.md:15)明确记录“原清单RGB组序列化未知，排除用旧组ID与原文件hash”。

据此，已定位的实现错误是**混用了历史分组标识与新定义的原始RGB字节摘要**。这是助手应在首次读取前核对的接口条件。此次没有证据支持把这个不等号解释为文件损坏、工作图碰撞或背景不适用。

旧摘要的具体历史序列化仍未知；本次失败记录未保存新计算的原始RGB摘要，也没有重新解码该图去猜测历史算法。后续不能通过尝试多种hash、选择恰好相等的口径来消除报错。

## 三、失败现场与身份保护

独立现场：[fit_inputs_v1](D:/华兴源创/实验/弱缺陷Demo/results/mura_response/semu_applicability_v1/fit_inputs_v1)。该目录保持失败及未完成状态，不追加补生成结果、不删除标记、不修改来源绑定来冒充恢复。

已核对实际generation.py与失败事务sources.json绑定一致；其SHA256为：

`1cd617d45c0d84e015ab517ba05a49ff91920b230dc74ae4dec814538134c576`

failure.json SHA256：

`be403ebf7770c207375dad5bb26c5f0abcd73bd1f6961bcde14ba5c65e63919c`

sources.json SHA256：

`0fc4ad18310e349db5f55a562986910f556d0db20b4b7146e813c9b75da505d0`

停止后仅读取失败JSON、目录清单及相关源码／元信息；没有再次打开任何源图。原准备、角色、幅度、位置、种子、几何、旧模型、阈值与历史结果未改。独立资源核验进程只作字节hash和元信息检查，不加载权重或执行pickle。

## 四、代码审阅处理与尚未验证事项

本次按executing-plans完成一次新的只读整段代码审阅，范围为新generation模块、封存接口及设计。首次来源取得前，针对两项Important意见作了一次代码修正：

- 封存文件从“遍历目录后全部收编”改为由固定元信息、9几何、7560请求残差及实际唯一输入索引推导显式文件白名单；未知文件或目录保留停止。
- 缓存核验增加ROI代数核对：未裁切通道必须等于固定sign／amplitude／alpha下的float32加和值，裁切通道必须位于对应边界，裁切比例从ROI越界条件独立核对。不调用render、不另存或替换刺激。

上述代码修正通过语法核验；本次在来源准入即停止，**尚未在完整真实输入上运行或证明全量语义核验通过**。没有因为审阅通过就宣称第3阶段完成。

两项Minor保留为后续独立适配的待补事项：完整缓存入口尚未额外交叉核验保存的总解码／逐路径访问计数与全部禁止操作零值；尚未单独核验unique_inputs顶层冗余references声明（逐引用实际集合和顺序另有核验代码）。本次无完整缓存，这两项不能写成已实际通过。

审阅边界的判定如下：实际碰撞、覆盖、弱变化、完整耗时及资源峰值仍未知；Q／F／D_hold跨角色工作重复按后续阶段核验；检测器兼容及预测用途未检验；物理采集组、近似重复和外观真实性不由精确hash证明；导入／文件守卫用于约束当前程序路径，不作为任意恶意Python代码的安全沙箱。按用户要求未增加软件测试套件、过程表、Git提交或合并流程；本次没有将静态核验表述为完整运行验证。

## 五、实际资源和入口状态

启动快照：可用RAM4.88861GiB、D盘空闲119.49697GiB，满足2GiB／40GiB门槛。本次进入输入事务约0.152秒即停止，未取得完整CPU生成耗时、实际峰值或GPU评分速度，不能据此估算全轮速度。

已实现`mura_response.generation generate-fit / verify-fit`。generate-fit在当前已有失败目录上会拒绝继续；verify-fit也要求完整事务，不能用于把当前目录认证为完成。**当前不提供重跑或续生成命令。**

仍可使用已实现的只读准备核验入口；它不会重新打开原图：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m mura_response.resources verify
```

## 六、下一项最小工作及停止边界

下一步仅制定**独立身份适配与重新取得补充方案**：保持旧组ID用于排除与固定角色，原文件hash继续严格核对；对新解码RGB及规范256工作RGB各保存独立、明确定义的摘要，不把新摘要强制等同未知序列化的旧组ID。

补充方案须保留本失败实现与目录，明确新模块／输出位置、来源绑定、首次旧训练图已解码一次的历史事实和新运行计数边界；复用同一420角色及D_fit60／7560字典，不换背景／种子／位置／幅度，不降低碰撞、覆盖或事务门槛。只写方案时不重读任何图像、不开发或重跑，不自动进入第4阶段。未核验的新入口不提供命令。

本次实际结论仅为：元信息准备身份有效，第3阶段首次取得因可定位的hash接口假设错误停止。尚无数字Mura输入有效性、检测响应或SEMU预测价值的新结果。
