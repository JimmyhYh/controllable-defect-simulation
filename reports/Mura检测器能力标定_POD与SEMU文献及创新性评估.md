# Mura检测器能力标定：POD与SEMU文献及创新性评估

日期：2026年10月28日。对象：拟开展的“检测器能力标定：以POD与SEMU为刻度，用于面板Mura，并给出补样本判据”。私人设备、公开材料、个人学习与非商业研究；公司内部材料隔离。

**判断：有研究价值，但原题目把已有方法组合与尚未验证的决策能力放在了一起，创新性论证不足。合成Mura测检测极限、真实背景叠加模拟缺陷做POD、SEMU量化Mura，以及按检测不确定性选择训练样本，都已有先例。更值得聚焦的是：SEMU代理刻度对机器检出难度的预测是否成立、适用条件是什么，以及受控探针能否预测真实Mura的漏检风险。当前证据尚未回答这两个问题，不能据此宣称已经形成新标定方法或补训练样本判据。**

本文件是选题评估，不是新实验协议，不授权训练、前向、重新生成、重校准或读取新评价数据。旧模型、阈值、缓存、报告及执行计划保持原状。项目现有探针结论以[核查修订报告](D:/华兴源创/报告/受控探针检测器能力标定核查修订报告.md)为准。

## 一、检索范围与判断强度

本轮围绕三条文献链检索，并核对下列12篇研究及官方资料：

- Mura检测／感知量化：`Mura SEMU detectability`、`Mura contrast sensitivity`、`Mura quantitative evaluation`、`Mura detection probability`及中文对应词。
- 检测能力评价：`synthetic defects probability of detection`、`real background superimposition POD`、`automatic visual inspection POD`、`model assisted POD sensitivity`。
- 样本决策：`defect detection active learning`、`POD sequential sampling`、`synthetic defects training evaluation`。

专用CrossRef／Scopus等学术检索连接器在本会话未提供；采用网页检索定位，再回到出版社、CVF、arXiv、大学／研究机构仓储、NASA和SEMI官方页面核验。本轮不是可证明穷尽的系统综述，也不构成“全球首次”检索。没有把搜索摘要的相对日期当发表年份。

以下逐篇标出阅读深度。能取得相关方法全文的，定位到章节／页码／图表；只能取得官方摘要或预览的，明确限制，不推断其完整实验没有做过某件事。未获得SEMI付费标准全文，不能宣称本项目符合现行标准。

## 二、最相关研究及与本项目的关系

### A. Mura检测极限与感知刻度

**[1] Chen, L.-C. & Kuo, C.-C. (2008). Automatic TFT-LCD mura defect inspection using discrete cosine transform-based background filtering and ‘just noticeable difference’ quantification strategies. Measurement Science and Technology, 19(1), 015507.** DOI：[10.1088/0957-0233/19/1/015507](https://doi.org/10.1088/0957-0233/19/1/015507)。[大学记录](https://scholars.lib.ntu.edu.tw/entities/publication/4cc30adb-cca0-4c2c-b6f7-a5409e8a49a9)、[工研院托管原文](https://aoiea.itri.org.tw/TC/includes/GetFile.ashx?chk=b9928491-7d72-411b-bcdf-164cde807498&id=108&mid=38&name=FileName)。

已读相关方法和实验全文。**最直接的重合在第4节、印刷页9、Fig.15／Table 2**：在LCD图像上加入灰度差10至1的人工圆形Mura，列出SEMU与检测成败，考察最低可检灰度差。因此，“逐渐减弱人工Mura并测检测极限”不能作为本项目首创。该实验与拟开展的跨背景、跨形态机器响应预测仍有范围区别，但区别须以新增实证成立。

**[2] Taniguchi, K., Ueta, K., Onishi, H. & Tatsumi, S. (2006). A MURA Detection Method Considering Human Vision Perception. IEEJ Transactions on Industry Applications, 126(11), 1539–1548.** DOI：[10.1541/ieejias.126.1539](https://doi.org/10.1541/ieejias.126.1539)。[官方文章](https://www.jstage.jst.go.jp/article/ieejias/126/11/126_11_1539/_article)、[原文PDF](https://www.jstage.jst.go.jp/article/ieejias/126/11/126_11_1539/_pdf/-char/ja)。

已核官方摘要及相关正文。摘要与检测结果部分讨论Mura检测所需对比度、分辨率及空间频率，并以显示器组件为例。因此，“检测灵敏度与尺度／空间频率有关”也是既有问题，不能把项目中一次patchsize或分辨率对照直接包装成新机制。

**[3] Su, W.-T. & Hwang, S.-L. (2011). Measurement of multiple JNDs for developing Mura ranking standards in LCD. Displays, 32(5), 274–283.** DOI：[10.1016/j.displa.2011.05.006](https://doi.org/10.1016/j.displa.2011.05.006)。[出版社摘要与预览](https://www.sciencedirect.com/science/article/pii/S0141938211000503)。

已读官方摘要、引言及结果预览，未取得完整付费正文。研究涉及25位受试者、六类Mura，发现多JND关系非线性且形态间敏感性不同。它针对人眼，而非机器POD；支持审慎检验刻度，却不直接证明SEMU对某个检测器无效。

**[4] Park, J. H. et al. (2024). Human visual system-based perceptual Mura index for quantitative Mura evaluation. Measurement, 227, 114289.** DOI：[10.1016/j.measurement.2024.114289](https://doi.org/10.1016/j.measurement.2024.114289)。[出版社摘要与预览](https://www.sciencedirect.com/science/article/pii/S0263224124001738)。

已读官方摘要、Highlights和引言预览，未取得全文。HPMI考虑背景灰度、尺寸／长轴及Mura类型，目标是更贴近人眼主观严重度。**“对比度／面积不够，还需背景和形态”不是新的通用发现。** 本项目可能研究的差异在于这些刻度对冻结机器检测器的预测效度，而非再提出一个未经人眼验证的“感知指标”。

**[5] Qiu, H., Huang, J., Feng, Y.-C. & Rong, P. (2024). Detection method of organic light-emitting diodes based on small sample deep learning. PLOS ONE, 19, e0297642.** DOI：[10.1371/journal.pone.0297642](https://doi.org/10.1371/journal.pone.0297642)。[官方全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297642)。

已读相关方法全文。定位“Quantitative evaluation”部分、Eqs.3–9及Fig.10：检测器提取候选后，用局部背景、SEMU／SEMUMaxMin定量并过滤。它说明“深度检测＋SEMU”已有直接先例；其量化后处理与“预先测量机器对受控刺激的响应曲线”目的不同。本文工业结果不与本项目数据或指标横向相减。

**[6] Liu, W., Liu, C., Liu, Q. & Yu, D. (2023). Assigned MURA Defect Generation Based on Diffusion Model. CVPR Workshops, 4395–4402.** [CVF官方论文页](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Liu_Assigned_MURA_Defect_Generation_Based_on_Diffusion_Model_CVPRW_2023_paper.html)、[已有本地S07原文](D:/华兴源创/文献/S07_Assigned_MURA_Diffusion_CVPRW2023.pdf)。

复核本地8页全文，重点§§3.3–3.4、§§4.1–4.3、Tables 1–2。位置条件、背景特征保留及Mura可见性已在其研究范围。摘要出现“visibility”，但相关方法与实验主要是条件生成、FID和下游检测；未见其建立本项目拟讨论的SEMU条件POD验证链。因此既不能说“首次控制可见性”，也不能仅凭摘要认定机器POD问题已被它完全解决。

### B. 受控模拟、真实背景与POD

**[7] Yosifov, M. et al. (2025). Superimposing Synthetic Defects into Real XCT Data and Segmentation-Based Comparison for Advanced Probability of Detection Evaluation. Journal of Nondestructive Evaluation, 44, 119.** DOI：[10.1007/s10921-025-01262-1](https://doi.org/10.1007/s10921-025-01262-1)。[官方全文](https://link.springer.com/article/10.1007/s10921-025-01262-1)。

已读相关全文，重点§§3.2–3.5和§5。将可控大小／形状的模拟孔隙叠加真实XCT背景，用3D U-Net分割并构造POD；§3.5以Jaccard阈值定义命中。**“真实背景＋模拟缺陷＋POD”已有直接先例。** 其物理成像模型与二维RGB外观扰动不同；搬到Mura不自动产生方法创新。a90/95的定义应另按官方统计依据核对，不照抄个别论文中含混的“下界”措辞。

**[8] Le Gratiet, L. et al. (2017). Model Assisted Probability of Detection Curves: New Statistical Tools and Progressive Methodology. Journal of Nondestructive Evaluation, 36, 8.** DOI：[10.1007/s10921-016-0387-z](https://doi.org/10.1007/s10921-016-0387-z)。[出版社记录](https://link.springer.com/article/10.1007/s10921-016-0387-z)、[2016年作者预印本全文](https://arxiv.org/html/1601.05914v1)。

已读作者全文相关方法，重点§§3.5、4、5。通过代理模型处理POD估计与不确定性，并研究多输入对整条POD曲线的敏感性。因此增加probit拟合、bootstrap或多因素敏感性本身不构成新统计方法；这些工具应服务于新的Mura问题。正式卷年为2017，不能把2016预印本另算一篇独立研究。

**[9] Kompanets, A., Leonetti, D., Duits, R., Maljaars, J. & Snijder, H. H. (2023). Probability of detection curve for the automatic visual inspection of steel bridges. ce/papers, 6(3–4), 814–824.** DOI：[10.1002/cepa.2585](https://doi.org/10.1002/cepa.2585)。[作者大学仓储全文](https://pure.tue.nl/ws/portalfiles/portal/307704923/ce_papers_-_2023_-_Kompanets.pdf)。

已读相关全文，重点印刷页819–821、Figs.4–7及Table 1。该文为视觉分割模型建立POD，明确讨论把误报像素计入或排除响应会改变曲线。因此，“报警不等于局部检出”“误报会抬高表观能力”不是本项目首创；本项目应把这些既有规范落实到Mura。其POD样本还包含部分训练图，不能直接照搬为独立验证设计。

**[10] Kim, F. H., Pintar, A. L., Scott, J. H. J. & Garboczi, E. J. (2023). Evaluation of Flaw Detection Algorithm Using Simulated X-Ray Computed Tomography of Ground Truth Data. Journal of Nondestructive Evaluation, Diagnostics and Prognostics of Engineering Systems, 6(4), 041005.** DOI：[10.1115/1.4063170](https://doi.org/10.1115/1.4063170)。[NIST原文](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=935170)。

已读原文相关方法及讨论。§2以一致的体素结构生成精确真值，在不同信噪比下比较分割、尺寸误差及二项POD，讨论向实验数据迁移。它提供“已知刺激／真值”及“像素指标与缺陷级能力分开”的先例。题录以PDF首页期刊名为准，未沿用NIST网页中与该DOI不一致的期刊字段。

### C. 补样本与训练收益

**[11] Lv, X., Duan, F., Jiang, J.-J., Fu, X. & Gan, L. (2020). Deep Active Learning for Surface Defect Detection. Sensors, 20(6), 1650.** DOI：[10.3390/s20061650](https://doi.org/10.3390/s20061650)。[出版社页面](https://www.mdpi.com/1424-8220/20/6/1650)、[开放全文仓储](https://pmc.ncbi.nlm.nih.gov/articles/PMC7146140/)。

本轮核官方摘要与开放仓储可检索图表；直接全文访问受限。研究用不确定性采样及类别采样分配选择待标注样本，并与随机等策略比较。因此“根据模型弱点决定补哪些数据”已有成熟相邻研究。若本项目提出新补训练样本规则，须证明比等预算随机补充／不确定性补充更有效，不能只给经验阈值。

**[12] Jeong, E.-Y. et al. (2021). A more reliable defect detection and performance improvement method for panel inspection based on artificial intelligence. Journal of Information Display, 22, 127–136.** DOI：[10.1080/15980316.2021.1876174](https://doi.org/10.1080/15980316.2021.1876174)。[官方全文](https://link.springer.com/article/10.1080/15980316.2021.1876174)。

已读相关方法与实验，重点§3.2及训练数据扩充实验。其面板外观缺陷工作用条件配对生成扩充训练，再评价真实缺陷，说明训练价值需要由训练干预验证。它并非本项目的低对比度Mura能力标定证据；其结果不能外推到现有公开数据或当前模拟器。

## 三、官方依据对题目边界的约束

**SEMU与现行DSEMU不能混用。** [SEMI D31官方目录](https://store-us.semi.org/products/d03100-semi-d31-guide-for-definition-of-measurement-index-dsemu-for-luminance-mura-in-fpd-image-quality-inspection)当前列出D31-0825，标题为DSEMU；公开摘要含长轴、对比度、距离因素及适用显示／观看条件。本项目旧面积公式来自历史SEMU口径，不能称已实现现行D31。只凭未标定RGB与约定mm/px，也不能赋予“SEMU=1等于这批图的人眼阈值”的绝对语义。若继续在公开RGB上研究，宜写“历史SEMU公式的数字代理”，并将配对数字对比度、像素面积、局部背景等原始量并列保留。

**置信界与误报处理是基本有效性要求。** [NASA/TM–2014-218183](https://ntrs.nasa.gov/citations/20140005337)讨论POI／POD及误报影响；其印刷页56–57解释保守检出界。POD下置信曲线达到0.90的位置对应所需刺激的上侧界，不能把a90分布低端当保守要求。修正实现错误是必要工作，不能单独称新方法。

**“补评价样本”已有序贯设计基础。** [NASA DOEPOD数据手册，NASA/TM–2015-218770](https://ntrs.nasa.gov/citations/20150013943)把POD能力验证与序贯取样／停止决策联系起来。它支持讨论何时还需评价样本，但不证明按同样规则补训练样本就能改善模型。

## 四、创新性逐项判断

| 拟作为贡献的内容 | 文献核对后的判断 | 还可能成立的贡献 |
| --- | --- | --- |
| 在真实屏幕图上逐档加入Mura，找检测极限 | [1]有直接早期先例，[2]亦讨论灵敏度要求 | 新的公开、可复用且验证充分的Mura评价资源；不能只换检测器重画图 |
| 合成缺陷＋真实背景＋POD | [7]已有明确组合，[10]已有可控真值／POD框架 | 针对Mura的有效性及迁移证据，而非一般组合首创 |
| 用SEMU评价深度模型检测结果 | [5]已有直接先例，[3–4]系统讨论感知量化 | 检验它对机器响应的预测充分性与适用范围 |
| 将形态、面积、背景加入POD分析 | [2–4]已有领域因素，[8]已有多输入敏感性方法 | 量化现代Mura检测器中特定交互及未见背景上的预测效果 |
| 同时报告正常误报、局部检出与置信界 | [9]与NASA已有依据 | 严谨协议的组成，不单列为方法发明 |
| 以探针响应推断结构瓶颈或补样本无效 | 静态响应不能识别训练干预后的变化 | 另有严格干预才可支持机制；现阶段撤回确定判断 |
| 用响应曲线给出补样本判据 | 评价取样有序贯先例；训练取样有主动学习先例 | 规则预测／决策效用经外部留出和等预算对照成立后，才可能成为新贡献 |

所以，**工程实现价值较明确；原组合的首创性较弱；收窄为测量预测效度问题后存在可研究空间，但仍是待验证的新颖性候选。** “本轮未发现完全同题论文”不等于“无人研究过”。本次也不是专利新颖性／自由实施检索。

## 五、建议保留的一个主问题

> **在预先规定的误报约束与检出／定位定义下，历史SEMU数字代理能否比原始对比度与面积，更好地预测冻结Mura检测器在未用于拟合的背景上的响应；若不能，加入背景、形态或空间频率后能否改善预测，并明确其失效条件？**

建议暂用题目：**《受控数字Mura扰动下的检测器响应表征与SEMU代理刻度适用性研究》**。

这一定位的新增信息应是“哪种刻度对哪类机器、在什么条件下能预测多少”，而不是“用POD画了一条曲线”。研究的是机器响应，不声称重新建立人眼标准，也不将模型softmax分数校准与这里的检出能力表征混为一谈。

还需注意：在固定尺度约定下，历史SEMU是对比度与面积的确定函数，**不会凭空增加这两个量之外的信息**。比较的实质是这种归一化／压缩是否有利于有限样本预测，是否遗漏了影响响应的重要条件。对照应含对比度单轴及对比度＋面积的合理模型，控制模型复杂度并用留出表现判断，不能把对弱对照的优势解释成新物理信息。公开RGB条件下，“能力标定”只宜作为工作称谓，实际成果是有范围的数字响应表征，不是已具备物理溯源的计量校准。

最自然的后续扩展是：**在不利用真实评价结果调参的条件下，数字探针给出的风险排序，能否预测真实Mura漏检，且优于只用对比度／面积等简单规则？** 这是比“模拟看起来合理”更强的外部效度问题。目前本项目尚无这项证据。

### 为什么这个问题有价值

从工程需求看，单个总体AP不能回答“多小、多弱、什么背景下容易漏”。一个有明确适用范围的响应模型，可以帮助判断评价覆盖不足、比较工作点、定位需要进一步核验的条件。它即使不提高检测分数，也可能产生测量价值。

从论文论证看，单独报告“机器不等于人眼”或“背景影响检出”仍然太泛。需要有定量、可重复、面向留出的预测检验，说明SEMU代理是否带来信息、哪些条件需要另建刻度；若提出替代模型，还须证明复杂度增加有实际收益。

从现有条件看，参数化模拟器、保存残差、既有检测器和来源核验可以复用；不需要为了研究响应表征立即训练新的大模型。主要难点在测量定义、样本独立性与真实迁移证据，计算资源只是其中一部分。

## 六、“补样本判据”需要拆成两类

| 目标 | 可以用什么证据支持 | 不能直接推出什么 |
| --- | --- | --- |
| 补标定／评价样本 | 当前档位没有覆盖转折区、区间太宽、某背景／形态覆盖不足；通过预定的取样与停止设计改善估计精度 | 不等于训练这些样本会提高检测器 |
| 补训练样本 | 冻结规则选择补充材料；与等预算随机／不确定性等对照实际训练；在独立评价上核定位、漏检与误报 | 不能由固定模型低POD、score/tau或单次拟合成功证明有效／无效 |

训练规则若主张选择的是“有用样本”，最终被检验的应是**补充后的真实检测收益及代价**。还要控制或报告真实异常重复次数、正常／异常曝光、阳性像素量和背景组成。过去两轮配对结果已表明这些组成会共同变化，不能重新忽略。

对PatchCore尤其要区分：往正常特征库补正常覆盖，与给监督模型补异常训练，是不同干预；不能为不使用异常训练的固定PatchCore推导“补异常图必然无用”的一般结论。

因此建议将补训练样本判据放到后续扩展。第一篇候选研究先证明刻度是否有效；若用户只希望工具和汇报成果，完成响应表征也可合理收束。

## 七、现有进度对论文的支持程度

本段只引用既有[核查修订报告](D:/华兴源创/报告/受控探针检测器能力标定核查修订报告.md)及[成果总览](D:/华兴源创/报告/项目成果总览与成文建议.md)，没有重算指标或读取新图。

**已经拥有的基础**：可控Mura式／线状模拟、实际残差与有效域核验、真实正常背景上的既有探针响应、PatchCore／G1等模型、统计与缓存核查能力，以及两轮真实工业配对训练的完整结果。它们可以支持研究工具、技术报告和新问题的可行性判断。

**尚不能当作论文核心结论的内容**：旧SEMU轴混入全局与局部背景差；模拟与真实侧口径不同；低档背景准入与原误报影响曲线；硬软边配对含强度和方向混杂；旧置信界定义需修正；采集分组未知。核查暴露的问题是项目的内部有效性问题，不能把“修复自己的测量错误”直接写成领域普遍新发现。

当前尚未得到“SEMU代理经过留出预测验证”“数字探针可以预测真实Mura漏检”或“探针指导补训练样本优于基线”的证据。两轮KSDD2合成收益小且不一致，也不能倒推出AnoLCD Mura收益受某个确定的探测极限限制；模型、任务、数据域和干预并不相同。

### 如果继续，论文论点至少应补齐哪些证据

以下是论点所需证据，不是已冻结的执行计划，也不授权自动运行：

1. **一致的测量定义**：明确数字域／亮度域、对比度与支持域、实际幅度与饱和、背景与零注入对照；拒绝用不一致的模拟／真实口径直接对齐。现有RGB条件下保留“代理”名称。
2. **公平且任务一致的响应**：区分整图报警、指定位置检出与分割；模型比较需要预先规定的误报约束，并报告实际误报及其不确定性。新工作点只能在独立新协议中确定，不回改旧tau。
3. **真正的刻度比较**：SEMU-only与对比度／面积、加入背景／形态等候选，在未用于拟合的背景上比较概率预测误差及不确定性，不能用训练拟合优度代替。按背景／可知采集组划分，相关变体不当独立样本。
4. **至少一个外部效度检验**：若论文题目声称“用于真实面板Mura能力”，须有受控数字结果与真实Mura的桥接；历史已看过的数据只能探索性解释。若只能做模拟域，题目和结论相应收窄。
5. **主张训练决策才增加干预**：先冻结补样本规则，再与等预算对照比较，报告全部预定种子与代价。若不做这一步，删除“已验证补训练样本判据”的主张。

a50／a90只在实际覆盖与模型适配允许时报告；曲线未跨越就保留不能估计，不为得到数字而强行外推。固定单模型上的能力条件也不能解释成模型家族的不可学习上限。

## 八、成文判断与建议

| 完成范围 | 更合适的成果定位 | 当前判断 |
| --- | --- | --- |
| 既有模拟、响应及本轮审计整理 | 技术研究报告／方法实施与局限文章 | 现在可以整理；保留全部修订与失败 |
| 修正定义后，仅重画同批背景的POD | 工程扩展／探索性案例 | 有用，但独立论文创新性仍偏弱 |
| 完成刻度的留出预测比较，给出稳定适用条件或可重复失效规律 | 面向Mura的测量／评价研究论文候选 | 值得考虑；新贡献由实际结果决定 |
| 再证明探针风险可预测真实漏检 | 更强的数字探针外部效度研究 | 价值更明确；真实材料与设计是门槛 |
| 再验证补训练样本决策的等预算收益 | 诊断与数据决策闭环论文候选 | 证据要求最高，当前距离最大 |

**推荐：保留方向，但将主线收窄为“数字Mura检测响应与SEMU代理刻度的预测效度”。暂不以“首个POD＋SEMU框架”或“已建立补训练样本判据”为卖点。** 先判断能否获得一致刻度和有意义的留出验证，再决定投入新实验；不能因为实现了拟合就承诺论文，也不因没有SOTA提升就否定测量研究。

若研究只能停留在当前同批模拟图、不能解决测量口径或没有所需外部证据，按技术报告收尾是合理选择。若能量化并复现一个此前未被可靠验证的预测关系／失效条件，且证明它影响工程决策，则有发展成研究论文的空间；本轮文献检索不能替代这部分实证，也不保证投稿结果。
