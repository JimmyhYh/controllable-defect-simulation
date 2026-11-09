# 第2阶段：共同请求与实际合成标签CPU核验

第1阶段完整协议／2655来源／7闭合目录、固定单头训练侧原生缓存与旧03池身份已核对。仅首次独占创建本轮E根及results/inputs_v1，transaction.json排他创建、fsync及读回通过；未认领已有未知目录或改动D盘旧成果。全部结果都是元信息、请求及核验证据，原图／缓存未复制、量化或重写。

实际1890份R5/N1885原生缓存完整核hash／二值GT／合法窗；固定2500份旧NPZ核float32[0,1]、已有翻转／有效域、原背景实际D及M_eff、ROI／域外严格零变化、原接受spec几何与覆盖≥0.90，四类各625。原接受几何仅重建ROI，无render或新合成。合成直接ImageNet归一化，没有再除255、转uint8、重新补边或二次翻转。

数据seed2026102300冻结一套10000共同请求，三初始化seed仅作未来协议值。A真实5000／正常5000，B真实2500／合成2500／正常5000；共享7500槽使用完全相同输入引用与数组hash，每R分别1000／500次，干预2500槽来源差异明确。实际曝光及像素阳性／有效域总数见summary.json，不能称像素级正负比例匹配或合成单因素因果。

固定25项兼容输入仅元信息，原21训练窗＋四类各首个旧合成；实际有效1460224像素／阳性26759。没有前向、训练或可学习性验收。

本次新请求10000行，不是模型更新。CPU保存段190.250秒；启动RAM7.000GiB／E空闲787.099GiB，仅快照，不是连续峰值。数据解码事件{'native_npz_decode_operations': 9202, 'synthetic_npz_decode_operations': 2500}，LRU最多64份原生缓存；1890是不同源图数，重复CPU读取不增加独立样本。未读取原RGB／原GT／Q/E/U或概率，模型／tensor／训练／校准／指标／渲染均0，旧阈值／权重／源码／环境不变。

## 已实现的只读核验入口

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.inputs verify
```

verify只核完整同身份结果／来源，默认不解码缓存、不写入。verify-decode是显式完整CPU复核，无模型或生成；prepare已有完整结果只走默认verify。未知／部分事务保留停止，不自动补样／重试。training_pack仅CPU取样包装，不是训练入口；training/scoring/metrics仍未实现。下一项仅另行授权第3阶段准备，所有训练继续由用户手动启动。
