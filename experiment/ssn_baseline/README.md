# 严格R5 SuperSimpleNet独立分支

目前只实现第1阶段来源／CPU资源准备，未实现训练或评分入口。不导入作者模型，不执行pickle、读取原图／GT／数组或加载参数。作者固定版本仅作未修改的来源子集；禁止运行vendor中的train.py／数据模块。

在项目目录使用现有解释器（无需安装）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m ssn_baseline.resources verify
```

已实现的独立入口：

- `acquire`：只取得固定commit的12份作者文本和4份来源许可／声明文本，保存逐文件SHA256及作者Git blob身份；不取权重／数据。已完整目录只读复用；部分目录保存停止，不自动重下载、换版本或覆盖。
- `verify-vendor`：只读核对上述来源、严格目录、hash和源码语法；不执行作者代码。
- `prepare`：核对旧元信息／旧核心源码、已有V1字节hash，读取现有包元信息及RAM／GPU／磁盘快照，独立保存resource_preparation；不安装或创建环境。完整缓存只读复用。
- `verify`：只读核对资源协议、来源与旧身份及本地V1字节hash。不联网、不刷新快照、不计算指标或启动模型。

结果：`results/ssn_baseline/ksdd2_r5_v1/resource_preparation/`。vendor：`vendor/SuperSimpleNet_98ab4d5/`。所有完成目录严格封存；未知文件／未完成事务保留并查因。不要手动删除标记、换commit或修改封存源码来使入口通过。

主候选为2024式adapt_cls_feat=True、dt=None、dilate=None，严格R5／N1885；Q不训练，U不弱监督，E不选型。保留特征合成，未加外部RGB合成。640×232与FP32／batch1／workers0是候选；权重参数加载／兼容、训练预算、资源峰值尚未实测。源码MIT及helper许可均保留，不等于V1权重任意商用授权；继续限个人非商业研究。所有训练由用户手动启动，当前没有训练命令。
