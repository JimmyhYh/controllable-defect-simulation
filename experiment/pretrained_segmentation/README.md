# 预训练单头分割：当前实际入口

当前只实现第1阶段CPU资源模块；没有训练、输入准备、校准、评分或指标命令。新环境和通用ResNet18权重均未取得。不安装或导入Torch／SMP，不读图像、GT、数组或模型权重，不执行pickle。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.resources verify
```

该命令只读核验完整CPU协议、固定来源、旧元信息／选定源码、现有解释器／包元信息、实际本地安装材料和静态依赖闭包。会流式hash本地已选wheel，包含约3.22GiB的Torch安装包；不反序列化或安装它。未知文件／部分事务／身份冲突保留停止，不自动补齐、覆盖或重跑。

同模块还实现`acquire`（仅固定清单源码／文本取得；已有完整vendor只核验）、`verify-vendor`（只读来源核验）及`prepare`（CPU协议准备；已有完整缓存只核验）。acquire需要网络权限，不取得任何权重、安装包或数据；prepare／verify禁止联网。首次运行顺序为acquire→prepare→verify，不自动串联后续阶段。

源码子集在vendor/SMP_0_5_0，CPU协议在results/pretrained_segmentation/ksdd2_r5_v1/resource_preparation。模型、依赖和图像的运行兼容尚未实测；MIT源码许可不替代模型卡的license: other或训练数据条件。用途限个人学习／非商业研究。

第2阶段创建新环境或模型资源后须在其新身份协议中引用并核对本阶段来源，不能要求本阶段仍断言“新环境不存在”。所有训练均由用户手动启动；本文不提供尚未实现的训练命令。
