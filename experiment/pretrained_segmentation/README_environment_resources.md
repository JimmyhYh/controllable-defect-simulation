# 第2阶段独立环境／资源入口

本阶段只取得固定SMP／timm安装包、ResNet18通用权重及建立独立环境，不导入模型包、解析tensor值或进行模型操作；训练入口尚未实现。第1阶段源码、README与封存结果不改。

已实现入口为`pretrained_segmentation.environment_resources`：prepare（CPU绑定准备）、acquire（三份固定资源）、install（离线隔离安装）、verify-acquisition（只读资源核验）、verify（只读本阶段完整核验）。入口不自动串联，不明确或部分事务保留停止，不自动重下载／修环境／换pin。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.environment_resources verify
```

prepare绑定原封存来源／设计／26项候选、原环境元信息、本机捆绑pip24.2及新模块身份；新环境创建后不调用第1阶段的取得前absence gate。acquire仅按固定URL／大小／hash取得两份wheel和model.safetensors，允许固定HF resolve转官方HTTPS CDN，实际文件载荷49464388字节；只按JSON header核对描述与偏移，不解码tensor。

install使用Python3.12.7自带pip wheel在新`.venv-pretrained-seg`中离线安装，24本地wheel＋2新wheel＋pip24.2，no-index／no-deps／require-hashes；不使用旧site-packages或修改旧环境。模型包导入和pickle操作、离线联网有显式guard；pip内部仅允许导入标准库pickle定义，不调用其序列化／反序列化。

资源目录models/pretrained_segmentation/resources_v1；独立配置results/pretrained_segmentation/ksdd2_r5_v1/environment_preparation_v1；安装／核验结果environment_v1。verify读完整缓存、hash已有约3.26GiB wheel材料并核新环境包元信息，会耗时；无前向、安装、下载或训练。不得使用新环境解释器执行此静态入口，必须使用上方原.venv，以核对原准备身份。

本阶段静态完成不等于权重可加载、BN行为正确或模型有效。仅私人设备、个人学习非商业研究；SMP源码MIT不替代模型卡license: other。所有训练持续由用户手动启动。
