# 最小参考条件接口Demo：阶段B静态资源

本模块只使用标准库做元信息、协议和本地文件核验。不会下载权重、安装软件、导入Torch／Diffusers／图像库、执行模型或训练。阶段A三份源码及21份输入产物保持冻结，原v3仍3/24通过、ready=false。

在 **PowerShell** 核验阶段B已冻结协议：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.resources_cli verify-resources
```

静态准备入口已实现；已有完整同身份结果时只核验复用：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.resources_cli prepare-resources
```

查看当前指定本地资源与独立环境的实际缺项，不下载或加载模型：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.resources_cli inspect-local-resources
```

三个入口均不读取背景／参考图像或数组字节。阶段B结果位于 `D:/华兴源创/实验/弱缺陷Demo/results/reference_interface_demo/resources_v1/`，包括固定20文件资源清单、成对推理协议、依赖候选、官方证据链接、阶段A身份、设备快照及中文最终说明。缓存仅核验不刷新旧快照；inspect命令只将当前状态打印到终端，不往冻结结果追加记录。

模型预期放置根为 `D:/华兴源创/实验/弱缺陷Demo/models/reference_interface_demo/v1/`，其中base保存SD1.5补绘镜像的16个指定文件，ip_adapter保存h94仓库的4个指定文件。具体revision、原仓库路径、local_path、大小、LFS SHA256／Git blob SHA1及固定下载来源链接见resource_manifest.json。总计约5.00GiB，含6个权重，不含环境软件包。不要下载全仓库的多种精度或.bin重复版本；本次没有提供／实现下载命令。

独立环境预期位置为 `D:/华兴源创/实验/弱缺陷Demo/.venv-reference-interface/`，候选核心依赖见requirements-reference-interface.txt。该文件尚未执行解析或安装，不代表兼容环境已建立；旧.venv不改动。实际传递依赖身份与运行兼容性仍须后续核验，本次无安装／激活命令。

inspect若找到模型文件，会流式读取整个权重文件字节计算hash，并仅将safetensors JSON header解析为形状元信息，不反序列化张量或实例化模型。全部文件hash在普通CPU磁盘上可能需数分钟，取决于介质，未实测总耗时。本次指定模型目录不存在，没有读取本地权重字节。文件正在复制、大小不符、未知.tmp／锁／文件、代码或旧元信息身份变化时停止保留，不删除或自动修复；请勿边复制边核验。

当前协议仅规划同一模型参考权重0／0.5、DDIM30步、512、FP16、同噪声与同masked-image潜变量；12份输出实际仍为0。完整资源核验也不等于获得前向授权，后续还需实现与审阅只读推理入口。RAM≥6GiB、空闲显存≥7168MiB、磁盘≥20GiB是保守工程启动规则，不是运行保证；8GB扩散峰值和速度尚未实测。

作用范围补充：本模块的open／导入／socket防护只覆盖当前调用路径，不是通用操作系统沙箱。未来新模块仍须单独审阅。所有训练继续由用户手动启动。
