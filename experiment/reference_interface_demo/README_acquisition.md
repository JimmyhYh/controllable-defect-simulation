# 固定资源取得与静态身份核验

> 当前状态（2026年10月16日）：用户已要求停止下载。4/20文件完整、1部分、15未开始，新环境仅pip24.2；不完整事务与下载残片保留。以下为入口原定义，**当前不得重跑acquire或verify-acquisition作为恢复操作**。等待手动资源，先按[手动补齐说明](D:/华兴源创/实验/弱缺陷Demo/results/reference_interface_demo/acquisition_v1/手动资源补齐说明.md)审阅；没有实现中断恢复或离线安装入口。

独立入口acquisition_cli仅下载20个固定文件、建立新环境、安装固定核心依赖并静态核验；不导入Torch／模型／图像包，不生成／训练，不读背景或参考图。下载流量约5.00GiB，环境软件包另计；下载时间取决于网络，不预设耗时。

PowerShell获取入口（已实现）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.acquisition_cli acquire
```

仅在首次空目录或已有完整同身份成果时运行。存在未知／不完整获取、新环境或模型目录时会停止保留，不能删除目录后强行重跑。每文件一次逻辑下载，允许正常HTTPS重定向，没有网络重试／续传；大小或hash不符保留.download.part。安装仅wheel、网络retry0；安装失败保留新环境和诊断，不自动重装、升级旧环境或改pin。

安装进程明确禁用所有pip配置文件并清除继承PIP／PYTHON选项，绑定捆绑pip24.2、显式新环境prefix，运行前检查解释器、安装scheme及空配置，避免系统pip.ini把安装重定向到旧环境。没有修改全局pip配置。

完成后只读核验（已实现）：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m reference_interface_demo.acquisition_cli verify-acquisition
```

完整缓存下，acquire也只进入该核验路径，不重新下载或安装。核验会流式hash全部模型文件并检查新环境元信息，因此可能需要数分钟，取决于磁盘；不会把权重反序列化为张量。pip check检查依赖元信息，使用禁止模型包导入的独立进程。

模型仅写入 `D:/华兴源创/实验/弱缺陷Demo/models/reference_interface_demo/v1/`；新环境仅为 `D:/华兴源创/实验/弱缺陷Demo/.venv-reference-interface/`；最终结果仅在 `D:/华兴源创/实验/弱缺陷Demo/results/reference_interface_demo/acquisition_v1/`。不修改阶段A／B六份源码、21／9文件、旧.venv或旧模型。旧环境元信息／RECORD及解释器身份前后比较，未全量重hash旧包二进制。

最终科学产物包括固定取得协议、20文件实际hash与来源、6权重必要header元信息、pip实际wheel报告、完整传递依赖及解释器身份、已有inspect-local-resources输出和中文最终说明；没有过程表或软件测试套件。

只有完整completion出现且核验通过才称资源取得完成。资源文件／环境身份通过不证明运行兼容、内核或8GB峰值可行，也不授权模型加载／前向；原v3仍3/24、原48请求目标不变。所有训练始终由用户手动启动。
