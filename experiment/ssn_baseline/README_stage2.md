# 第2阶段：隔离环境与核心静态准备

本页独立新增，不修改已封存README.md／resources.py／vendor。新环境只准备Torch／torchvision等核心依赖，不等于数据管线、参数加载或模型运行已通过。全部训练继续由用户手动启动，当前未实现小试或训练CLI。

实际已实现且只读核验的入口：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m ssn_baseline.static_preparation verify
```

它核对第1阶段封存、vendor、原split／旧核心源码／已有V1字节hash，静态核对新环境METADATA／RECORD及包装源码、AST与标量；新环境仅开元信息子进程，不导入Torch／模型、不读取图像／GT／概率、不运行前向。

另有已实现CPU入口：`ssn_baseline.environment inventory`核对14份既有本地wheel及依赖闭包；`ssn_baseline.environment_module_bootstrap prepare／install／verify`承担独立已知空环境引导／离线安装／只读核验；`static_preparation prepare`建立独立派生核心／协议。完整缓存只读复用，部分／未知事务保留停止，不自动重试、修复或换pin。

原environment install在ensurepip步骤遇到沙箱默认TEMP拒绝写入；environment_continuation安装调用又在真正安装前遇到Windows pip自修改保护，两次均确认包数0。两次失败目录、封存配置与源码保持不变，不能当当前恢复入口。最终environment_module_bootstrap依据本机ensurepip源码采用alter_sys=True的标准模块调用语义，绑定解释器／pyvenv.cfg、0包空环境、两份failure、相同14份wheel／pin及Python自带pip24.2，使用项目内TEMP和新的独立事务；不删失败标记或改旧环境。它仅承接本阶段获准依赖安装，未运行模型。当前完整核验统一使用上方static_preparation verify。

环境清单／首次失败保存在`results/ssn_baseline/ksdd2_r5_v1/environment_v1/`，第二次失败在`environment_bootstrap_continuation_v1/`，最终环境在`environment_module_bootstrap_v1/`，最终核心协议在`core_preparation_v1/`；模型核心派生代码在`ssn_baseline/core_v1/`。未来`model.build_model`会加载本地V1并执行作者构造器尺寸前向，当前禁止调用；静态命令只解析该文件，未导入或执行。损失准备于objective.py，没有训练循环或参数更新。

冻结2024式adapt_cls_feat=True／dt=None／dilate=None及严格R5／N1885。Q不训练或选配置，U不弱监督，E不选型／训练期test；保留特征合成、双头、BN、作者损失和AdamW三组，不套旧GN或旧阈值。640×232与原生逆映射顺序见contracts.py／protocol.json；预算／采样／gate仍待第3阶段，不能直接运行模型。

当前新环境没有OpenCV，未来CPU解码／resize模块与其运行依赖仍须核对，不把端到端数据管线称为已准备完成。不得借用旧site-packages破坏隔离；若需额外安装，先核验材料及流量。来源MIT及helper许可不等于权重任意商用许可，继续限个人非商业研究。
