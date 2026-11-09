# 配对分支第1阶段：CPU来源与资源

仅本阶段common／resources与空__init__已实现，不导入旧项目模块、Torch／SMP／timm／NumPy或图像包。固定旧2500材料仅stream hash，不解码；通用权重仅stream hash及JSON header，原模型／optimizer只hash不加载，不执行pickle。独立模型环境的子进程仅用标准库核27项分发包、dist-info／解释器及安装源码身份，不安装或升级。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.resources inspect
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.resources prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.resources verify
```

inspect只读、不写结果；prepare仅新建D盘resource_preparation_v1科学协议，已有完整结果时只读verify；verify核封存结果、源hash、独立环境和当前资源，不改原结果。任何未知／部分目录、hash或依赖异常停止保留，无自动恢复／替换。退出0表示本阶段元信息与资源条件成立；2表示元信息有效但当前资源条件未满足，不能自动进入第2阶段。它们都不启动模型或训练。

只在prepare本次已经新建自己的事务后才允许记录failure；只读入口或接手已有部分目录不追加任何文件。准备与复核均核上游完整文件／目录集合，旧合成另两池仅枚举元信息、不打开载荷。仅既有安装源码声明的空workspace_temp精确允许；其他未知空目录也拒绝。环境子进程的身份检查用显式条件，启用Python优化模式也不能绕过。

第1阶段固定结果在`D:/华兴源创/实验/弱缺陷Demo/results/pretrained_synthetic_pairing/ksdd2_r5_v1/resource_preparation_v1`。用户指出E盘空间后，未来新大结果规划到`E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results`与同级models；本阶段不创建或写入E，未实测写入能力，不移动任何旧文件。

首次准入还要求E目标未被占用且目标及祖先没有目录联接／重解析点；本阶段不自动认领已存在目标。后续阶段创建E目录后，由其自身完整协议核验，不再用本阶段“新目录首次准入”冒充恢复。common.write与install_guard只限阶段1，后续须独立实现自己的E事务，不改这些冻结函数或全局路径。

CPU准入RAM／D盘各≥2GiB；E整轮最低130GiB且按末组AP仍需80GiB的完整模型／概率／保留表估算取更高者。未来模型RAM≥4.5GiB、空闲显存≥6GiB及GPU完整恢复均不由本阶段担保。数据请求seed2026102300和三个初始化seed只作为协议值冻结，没有生成训练请求或构造参数。

计划进度文字可更新，verify仅绑定第1／4节科学协议和显式输出根；不把可变摘要的全文件hash作为固定依赖。所有旧成果只读，E历史已见，仅探索性；后续输入、训练和评分入口未实现，全部训练由用户手动启动。
