# 独立E1004真实评分

新文件real_scoring.py／real_runtime.py／real_cache.py，仅引用封存训练／校准，不改旧入口或全局路径。CPU准备不读取Q/E/U图像、GT或概率字节；Q缓存只核关闭命名空间、完成身份及既有全量核验证据。GT仅单独绑定规范路径、已有来源hash和原生尺寸，实际类别／阳性分母留待未来CPU指标授权。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_scoring prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_scoring verify
```

首组人工实际评分（本次不运行）：

```powershell
& '.\.venv-pretrained-seg\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_scoring run --group A --seed 2026102301
```

完整结果存在后，CPU只读核验：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.real_scoring verify-results --group A --seed 2026102301
```

固定A01/B01/A02/B02/A03/B03，每模型新进程、最终10000步权重与本组原Q tau。每模型E1004／4270窗／146,411,528原生像素，6024模型图像评分仍1004唯一图。256窗stride128尾窗去重、右反射补边剔除、ImageNet归一化、y升序FP32等权原始异常softmax拼回max直接复用冻结predict_native；GT不进入预测，评分不计算阈值计数／AP／AUROC、不重校准。

结果results/real_evaluation_v1/{组}_{seed}；运行runtime_real_evaluation_v1/{组}_{seed}。逐窗前fsync意向、完整runtime owner／文件hash／目录清单，前组完整才准入后组。完整同身份缓存只核验，不重复前向；任何部分／未知目录、锁、空目录或不明确前向保留停止，不补推、不缩成功交集或删标记。运行时临时文件若在退出后被第三方改动也严格停止查因。

CPU准备RAM2GiB且D/E各2GiB；模型启动RAM4.5GiB／空闲显存6GiB／E20GiB，逐图RAM2GiB／E1GiB。本轮E实际运行兼容、耗时和峰值未实测，先导约172秒仅历史参考。未来CPU指标通过real_cache.iter_metric_inputs读取完整概率／分数与GT元信息，不解码GT或计算指标，当前不得调用。

只做来源、AST／CLI、整数窗口与虚构标量核验，不建过程表、软件测试、提交或工作树。E历史已见，仅探索性评价，不称新未见验证、稳定收益或屏幕效果。所有训练仍由用户手动启动。
