# 独立六模型Q200正常校准

只复用本轮六组最终10000步权重，按A01、B01、A02、B02、A03、B03顺序分别校准。封存训练／输入／旧评分代码及全局路径不变。准备和核验使用CPU环境；每次实际校准由一个显式单模型命令启动，不提供自动批量入口。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.calibration prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.calibration verify
```

人工首组A01（准备完成后使用，本阶段不执行）：

```powershell
& '.\.venv-pretrained-seg\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.calibration run --group A --seed 2026102301
```

完成后的只读核验：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.calibration verify-results --group A --seed 2026102301
```

准备结果在E盘results/calibration_preparation_v1。实际结果分别在results/calibration_v1/{组}_{seed}，运行临时目录为runtime_calibration_v1/{组}_{seed}。每模型原200正常图／844窗，原生29,202,876像素仅来自元信息；六模型1200次图像评分仍为200张唯一正常图。

256窗／stride128／尾窗去重，右反射补边剔除，RGB保留对比度并按ImageNet归一化；y升序FP32等权拼回原始异常softmax，保存float32原生概率及max。每模型q=.95／linear，score严格大于新tau；不用旧先导tau或Q分数。同池FP/200非独立验证，不支持异常召回、AP或屏幕收益。

预测器不读取GT或标签，不创建optimizer、不更新参数或BN；最终参数／BN前后与权重来源核验。每图一次解码，每窗一次前向；运行前要求前组完整，各窗前写入持久化意向。完整同身份缓存仅只读核验，不重复前向或分位数。任何部分／未知输出或仅剩runtime都停止保留，不删标记、不补推或自动求阈值。Ctrl+C/OOM须先诊断。

每组完成时将runtime的owner、全部临时文件hash和完整目录清单绑定至结果中的runtime_manifest.json；新增未知文件、锁、空目录、owner或内容变化均停止。当前缓存和后组准入共用相同元信息／hash核验，不读取前组概率载荷或重新分位数。该严格规则也会拒绝第三方退出后自动改动临时文件的情况，须保留查因，不能将其自动视为可忽略；实际第三方／CUDA交互尚未运行。

CPU准备RAM>=2GiB、D/E各>=2GiB；模型启动RAM>=4.5GiB、空闲显存>=6GiB、E>=20GiB，逐图RAM>=2GiB／E>=1GiB。原生适配、CUDA和时间／资源只静态准备，未实测；先导约34.75秒/模型不保证本轮时间。训练始终由用户手动启动。本阶段不安装、下载或操作Q/E/U像素与GT，不自动运行实际校准或E评价，不建过程表／软件测试／提交流程。
