# 正式模型独立正常校准

只绑定最终10000步，不选择中间权重。准备与实际评分／定阈值分开，不改任何已封存入口或模块全局路径。训练继续仅由用户手动启动，本模块没有训练入口。

CPU准备／核验使用旧CPU环境，不读取Q／E／U原图、GT或概率，不导入模型：

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.calibration_cli prepare
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.calibration_cli verify
```

实际入口已经实现，但准备任务不执行。只在用户明确授权实际校准后运行：

```powershell
& '.\.venv-pretrained-seg\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.calibration_cli run
```

固定Q200原序，每图一次、每预定窗一次；预测器只接收RGB。单分割头softmax／FP32／batch1／workers0、ImageNet归一化、原生256窗／stride128／尾窗去重／右反射补边剔除；按y升序float32累加、uint16覆盖、float32等权平均，保存原始float32原生概率图及max，不平滑或拉伸。

全部完成且参数／BN前后不变才从200个原生max作q0.95／linear，严格大于τ判异常。同池FP/200不是独立验证，不能判断异常召回、真实检测提升或屏幕效果。

准备结果：`results/pretrained_segmentation/ksdd2_r5_v1/normal_calibration_preparation_v1`；实际结果：同分支`normal_calibration_v1`，二者独立封存。完成缓存同入口只核验；存在部分／未知结果、残留临时目录或failure时保留停止，不能删标记、移目录冒充新结果、重复评分、自动补推理或补阈值。新运行先核资源再建立全新临时目录，不静默复用；硬中断可能使前向状态不明，必须另行诊断方案。

CPU完整结果核验（仅在真实结果已完成后使用，不重算Q分位数或前向）：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_segmentation.calibration_cli verify-results
```

CPU准备RAM≥2GiB／磁盘≥2GiB；新模型前向启动RAM≥4.5GiB／显存≥6GiB／磁盘≥20GiB；每图前RAM≥2GiB／磁盘≥1GiB，否则保留停止，不降低门槛。完整结果CPU核验RAM≥2GiB／磁盘≥1GiB。实际耗时与峰值尚待Q运行实测，不按训练时间承诺。运行不自动读取E、重新训练或进入指标阶段。
