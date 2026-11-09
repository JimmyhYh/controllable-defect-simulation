# 第2阶段：CPU共同请求与输入核验

仅新增inputs.py及本说明，不修改冻结common/resources、原单头代码或旧路径。使用CPU .venv；无Torch/SMP/timm/模型导入或tensor／训练／评分。原1890缓存及固定2500旧合成从同一字节快照核hash后以allow_pickle=False读取；不打开原RGB／GT或Q/E/U概率。仅原接受ellipse/line spec的geometry.rasterize用于ROI核验，render禁止导入。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.inputs prepare
```

prepare首次只允许独占创建E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results/inputs_v1，E实验根已存在而没有本组完整结果时停止，不认领或覆盖。transaction.json独占创建／fsync／读回即实际写入条件核验。未知／部分事务保留，只有本次创建的新事务可记录failure，不自动删除或重试。完整同身份产物已存在则只读verify。

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.inputs verify
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.inputs verify-decode
```

verify默认只读核封存结果／来源hash，不解码缓存；verify-decode显式再作完整CPU数值复核，不写任何结果，不自动执行。两者都不训练、前向、生成、校准或算模型指标。

数据seed2026102300只生成一次共同10000请求供三对初始化共用。共享7500槽指同一引用及完整归一化image／target／valid数组hash；合成缓存已经翻转及补边，直接从float32[0,1]减ImageNet均值／除标准差，不再除255、转uint8或二次翻转。背景重建遵循旧翻转，标签来自实际D>float32(1/255)。正常／异常图像级曝光匹配，不声称像素级曝光或真实异常曝光相同。

training_pack(group,update)只返回CPU数组及身份，限定CPU解释器且安装禁止模型导入的审计钩子；不能直接拿到未来模型进程调用。第3阶段须另立训练侧安全适配与身份核验，当前参数更新入口尚未实现。CPU RAM≥2GiB，D少量代码空间≥2GiB，E按第1阶段全轮预计最低135.662GiB核对，实际资源每阶段重新检查。所有训练仍由用户手动启动，不自动进入第3阶段。
