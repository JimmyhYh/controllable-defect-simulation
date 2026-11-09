# 第3阶段：独立配对训练准备

本轮只执行下列CPU准备／只读核验入口，不启动模型、兼容前向或训练。使用原CPU环境，不修改任何旧环境、全局路径或封存模块。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.training prepare
```

准备结果独立放E:/华兴源创公开实验/pretrained_synthetic_pairing/ksdd2_r5_v1/results/training_preparation_v1。已有完整同身份结果只读核验；未知／部分目录、锁、failure或hash不同均停止保留，不覆盖、补跑或删标记。

```powershell
Set-Location -LiteralPath 'D:\华兴源创\实验\弱缺陷Demo'
& '.\.venv\Scripts\python.exe' -B -X utf8 -m pretrained_synthetic_pairing.training verify
```

CPU输入永久禁止模型导入的审计钩子仍保留。新独立CPU子进程只复用冻结inputs的身份／元信息帮助函数，再独立核完整来源、闭合目录和封存结果；不调用旧inputs verify/resources或修改全局路径，不在模型进程导入inputs。新training_data只在未来显式模型操作中延迟导入NumPy：按冻结源hash从同一NPZ字节解码，独立复现预处理，使用前逐项核冻结image／target／valid及联合hash。原生uint8只除255一次；合成float不再除255、转uint8、翻转或补边。新模型环境NumPy与CPU环境的数组字节兼容尚未运行，未来25项check_forward及逐请求身份不符即停止，不放宽hash或换样。

training／training_runtime已实现显式人工运行包装，但本轮不提供其启动命令。正式入口强制要求独立25项只读兼容完整gate、相同冻结实现／通用资源／环境／请求身份及六组顺序。只读兼容获另行授权后才运行，不能把CPU准备成功当已完成模型兼容；兼容通过后再给首组实际人工训练命令。没有默认自动操作或串联六组训练。

正式每组10000更新、每1000步完整检查点，A先B后、seed2026102301/02/03；R5／N1885与固定旧2500材料、数据seed2026102300和三对共同请求不变。通用ImageNet编码器＋新解码器／头初始化，严格复用原模型、CE＋Dice、Adam两组LR、固定编码器BN affine微调／decoder GN8、FP32/batch1/workers0。1—8000次LR1e-5/1e-4；8001—9000为2e-6/2e-5；9001—10000为4e-7/4e-6。显式恢复按已完成步数确定下一更新，10000后无下一步。

完整恢复保存本组模型／受限weights_only optimizer、Python／NumPy／Torch CPU与全部CUDA RNG、完整请求／LR／有限损失前缀与像素曝光、参数／BN签名、资源和来源binding。只允许手动指定自己的最新完整保存点；部分最后检查点不能跳过，部分最终封存不能从10000点假装恢复。Ctrl+C不额外保存，OOM／未知参数更新保留查因。GPU恢复未实测。

首次CPU准备RAM≥2GiB／D≥2GiB，E按全轮预计最低135.662GiB；普通只读核验RAM／D／E各≥2GiB，不重新套用全轮初始预算。模型启动RAM≥4.5GiB／空闲显存≥6GiB／E≥20GiB，首组从头另核全轮空间，显式恢复和后续组不重复初始预算。每100步采样，RAM<2GiB在已完成边界保存本组完整停止点。前向逐图边界RAM≥2GiB／E≥1GiB；不降低规格。新包装／逐输入hash／新训练耗时尚未实测；先导10000步约11.32分钟仅历史参考，不承诺新六组时长。

本轮不读Q／E／U原图、GT或概率，不重校准／算指标或用旧E选型；E已见仅探索性。图像级正常／异常曝光匹配不等于真实异常或阳性像素／背景组成匹配。所有训练由用户手动启动。
