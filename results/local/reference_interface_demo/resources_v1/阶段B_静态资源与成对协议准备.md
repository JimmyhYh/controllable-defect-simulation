# 阶段B：静态资源与成对推理协议准备

日期：2026年10月16日。状态：STATIC_PROTOCOL_PREPARED_RESOURCES_PENDING。阶段B静态准备完成，不代表资源已取得或模型可运行；ready_for_stage_C=false。

## 固定输入与隔离

绑定阶段A completion的SHA256：6d62ef6137d2611442d5038842d558a72f5f99bd9226c9d893d920205d1c608b。仅复核封存文字元信息、三份阶段A源码及64份旧文字／源码／元信息；继承已核验的来源与二进制hash，本轮不打开任何图像或数组字节，不重新审阅结构。仍为1949原move、019／024／020、两种子，6请求／3几何／1背景。原v3仍3/24、ready=false，不替代原48请求。

阶段A代码／21文件及旧成果不变。阶段B采用resources.py、resources_cli.py、resource_spec.py独立实现与本目录独立协议，没有改旧CLI。基础模型、编码器、适配器未加载，未导入Torch、下载权重、安装依赖、前向、合成或训练。

## 候选与官方依据

基础模型固定为[stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/blob/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb/README.md)，revision=8a4288a76071f7280aedbdb3253bdb9e9d5d84bb。该仓库明确是非Runway关联镜像，不能称原厂官方发布；模型卡标示CreativeML OpenRAIL-M及研究用途。[模型卡所链许可](https://huggingface.co/spaces/CompVis/stable-diffusion-license)规定使用限制及再分发时的许可／声明保留；镜像清单未列单独LICENSE文件。本项目继续仅个人非商业公开研究并保留这些限制，不将镜像声明当作原厂权重同一性证明。

参考条件固定为[h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/README.md)的models/ip-adapter-plus_sd15.safetensors及models/image_encoder，revision=018e402774aeeddd60609b4ecdb7e298259dc729。作者模型卡说明Plus使用ViT-H局部特征；不是SDXL、FaceID或bigG。适配器库标示Apache-2.0，原[LAION ViT-H模型卡](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/README.md)标示MIT；本轮绑定的是h94提供的转换文件，未另换LAION权重。

静态配置确认补绘UNet输入9通道／交叉注意力768，图像编码器224输入／patch14／hidden1280。[Diffusers v0.35.1参考加载源码](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/src/diffusers/loaders/ip_adapter.py)与[补绘API](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/inpaint)提供对应入口，形成可审查的接线依据；本地权重header、完整参数和运行兼容性仍待核验。它们不是弱Mura有效性证据。

## 已冻结的成对设置

同一StableDiffusionInpaintPipeline和已加载适配器，参考关闭0、开启0.5；不卸载适配器。[官方注意力实现](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/src/diffusers/models/attention_processor.py)的零scale分支跳过IP贡献，未来仍须核对实际处理器和全部scale。

512×512、batch1、FP16、DDIM30步、strength1.0、CFG7.5、eta0；精确beta／调度设置见paired_inference_protocol.json。提示固定为“a dark display surface with a subtle diffuse brightness non-uniformity”，负提示固定为“text, symbols, objects, watermark, sharp edges, high contrast”。这是生成前的工程选择，不是按样例调出的最优参数。

参考保留阶段A原始512上下文，不涂黑语义掩码；按[固定版本CLIPImageProcessor](https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/models/clip/image_processing_clip.py)标准224缩放／中心裁剪／均值方差归一化。该预处理不改变封存原图，不做自适应对比度增强；224编码可能丢失弱Mura，作用未知。Plus取倒数第二层特征；条件／无条件特征同一份缓存供开关条件使用。原参考掩码用于既定几何，不额外增加IP注意力区域掩码。

每种子未来在CPU产生一份[1,4,64,64]float32初始噪声，保存hash后转FP16，成对共享，传入latents时不自行乘scheduler sigma。每请求的masked-image VAE采样用单独同seed CPU生成器，沿用作者sample行为而非改用mode，预先保存一份潜变量并通过masked_image_latents共享。重置调度与条件生成器，eta0；只在同设备／软件条件下约束比较，不承诺跨版本逐位一致。以上本轮均只写协议，没有产生Torch噪声或潜变量。

仅使用SDPA math，保持IPAdapterAttnProcessor2_0；不启用可能替换处理器的attention slicing、xformers、CPU offload或VAE tiling。关闭TF32／benchmark、启用确定性要求，保留安全检查器；不支持就停止，不自动切换。原始模型输出与R内受限复合未来分别保存，域外零变化来自工程复合，不能算模型学会保留背景。

## 尚缺资源与实际设备

resource_manifest.json列出20个确切文件，其中6个权重，总计5368822000字节（5.000GiB，含小配置／分词器／模型卡，不含Python软件包）。ViT-H下载文件自身约2.35GiB；模型卡泛称适配器轻量不包括它。仅选择base中fp16 safetensors，以及h94的一个Plus文件和一个编码器safetensors，不下载整个模型仓库或.bin重复版本。文件大小、LFS SHA256／小文件Git blob SHA1及固定revision链接均已保存。

预期本地根：D:\华兴源创\实验\弱缺陷Demo\models\reference_interface_demo\v1，分别放base／ip_adapter子目录；尚未创建或取得。本次独立环境D:\华兴源创\实验\弱缺陷Demo\.venv-reference-interface也未创建。requirements-reference-interface.txt仅为经PyPI元信息审查的核心候选pin，Python3.12.7、Torch2.8.0+cu128等；没有执行完整依赖解析、安装、二进制导入或运行，不把候选依赖称为可用环境。后续另存完整传递依赖与实际身份，旧环境不修改。

实际快照：NVIDIA GeForce RTX 5060 Laptop GPU，总显存8151MiB、空闲7899MiB，驱动595.97；RAM可用4.958／总15.622GiB，磁盘空闲218.885GiB。快照非峰值。保守启动门槛为可用RAM≥6GiB、显存≥7168MiB、磁盘≥20GiB，当前分别{'RAM': False, 'GPU': True, 'disk': True}。这些是工程门槛而非实测最低需求；约3.8GiB的FP16权重驻留估算不含激活／临时副本／CUDA及加载峰值，8GB运行可行性仍未知，不能套用旧DRAEM耗时。

## 已实现命令与停止规则

独立resources_cli提供prepare-resources、verify-resources、inspect-local-resources；没有下载、安装、模型加载或生成命令。完整同身份prepare只核验缓存，不刷新快照或网络；inspect只读当前指定目录、文件hash与safetensors JSON header及目标环境dist-info。若权重存在，完整hash会流式读取整个权重文件字节，header解析只读取其中JSON元信息；不导入Torch、反序列化张量或实例化模型。header只核对必要接口尺寸，不代替完整模型加载兼容性检验。本次指定模型目录不存在，没有读取任何本地权重字节。

未知文件、partial／锁、身份／API不符、资源不足、OOM、安全拦截或不明确前向必须保留并查因，不换seed、挑结果、放宽规则或自动重试。当前缺环境和权重，且前向实现尚不存在，不进入阶段C。全部未来训练继续由用户手动启动。

元信息访问曾在sandbox socket连接处报WinError10013，经systematic-debugging定位访问层，获准的同类公开元信息请求成功；没有将该错误当成模型下架，也没有改项目数据或安全检查器。
