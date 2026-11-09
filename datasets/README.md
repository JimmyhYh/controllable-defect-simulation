# 数据集与获取方式

本仓库为私人设备上的公开资源学习项目，限个人学习与非商业研究，公司内部材料不在范围内。

## 随 Release 分发的原始数据

[打开数据与元信息下载页](https://github.com/JimmyhYh/controllable-defect-simulation/releases/tag/snapshot-20261109)（需登录有权限的账号）。

* **KolektorSDD2**：见 `KolektorSDD2-with-license.zip`。外层包包含未改动的原始ZIP、完整CC BY-NC-SA 4.0条款与署名。官方入口：https://www.vicos.si/resources/kolektorsdd2/
* **MVTec AD leather 子集**：见 `MVTec-leather-with-license.zip`。外层包包含未改动的原始tar.xz及原许可／署名。官方入口：https://www.mvtec.com/research-teaching/datasets/mvtec-ad

校验值见 assets.json。外层打包不改变原图／掩码，仍按各自原许可使用；不是完整15类别MVTec AD镜像。

## 仅提供获取入口，不镜像原始内容

* **AnoLCD**：https://github.com/SijieLuo/DLAS-Net 。作者提供百度网盘下载；未确认明确再分发条款，不上传原始包、原图或GT。本地核验为2400训练正常、350测试正常、750测试异常、750掩码，与网页旧描述存在差异，不能以网页数量覆盖本地核验。
* **DTD**：https://www.robots.ox.ac.uk/~vgg/data/dtd/ 。官方提供研究用途数据；本次不推定全面再分发授权，由使用者从官方获取并遵守原条件。

报告中有限的既有样例图属于阶段研究说明，不构成上述数据集镜像。没有独立新增真实屏幕数据集发布。

## 本地复现路径

原KSDD2解压到 experiment/data/fewshot_synthetic/ksdd2_v1，leather解压到 experiment/data/mvtec_ad/leather。历史配置和源码保存了D/E盘绝对路径及源码身份守卫，迁移后不能直接声称可运行；应建立独立适配入口，不改旧hash或失败结论。本上传是归档交接，不是已验证的跨机器一键复现包。
