# Fixed source subset and resource conditions

20 unmodified SMP source/document/license files are pinned to v0.5.0 commit
420ce84b0c2df0286fa9bb2bd1499eea625c9b33. Their actual SHA256 and Git blob
identities are recorded. This is an inspection subset, not an importable clone.
No author source, model or package initializer is executed at this stage.

SMP source is MIT. The separately pinned ResNet18 model card says license: other
and points to the original resnet18-5c106cde.pth artifact. Neither that label nor
SMP's source license establishes an unrestricted grant for ImageNet weights.
The upstream torchvision README warns that pretrained models can have training-
data-derived terms. Its already archived v0.23.0 README/LICENSE are copied as
notices only; they do not prove an exact training lineage or new weight grant.
Use remains personal learning / noncommercial research, not commercial clearance.

Timm 1.0.20 is a declared SMP dependency, chosen explicitly as a fixed candidate
satisfying the published lower bound, not an automatic upgrade. PyPI metadata
reports Apache-2.0. Actual future wheels and their license files still require
verification. There is no claim of runtime compatibility or paper replication.

Only UTF-8 code, documentation, configuration and package/repository metadata
are acquired. The approximately 44.64MiB safetensors file and two missing wheels
are NOT downloaded here. No task weights, dataset, pixels, GT or arrays are read.
