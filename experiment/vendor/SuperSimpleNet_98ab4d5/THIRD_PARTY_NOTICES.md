# Source and license preservation

The 12 author files are unmodified bytes from blaz-r/SuperSimpleNet at
98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d. LICENSE is the author's MIT notice.
Only model/feature_extractor.py, model/supersimplenet.py, common/loss.py and
common/perlin_noise.py are candidate core sources. train.py, datamodules and
requirements are reference evidence ONLY, not runnable approved entrypoints.
They include dependencies and data/test behavior that this project excludes.

common/loss.py credits Apple ml-destseg/model/losses.py; common/perlin_noise.py
credits VitjanZ/DSR_anomaly_detection/perlin.py. Their original comments remain.
Supplemental upstream LICENSE snapshots are retained separately. Their pinned
license commits do not establish exact copied-code lineage or resolve all rights.
No helper is relabelled as original project code. DSR's upstream Apache-2.0 and
Apple's own license text must be read alongside the author MIT notice.

Torchvision v0.23.0 code LICENSE and README are supplemental snapshots; the tag
URL and actual file hashes are recorded. Its pretrained-model disclaimer may
involve training-data-derived terms; code licensing is not weight licensing.
No separate complete rights grant for this ImageNet V1 artifact was established.
Use remains personal learning / noncommercial research, not commercial clearance.
Only existing generic V1 weight bytes are hashed; no pickle is deserialized.

No task-trained weights, data, images, GT or arrays are acquired here. This is
a source subset, not a complete runnable clone. Never run author train.py or
import the model during this static resource stage.
