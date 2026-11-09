"""Strict fully-segmented subset of author train.py objective; runtime only.

Preserves normal/anomaly separate means and empty-set behavior. No training
loop, optimizer construction, backward, step, dataset or metric is provided.
"""


def loss_terms(anomaly_map, score, mask, label, is_segmented):
    import torch
    from .core_v1.loss import focal_loss
    if not bool(is_segmented.bool().all()):
        raise RuntimeError("Strict R5/normal wrapper rejects weak/unsegmented samples")
    if anomaly_map.shape != mask.shape or anomaly_map.shape[0] != 2 * is_segmented.shape[0]:
        raise RuntimeError("Expected author feature duplication B->2B and returned synthetic masks")
    th = 0.5
    seg_focal = focal_loss(torch.sigmoid(anomaly_map), mask, reduction=None)
    seg_l1 = torch.zeros_like(anomaly_map)
    normal_scores = anomaly_map[mask == 0]
    seg_l1[mask == 0] = torch.clip(normal_scores + th, min=0)
    anomalous_scores = anomaly_map[mask > 0]
    seg_l1[mask > 0] = torch.clip(-anomalous_scores + th, min=0)
    is_segmented = torch.cat((is_segmented, is_segmented)).type(torch.bool)
    bad_loss = seg_l1[is_segmented][mask[is_segmented] > 0]
    good_loss = seg_l1[is_segmented][mask[is_segmented] == 0]
    focal_val = seg_focal[is_segmented]
    good_loss = good_loss.mean() if len(good_loss) else 0
    bad_loss = bad_loss.mean() if len(bad_loss) else 0
    focal_val = focal_val.mean() if len(focal_val) else 0
    seg_loss = good_loss + bad_loss + focal_val
    cls_loss = focal_loss(torch.sigmoid(score), label)
    return {"total": seg_loss + cls_loss, "seg_normal_l1": good_loss,
            "seg_anomaly_l1": bad_loss, "seg_focal": focal_val, "classification_focal": cls_loss}
