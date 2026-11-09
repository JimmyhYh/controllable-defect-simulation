"""Static SSN core/coordinate contract, with no runtime package imports."""

MODEL_CONFIG = {
    "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"], "patch_size": 3,
    "adapt_cls_feat": True, "dt": None, "dilate": None,
    "noise": True, "perlin": True, "no_anomaly": "empty", "bad": True,
    "overlap": False, "noise_std": 0.015, "perlin_thr": 0.6,
    "stop_grad": False, "adapt_lr": 0.0001, "seg_lr": 0.0002, "dec_lr": 0.0002,
    "gamma": 0.4,
}

INPUT_OUTPUT = {
    "input_hw": [640, 232], "RGB": "native uint8 RGB; no contrast stretch",
    "RGB_resize": "OpenCV INTER_LINEAR on uint8, H=640 W=232, then /255 and channel normalization",
    "RGB_normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "GT": "native binary mask must be checked before use; preserve native GT unchanged",
    "GT_to_input": "OpenCV INTER_NEAREST to 640x232, binary float32 in {0,1}; no dilation or distance mask",
    "GT_to_feature": "torch bilinear to (160,58), align_corners=True; values <0.5 become 0, >=0.5 become 1",
    "feature_shape_static": [1536, 160, 58], "feature_copy": "author generator duplicates B to 2B; not two independent real images",
    "empty_feature_GT": "stop if any R5 becomes empty; do not replace sample, dilate or borrow U",
    "native_score_order": ["model.eval: feature logit -> bilinear to 640x232 (align_corners=False)",
                           "author GaussianBlur kernel25 sigma4 reflect on logits",
                           "bilinear smoothed input logits to native H,W, align_corners=False",
                           "one-channel sigmoid to native float32 probability, each native pixel once"],
    "classification": "sigmoid of independent raw classification-head logit; no spatial pooling substitution",
    "localization_max": "max of native sigmoid map; supplemental, not the classification score",
    "calibration": "not performed or numerically frozen at this stage; future Q only, no old threshold reuse",
    "evaluation": "no test-pool minmax, smoothing after sigmoid, resized evaluation GT, softmax or best-checkpoint selection",
    "BN": "author BatchNorm retained; not old GN or stateless BN",
    "budget": "pilot steps/epochs, sample sequence, gate and formal budget remain unfrozen",
    "data_codec_status": "core environment has no OpenCV wheel; decode/resize implementation and runtime dependency are a stage-3 prerequisite, not claimed ready",
}

OPTIMIZER = {
    "algorithm": "author get_optimizers: AdamW, three groups, no backbone params",
    "groups": [{"parameters": "feature_adaptor", "lr": 0.0001, "weight_decay": "inherited AdamW default 0.01"},
               {"parameters": "discriminator.seg", "lr": 0.0002, "weight_decay": 0.00001},
               {"parameters": "dec_head + fc_score", "lr": 0.0002, "weight_decay": 0.00001}],
    "scheduler": "author epoch-unit MultiStepLR at 0.8*epochs,0.9*epochs, gamma0.4; not old 8000/9000 step schedule",
    "epochs": None, "optimizer_constructed": False,
    "loss": "seg normal truncated-L1 mean + seg anomaly truncated-L1 mean + all-pixel focal mean + cls focal mean",
    "truncated_L1_margin": 0.5, "focal_gamma": 4, "focal_alpha": -1,
    "empty_normal_or_anomaly_subset": "zero contribution, as author train.py; no zero-division",
    "segmentation_available": "all strict-R5/normal samples fully segmented; no weak labels or loss-mask weighting",
    "gradient_clipping_future": "author supervised entry: clip global gradient norm to1 if enabled; not executed",
}
