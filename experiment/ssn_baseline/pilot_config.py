"""Stage-3 technical pilot policy, independent of historical probes/results."""
INIT_SEED = 2026102101
ORDER_SEED = 2026102102
EPOCHS = 50
UPDATES_PER_EPOCH = 32
STEPS = 1600
CHECKPOINT_EPOCHS = [10, 20, 30, 40, 50]
MILESTONES = [40, 45]
GATE = {
    "scope": "same 21 training-side images only; technical fitting, not validation or performance improvement",
    "feature_global_pixel_AP_min": 0.95,
    "each_R5_feature_pixel_AP_min": 0.85,
    "classification_reference_score": 0.5,
    "classification_comparison": "strictly greater than",
    "R5_TP_required": 5,
    "N16_FP_max": 1,
    "all_scores_finite_and_each_feature_map_nonconstant": True,
    "final_step_required": 1600,
    "intermediate_weights_allowed_for_gate": False,
    "native_pixel_AP": "descriptive only; no gate to confound fitting with blur/native inverse mapping",
    "rationale": "high pooled feature-scale fitting plus a per-real-image floor prevents the largest mask dominating; classify all5 and at least15/16 normals correctly at a fixed reference, not a Q-calibrated threshold",
    "status": "pre-run engineering acceptance policy, not a paper performance claim or guaranteed achievable value",
}
RESOURCES = {"start_RAM_GiB": 6, "start_free_VRAM_GiB": 6, "start_disk_GiB": 80,
             "running_RAM_GiB": 3, "precision": "FP32", "batch": 1, "workers": 0,
             "time": "unknown before manual pilot; no old-model timing substituted"}
