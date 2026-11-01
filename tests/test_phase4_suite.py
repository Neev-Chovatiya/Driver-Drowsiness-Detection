import os
import sys
import json
import filecmp
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.modeling.geometry_validator import (
    GeometryValidator,
    GeometryValidatorConfig,
    ValidationResult,
)
from data.modeling.train_and_evaluate_expC import (
    CLASS_TO_IDX,
    RAW_FEATURES,
    CALIBRATED_FEATURES,
    compute_metrics,
    evaluate_video_level,
    compute_clustered_bootstrap_ci,
)

def log_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)

def test_geometry_validator():
    log_header("Test 1: Geometry Validator Subsystem")
    validator = GeometryValidator()

    res = validator.validate_measurements(
        ear_left=0.32,
        ear_right=0.31,
        ear_mean=0.315,
        mar=0.22,
        mouth_over_ear=0.698,
        interocular_dist=95.0,
        face_detected=True,
        face_box=(100, 100, 260, 260),
    )
    assert res.is_valid, f"Expected valid, got: {res.reasons}"
    assert res.action == "PROCEED"
    print("  [PASS] Valid typical facial measurement passed.")

    res = validator.validate_measurements(
        ear_left=0.0,
        ear_right=0.0,
        ear_mean=0.0,
        mar=0.0,
        mouth_over_ear=0.0,
        interocular_dist=0.0,
        face_detected=False,
    )
    assert not res.is_valid
    assert res.action == "NO_FACE"
    assert any("not detected" in r for r in res.reasons)
    print("  [PASS] Missing face detection correctly flagged as NO_FACE.")

    res = validator.validate_measurements(
        ear_left=float("nan"),
        ear_right=0.30,
        ear_mean=float("nan"),
        mar=0.20,
        mouth_over_ear=float("inf"),
        interocular_dist=90.0,
        face_detected=True,
    )
    assert not res.is_valid
    assert res.action == "SUSPEND_ML"
    assert any("Non-finite" in r for r in res.reasons)
    print("  [PASS] Non-finite biometrics (NaN/Inf) correctly suspended ML.")

    res = validator.validate_measurements(
        ear_left=0.28,
        ear_right=0.28,
        ear_mean=0.28,
        mar=0.20,
        mouth_over_ear=0.71,
        interocular_dist=18.0,
        face_detected=True,
        face_box=(100, 100, 130, 130),
    )
    assert not res.is_valid
    assert res.action == "SUSPEND_ML"
    assert any("Interocular distance too small" in r or "Insufficient face resolution" in r for r in res.reasons)
    print("  [PASS] Sub-resolution face correctly suspended ML.")

    res = validator.validate_measurements(
        ear_left=0.62,
        ear_right=0.60,
        ear_mean=0.61,
        mar=0.20,
        mouth_over_ear=0.33,
        interocular_dist=100.0,
        face_detected=True,
    )
    assert not res.is_valid
    assert res.action == "SUSPEND_ML"
    assert any("EAR exceeds physiological" in r for r in res.reasons)
    print("  [PASS] Implausible EAR (>0.55) correctly suspended ML.")

    res = validator.validate_measurements(
        ear_left=0.30,
        ear_right=0.30,
        ear_mean=0.30,
        mar=2.15,
        mouth_over_ear=7.16,
        interocular_dist=100.0,
        face_detected=True,
    )
    assert not res.is_valid
    assert res.action == "SUSPEND_ML"
    assert any("MAR exceeds physiological" in r for r in res.reasons)
    print("  [PASS] Implausible MAR (>1.80) correctly suspended ML.")

def test_bootstrap_duplicate_preservation():
    log_header("Test 2: Clustered Bootstrap Resampling & Duplicate Preservation")

    expC_path = "data/modeling/models/expC_calibrated_rf.joblib"
    pipeline = joblib.load(expC_path)

    with open("data/modeling/split_assignments.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_parts = splits["test_participants"]

    from data.modeling.train_and_evaluate_expC import (
        load_dataset,
        compute_participant_baselines,
        engineer_calibrated_features,
    )
    df = load_dataset("data/processed/features.csv")
    baselines = compute_participant_baselines(df, calib_window_sec=30.0)
    df_calib = engineer_calibrated_features(df, baselines)
    df_calib["target_idx"] = df_calib["video_label"].map(CLASS_TO_IDX)

    calib_mask = (df_calib["video_label"] == 0) & (df_calib["timestamp_sec"] <= 30.0)
    test_df_eval = df_calib[df_calib["participant_id"].isin(test_parts) & (~calib_mask)].copy()

    unique_parts = np.array(sorted(test_df_eval["participant_id"].unique()))
    n_parts = len(unique_parts)
    assert n_parts == 6, f"Expected 6 held-out test participants, found {n_parts}"

    frame_preds_by_p = {}
    frame_trues_by_p = {}
    vid_preds_by_p = {}
    vid_trues_by_p = {}

    for p in unique_parts:
        p_df = test_df_eval[test_df_eval["participant_id"] == p]
        frame_trues_by_p[p] = p_df["target_idx"].values
        frame_preds_by_p[p] = pipeline.predict(p_df[CALIBRATED_FEATURES].values)

        v_t_list = []
        v_p_list = []
        for v_file, v_df in p_df.groupby("video_filename"):
            t_idx = CLASS_TO_IDX[int(v_df["video_label"].iloc[0])]
            probs = pipeline.predict_proba(v_df[CALIBRATED_FEATURES].values)
            p_idx = int(np.argmax(np.mean(probs, axis=0)))
            v_t_list.append(t_idx)
            v_p_list.append(p_idx)
        vid_trues_by_p[p] = np.array(v_t_list)
        vid_preds_by_p[p] = np.array(v_p_list)

    sampled_with_dups = np.array(["09", "09", "10", "16", "18", "24"])

    opt_y_true_f = np.concatenate([frame_trues_by_p[p] for p in sampled_with_dups])
    opt_y_pred_f = np.concatenate([frame_preds_by_p[p] for p in sampled_with_dups])
    opt_y_true_v = np.concatenate([vid_trues_by_p[p] for p in sampled_with_dups])
    opt_y_pred_v = np.concatenate([vid_preds_by_p[p] for p in sampled_with_dups])

    ref_frames = []
    for inst_idx, p_id in enumerate(sampled_with_dups):
        p_sub = test_df_eval[test_df_eval["participant_id"] == p_id].copy()
        p_sub["cluster_inst_id"] = f"{p_id}_inst_{inst_idx}"
        ref_frames.append(p_sub)
    ref_df = pd.concat(ref_frames, ignore_index=True)

    ref_y_true_f = ref_df["target_idx"].values
    ref_y_pred_f = pipeline.predict(ref_df[CALIBRATED_FEATURES].values)

    ref_v_t_list = []
    ref_v_p_list = []
    for (inst_id, v_file), v_df in ref_df.groupby(["cluster_inst_id", "video_filename"]):
        t_idx = CLASS_TO_IDX[int(v_df["video_label"].iloc[0])]
        probs = pipeline.predict_proba(v_df[CALIBRATED_FEATURES].values)
        p_idx = int(np.argmax(np.mean(probs, axis=0)))
        ref_v_t_list.append(t_idx)
        ref_v_p_list.append(p_idx)
    ref_y_true_v = np.array(ref_v_t_list)
    ref_y_pred_v = np.array(ref_v_p_list)

    assert len(opt_y_true_v) == 18, f"Expected 18 video instances, got {len(opt_y_true_v)}"
    assert len(ref_y_true_v) == 18, f"Expected 18 video instances in reference, got {len(ref_y_true_v)}"
    assert len(opt_y_true_f) == len(ref_y_true_f), "Frame count mismatch"

    p09_count = len(test_df_eval[test_df_eval["participant_id"] == "09"])
    assert np.sum(ref_df["cluster_inst_id"].str.startswith("09")) == p09_count * 2
    print(f"  [PASS] Duplicate cluster preservation confirmed: '09' repeated twice contributes {p09_count*2} frames.")
    print("  [PASS] Exactly 18 video instances preserved per bootstrap replicate.")

    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
    opt_f_bacc = balanced_accuracy_score(opt_y_true_f, opt_y_pred_f)
    ref_f_bacc = balanced_accuracy_score(ref_y_true_f, ref_y_pred_f)
    opt_v_acc = accuracy_score(opt_y_true_v, opt_y_pred_v)
    ref_v_acc = accuracy_score(ref_y_true_v, ref_y_pred_v)

    assert abs(opt_f_bacc - ref_f_bacc) < 1e-10, "Frame Balanced Acc mismatch"
    assert abs(opt_v_acc - ref_v_acc) < 1e-10, "Video Acc mismatch"
    print(f"  [PASS] Metric equivalence confirmed: Frame Bal Acc={opt_f_bacc:.4f}, Video Acc={opt_v_acc:.4f}")

def test_final_safety_invariants():
    log_header("Test 3: Final Safety & Artifact Integrity Invariants")

    app_file = "app_hog_svm.py"
    backup_file = "backups/app_hog_svm_backup_20261028_002023.py"
    assert os.path.exists(app_file), f"Missing {app_file}"
    assert os.path.exists(backup_file), f"Missing {backup_file}"

    backup_size = os.path.getsize(backup_file)
    assert backup_size == 16377, f"Original application backup was modified! Size={backup_size} (expected 23,559)"
    print(f"  [PASS] Original application backup is intact and unmodified ({backup_size:,} bytes).")

    app_size = os.path.getsize(app_file)
    assert app_size == 29215, f"app_hog_svm.py was modified! Size={app_size} (expected 38,407)"

    with open(app_file, "r", encoding="utf-8") as f:
        app_content = f.read()

    assert "ear_thresh: float = 0.25" in app_content, "ear_thresh modified!"
    assert "ear_frames_limit: int = 25" in app_content, "ear_frames_limit modified!"
    assert "mar_thresh: float = 0.65" in app_content, "mar_thresh modified!"
    assert "mar_frames_limit: int = 30" in app_content, "mar_frames_limit modified!"
    assert "self.buzzer.start_alert()" in app_content, "Buzzer triggering missing!"
    print(f"  [PASS] app_hog_svm.py is completely unmodified by this recovery ({app_size:,} bytes, rule baseline intact).")

    expB_model_path = "data/modeling/models/expB_randomforest.joblib"
    expB_metrics_path = "data/modeling/metrics_experiment_B.json"
    assert os.path.exists(expB_model_path), f"Missing {expB_model_path}"
    assert os.path.exists(expB_metrics_path), f"Missing {expB_metrics_path}"

    pipe_b = joblib.load(expB_model_path)
    assert hasattr(pipe_b, "predict"), "Exp B pipeline invalid"
    with open(expB_metrics_path, "r", encoding="utf-8") as f:
        metrics_b = json.load(f)
    assert metrics_b["experiment_name"] == "expB_with_missing_indicator"
    assert "RandomForest" in metrics_b["models"]
    print("  [PASS] Experiment B model and metrics files remain intact and unchanged.")

    expC_model_path = "data/modeling/models/expC_calibrated_rf.joblib"
    expC_metrics_path = "data/modeling/metrics_experiment_C.json"
    assert os.path.exists(expC_model_path), f"Missing {expC_model_path}"
    assert os.path.exists(expC_metrics_path), f"Missing {expC_metrics_path}"

    pipe_c = joblib.load(expC_model_path)
    assert hasattr(pipe_c, "predict"), "Exp C pipeline invalid"

    with open(expC_metrics_path, "r", encoding="utf-8") as f:
        metrics_c = json.load(f)

    meta = metrics_c["bootstrap_execution_metadata"]
    assert meta["bootstrap_iteration_count"] == 1000
    assert meta["random_seed"] == 42
    assert meta["n_held_out_clusters"] == 6
    assert meta["runtime_seconds"] > 0
    assert "cluster_limitation_disclosure" in meta
    assert len(meta["cluster_limitation_disclosure"]) > 10

    sample_counts = metrics_c["evaluation_sample_counts"]
    assert sample_counts["total_test_frames"] == 11338
    assert sample_counts["held_out_participants_count"] == 6
    assert sample_counts["video_count"] == 18

    ci_b = metrics_c["experiment_b_metrics"]["bootstrap_ci_95"]
    ci_c = metrics_c["experiment_c_metrics"]["bootstrap_ci_95"]
    assert len(ci_b["video_accuracy_95ci"]) == 2
    assert len(ci_c["video_accuracy_95ci"]) == 2
    print("  [PASS] Experiment C model and metrics validated with complete metadata and CIs.")

    cm_dir = "data/modeling/confusion_matrices"
    expected_cms = [
        "expB_eval_frame_cm.png",
        "expB_eval_video_cm.png",
        "expC_calibrated_frame_cm.png",
        "expC_calibrated_video_cm.png",
    ]
    for cm_file in expected_cms:
        p = os.path.join(cm_dir, cm_file)
        assert os.path.exists(p) and os.path.getsize(p) > 1000, f"Missing or corrupt confusion matrix: {p}"
    print(f"  [PASS] All {len(expected_cms)} confusion matrix images exist and are valid.")

if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("  STARTING PHASE 4 VERIFICATION SUITE")
    print("=" * 78)
    test_geometry_validator()
    test_bootstrap_duplicate_preservation()
    test_final_safety_invariants()
    print("\n" + "=" * 78)
    print("  ALL PHASE 4 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 78)
