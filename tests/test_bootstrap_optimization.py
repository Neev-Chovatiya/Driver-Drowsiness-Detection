import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.modeling.train_and_evaluate_expC import (
    CLASS_TO_IDX,
    compute_metrics,
    evaluate_video_level,
)

def run_bootstrap_validation():
    print("=" * 74)
    print("  STEP 3: BOOTSTRAP OPTIMIZATION & EQUIVALENCE VALIDATION")
    print("=" * 74)

    model_path = "data/modeling/models/expC_calibrated_rf.joblib"
    print(f"[TEST 1] Loading calibrated model from '{model_path}'...")
    pipeline = joblib.load(model_path)
    assert pipeline is not None

    from data.modeling.train_and_evaluate_expC import (
        load_dataset,
        compute_participant_baselines,
        engineer_calibrated_features,
        CALIBRATED_FEATURES
    )

    df = load_dataset("data/processed/features.csv")
    with open("data/modeling/split_assignments.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_parts = splits["test_participants"]

    baselines = compute_participant_baselines(df, calib_window_sec=30.0)
    df_calib = engineer_calibrated_features(df, baselines)
    df_calib["target_idx"] = df_calib["video_label"].map(CLASS_TO_IDX)

    calib_mask = (df_calib["video_label"] == 0) & (df_calib["timestamp_sec"] <= 30.0)
    test_df_eval = df_calib[df_calib["participant_id"].isin(test_parts) & (~calib_mask)].copy()

    unique_parts = np.array(sorted(test_df_eval["participant_id"].unique()))
    n_parts = len(unique_parts)
    print(f"  Test participants: {unique_parts.tolist()} ({n_parts} clusters)")
    print(f"  Test evaluation frames: {len(test_df_eval)}")

    def reference_bootstrap_replicate(pipeline, test_df, sampled_parts, feature_cols):
        replicate_frames = []
        for instance_idx, p_id in enumerate(sampled_parts):
            p_sub = test_df[test_df["participant_id"] == p_id].copy()
            p_sub["cluster_instance_id"] = f"{p_id}_inst_{instance_idx}"
            replicate_frames.append(p_sub)
        rep_df = pd.concat(replicate_frames, ignore_index=True)

        y_true_f = rep_df["target_idx"].values
        y_pred_f = pipeline.predict(rep_df[feature_cols].values)
        f_bacc = balanced_accuracy_score(y_true_f, y_pred_f)
        f_f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)

        video_preds = []
        video_trues = []
        for (inst_id, v_file), v_df in rep_df.groupby(["cluster_instance_id", "video_filename"]):
            true_label = CLASS_TO_IDX[int(v_df["video_label"].iloc[0])]
            X_video = v_df[feature_cols].values
            probs = pipeline.predict_proba(X_video)
            pred_label = int(np.argmax(np.mean(probs, axis=0)))
            video_trues.append(true_label)
            video_preds.append(pred_label)

        v_acc = accuracy_score(video_trues, video_preds)
        v_f1 = f1_score(video_trues, video_preds, average="macro", zero_division=0)

        return {
            "frame_bal_acc": f_bacc,
            "frame_macro_f1": f_f1,
            "video_acc": v_acc,
            "video_macro_f1": v_f1,
            "total_video_count": len(video_trues),
            "total_frame_count": len(y_true_f),
        }

    print("\n[TEST 2] Precomputing predictions for optimized method...")
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

    def optimized_bootstrap_replicate(sampled_parts):
        y_true_f = np.concatenate([frame_trues_by_p[p] for p in sampled_parts])
        y_pred_f = np.concatenate([frame_preds_by_p[p] for p in sampled_parts])

        f_bacc = balanced_accuracy_score(y_true_f, y_pred_f)
        f_f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)

        y_true_v = np.concatenate([vid_trues_by_p[p] for p in sampled_parts])
        y_pred_v = np.concatenate([vid_preds_by_p[p] for p in sampled_parts])

        v_acc = accuracy_score(y_true_v, y_pred_v)
        v_f1 = f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)

        return {
            "frame_bal_acc": f_bacc,
            "frame_macro_f1": f_f1,
            "video_acc": v_acc,
            "video_macro_f1": v_f1,
            "total_video_count": len(y_true_v),
            "total_frame_count": len(y_true_f),
        }

    print("\n[TEST 3] Comparing Reference vs Optimized on 10 fixed replicates with duplicate clusters...")
    rng = np.random.RandomState(42)

    for rep_i in range(1, 11):
        sampled = rng.choice(unique_parts, size=n_parts, replace=True)

        has_duplicates = len(sampled) != len(set(sampled))

        ref_res = reference_bootstrap_replicate(pipeline, test_df_eval, sampled, CALIBRATED_FEATURES)
        opt_res = optimized_bootstrap_replicate(sampled)

        assert ref_res["total_video_count"] == 18, f"Expected 18 video instances, got {ref_res['total_video_count']}"
        assert opt_res["total_video_count"] == 18, f"Expected 18 video instances, got {opt_res['total_video_count']}"
        assert ref_res["total_frame_count"] == opt_res["total_frame_count"]

        assert abs(ref_res["frame_bal_acc"] - opt_res["frame_bal_acc"]) < 1e-10, "Frame Balanced Acc mismatch"
        assert abs(ref_res["frame_macro_f1"] - opt_res["frame_macro_f1"]) < 1e-10, "Frame Macro F1 mismatch"
        assert abs(ref_res["video_acc"] - opt_res["video_acc"]) < 1e-10, "Video Acc mismatch"
        assert abs(ref_res["video_macro_f1"] - opt_res["video_macro_f1"]) < 1e-10, "Video Macro F1 mismatch"

        print(f"  Replicate {rep_i:02d} (Sampled: {sampled.tolist()}, Duplicates={has_duplicates}):")
        print(f"    Frame Bal Acc: {opt_res['frame_bal_acc']:.4f} == {ref_res['frame_bal_acc']:.4f} (MATCH)")
        print(f"    Video Acc:     {opt_res['video_acc']:.4f} == {ref_res['video_acc']:.4f} (MATCH, 18 videos)")

    print("\n[VALIDATION SUCCESSFUL] Optimized bootstrap produces 100% identical results to reference.")
    print("Exact numerical equivalence verified.")

if __name__ == "__main__":
    import json
    run_bootstrap_validation()
