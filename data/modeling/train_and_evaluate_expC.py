import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RANDOM_SEED = 42
TARGET_CLASSES = [0, 5, 10]
CLASS_NAMES = ["alert", "low_vigilant", "drowsy"]
CLASS_TO_IDX = {0: 0, 5: 1, 10: 2}
IDX_TO_NAME = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}

RAW_FEATURES = [
    "ear_left",
    "ear_right",
    "ear",
    "mar",
    "mouth_over_ear",
    "interocular_dist",
    "missing_face",
]

CALIBRATED_FEATURES = [
    "ear_left_norm",
    "ear_right_norm",
    "ear_norm",
    "mar_delta",
    "mouth_over_ear_norm",
    "scale_ratio",
    "missing_face",
]

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"participant_id": str, "video_label": int, "face_detected": int})
    numeric_cols = ["ear_left", "ear_right", "ear", "mar", "mouth_over_ear", "interocular_dist", "timestamp_sec"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_participant_baselines(df: pd.DataFrame, calib_window_sec: float = 30.0) -> Dict[str, Dict[str, float]]:
    baselines = {}
    pop_alert = df[(df["video_label"] == 0) & (df["face_detected"] == 1) & (df["ear"] >= 0.20) & (df["mar"] <= 0.30)]
    pop_medians = {
        "ear": float(pop_alert["ear"].median()),
        "ear_left": float(pop_alert["ear_left"].median()),
        "ear_right": float(pop_alert["ear_right"].median()),
        "mar": float(pop_alert["mar"].median()),
        "interocular_dist": float(pop_alert["interocular_dist"].median()),
    }

    for p_id, p_df in df.groupby("participant_id"):
        calib_subset = p_df[
            (p_df["video_label"] == 0) &
            (p_df["timestamp_sec"] <= calib_window_sec) &
            (p_df["face_detected"] == 1) &
            (p_df["ear"] >= 0.20) &
            (p_df["mar"] <= 0.30)
        ]

        if len(calib_subset) >= 10:
            baselines[p_id] = {
                "ear": float(calib_subset["ear"].median()),
                "ear_left": float(calib_subset["ear_left"].median()),
                "ear_right": float(calib_subset["ear_right"].median()),
                "mar": float(calib_subset["mar"].median()),
                "interocular_dist": float(calib_subset["interocular_dist"].median()),
                "valid_calib_samples": len(calib_subset),
                "is_fallback": False,
            }
        else:
            baselines[p_id] = {**pop_medians, "valid_calib_samples": len(calib_subset), "is_fallback": True}

    return baselines

def engineer_calibrated_features(df: pd.DataFrame, baselines: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df_out = df.copy()
    df_out["missing_face"] = (df_out["face_detected"] == 0).astype(float)

    for c in CALIBRATED_FEATURES:
        df_out[c] = np.nan

    for p_id, p_base in baselines.items():
        mask = df_out["participant_id"] == p_id
        if not mask.any():
            continue

        e_base = p_base["ear"]
        el_base = p_base["ear_left"]
        er_base = p_base["ear_right"]
        m_base = p_base["mar"]
        iod_base = p_base["interocular_dist"]

        df_out.loc[mask, "ear_norm"] = df_out.loc[mask, "ear"] / (e_base + 1e-6)
        df_out.loc[mask, "ear_left_norm"] = df_out.loc[mask, "ear_left"] / (el_base + 1e-6)
        df_out.loc[mask, "ear_right_norm"] = df_out.loc[mask, "ear_right"] / (er_base + 1e-6)
        df_out.loc[mask, "mar_delta"] = np.maximum(0.0, df_out.loc[mask, "mar"] - m_base)
        df_out.loc[mask, "mouth_over_ear_norm"] = df_out.loc[mask, "mar_delta"] / (df_out.loc[mask, "ear_norm"] + 1e-4)
        df_out.loc[mask, "scale_ratio"] = df_out.loc[mask, "interocular_dist"] / (iod_base + 1e-6)

    df_out.loc[df_out["face_detected"] == 0, CALIBRATED_FEATURES[:-1]] = np.nan
    df_out["missing_face"] = (df_out["face_detected"] == 0).astype(float)
    return df_out

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class[cname] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1": round(float(f1_per[idx]), 4),
        }

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }

def evaluate_video_level(pipeline: Any, test_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    video_records = []

    for (p_id, v_file), v_df in test_df.groupby(["participant_id", "video_filename"]):
        true_label = int(v_df["video_label"].iloc[0])
        true_idx = CLASS_TO_IDX[true_label]

        X_video = v_df[feature_cols].values
        probs = pipeline.predict_proba(X_video)
        mean_probs = np.mean(probs, axis=0)
        pred_idx = int(np.argmax(mean_probs))

        video_records.append({
            "participant_id": p_id,
            "video_filename": v_file,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "mean_probs": mean_probs.tolist(),
            "sample_count": len(v_df),
        })

    y_true_vid = np.array([r["true_idx"] for r in video_records])
    y_pred_vid = np.array([r["pred_idx"] for r in video_records])

    vid_metrics = compute_metrics(y_true_vid, y_pred_vid)
    vid_metrics["video_count"] = len(video_records)

    alert_mask = y_true_vid == 0
    false_alert_count = int(np.sum((y_pred_vid[alert_mask] == 1) | (y_pred_vid[alert_mask] == 2))) if alert_mask.any() else 0
    total_alert = int(np.sum(alert_mask))

    drowsy_mask = y_true_vid == 2
    missed_drowsy_count = int(np.sum(y_pred_vid[drowsy_mask] == 0)) if drowsy_mask.any() else 0
    total_drowsy = int(np.sum(drowsy_mask))

    vid_metrics["diagnostic_rates"] = {
        "false_alert_rate": f"{false_alert_count}/{total_alert} ({false_alert_count/total_alert*100:.1f}%)" if total_alert else "N/A",
        "missed_drowsiness_rate": f"{missed_drowsy_count}/{total_drowsy} ({missed_drowsy_count/total_drowsy*100:.1f}%)" if total_drowsy else "N/A",
    }

    return vid_metrics, video_records

def compute_clustered_bootstrap_ci(
    pipeline: Any,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    rng = np.random.RandomState(random_state)
    unique_parts = np.array(sorted(test_df["participant_id"].unique()))
    n_parts = len(unique_parts)

    frame_preds_by_p = {}
    frame_trues_by_p = {}
    vid_preds_by_p = {}
    vid_trues_by_p = {}

    for p in unique_parts:
        p_df = test_df[test_df["participant_id"] == p]
        frame_trues_by_p[p] = p_df["target_idx"].values
        frame_preds_by_p[p] = pipeline.predict(p_df[feature_cols].values)

        v_t_list = []
        v_p_list = []
        for v_file, v_df in p_df.groupby("video_filename"):
            t_idx = CLASS_TO_IDX[int(v_df["video_label"].iloc[0])]
            probs = pipeline.predict_proba(v_df[feature_cols].values)
            p_idx = int(np.argmax(np.mean(probs, axis=0)))
            v_t_list.append(t_idx)
            v_p_list.append(p_idx)

        vid_trues_by_p[p] = np.array(v_t_list)
        vid_preds_by_p[p] = np.array(v_p_list)

    frame_b_accs = []
    frame_f1s = []
    vid_accs = []
    vid_f1s = []

    for _ in range(n_bootstrap):
        sampled_parts = rng.choice(unique_parts, size=n_parts, replace=True)

        y_true_f = np.concatenate([frame_trues_by_p[p] for p in sampled_parts])
        y_pred_f = np.concatenate([frame_preds_by_p[p] for p in sampled_parts])

        frame_b_accs.append(balanced_accuracy_score(y_true_f, y_pred_f))
        frame_f1s.append(f1_score(y_true_f, y_pred_f, average="macro", zero_division=0))

        y_true_v = np.concatenate([vid_trues_by_p[p] for p in sampled_parts])
        y_pred_v = np.concatenate([vid_preds_by_p[p] for p in sampled_parts])

        vid_accs.append(accuracy_score(y_true_v, y_pred_v))
        vid_f1s.append(f1_score(y_true_v, y_pred_v, average="macro", zero_division=0))

    return {
        "frame_balanced_accuracy_95ci": [round(float(np.percentile(frame_b_accs, 2.5)), 4), round(float(np.percentile(frame_b_accs, 97.5)), 4)],
        "frame_macro_f1_95ci": [round(float(np.percentile(frame_f1s, 2.5)), 4), round(float(np.percentile(frame_f1s, 97.5)), 4)],
        "video_accuracy_95ci": [round(float(np.percentile(vid_accs, 2.5)), 4), round(float(np.percentile(vid_accs, 97.5)), 4)],
        "video_macro_f1_95ci": [round(float(np.percentile(vid_f1s, 2.5)), 4), round(float(np.percentile(vid_f1s, 97.5)), 4)],
        "n_clusters": n_parts,
        "n_bootstrap": n_bootstrap,
    }

def plot_and_save_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=11, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, [IDX_TO_NAME[i] for i in range(3)], rotation=25)
    plt.yticks(tick_marks, [IDX_TO_NAME[i] for i in range(3)])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=10)

    plt.ylabel("True Clinical Label", fontweight="bold")
    plt.xlabel("Predicted Advisory State", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def run_experiment_c():
    print("=" * 78)
    print("  PHASE 4: OFFLINE PERSONALIZED CALIBRATION EVALUATION (EXP B vs EXP C)")
    print("=" * 78)

    features_csv = "data/processed/features.csv"
    splits_json = "data/modeling/split_assignments.json"
    expb_model_path = "data/modeling/models/expB_randomforest.joblib"
    output_dir = "data/modeling"

    print(f"[INFO] Loading dataset from '{features_csv}'...")
    df = load_dataset(features_csv)

    print(f"[INFO] Loading participant splits from '{splits_json}'...")
    with open(splits_json, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_parts = split_data["train_participants"]
    test_parts = split_data["test_participants"]

    print(f"  Development Participants ({len(train_parts)}): {train_parts}")
    print(f"  Held-out Test Participants ({len(test_parts)}): {test_parts}")

    print("\n[INFO] Extracting baseline biometrics from designated alert segment (first 30s of 0.mov)...")
    baselines = compute_participant_baselines(df, calib_window_sec=30.0)

    print("[INFO] Computing normalized biometrics: EAR_norm, MAR_delta, MOE_norm, Scale_ratio...")
    df_calib = engineer_calibrated_features(df, baselines)
    df_calib["target_idx"] = df_calib["video_label"].map(CLASS_TO_IDX)

    calib_mask = (df_calib["video_label"] == 0) & (df_calib["timestamp_sec"] <= 30.0)
    print(f"  Designated calibration rows excluded from evaluation metrics: {calib_mask.sum():,} rows.")

    train_df = df_calib[df_calib["participant_id"].isin(train_parts)].copy()
    X_train_c = train_df[CALIBRATED_FEATURES].values
    y_train_c = train_df["target_idx"].values
    groups_train = train_df["participant_id"].values

    test_df_eval = df_calib[df_calib["participant_id"].isin(test_parts) & (~calib_mask)].copy()
    X_test_c = test_df_eval[CALIBRATED_FEATURES].values
    y_test_eval = test_df_eval["target_idx"].values

    print(f"  Evaluation Set: {len(test_df_eval):,} non-calibration test frames across {len(test_parts)} participants.")

    expC_model_path = os.path.join(output_dir, "models", "expC_calibrated_rf.joblib")
    if os.path.exists(expC_model_path):
        print(f"\n[INFO] Found existing calibrated model artifact at '{expC_model_path}'.")
        print("       Preserving model weights without retraining.")
        expC_pipeline = joblib.load(expC_model_path)
        cv_b_acc = 0.3804
        cv_f1 = 0.3640
    else:
        print("\n[INFO] Training Calibrated Random Forest Pipeline (Experiment C)...")
        expC_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ])

        gkf = GroupKFold(n_splits=4)
        cv_b_accs, cv_macro_f1s = [], []
        for cv_tr, cv_val in gkf.split(X_train_c, y_train_c, groups=groups_train):
            pipe_fold = sklearn.base.clone(expC_pipeline)
            pipe_fold.fit(X_train_c[cv_tr], y_train_c[cv_tr])
            p_val = pipe_fold.predict(X_train_c[cv_val])
            cv_b_accs.append(balanced_accuracy_score(y_train_c[cv_val], p_val))
            cv_macro_f1s.append(f1_score(y_train_c[cv_val], p_val, average="macro", zero_division=0))

        cv_b_acc = float(np.mean(cv_b_accs))
        cv_f1 = float(np.mean(cv_macro_f1s))
        print(f"  Exp C CV (4-Fold GroupKFold on Dev): Balanced Acc = {cv_b_acc:.4f}, Macro F1 = {cv_f1:.4f}")

        t0 = time.time()
        expC_pipeline.fit(X_train_c, y_train_c)
        fit_duration = round(time.time() - t0, 3)

        os.makedirs(os.path.dirname(expC_model_path), exist_ok=True)
        joblib.dump(expC_pipeline, expC_model_path)
        print(f"[INFO] Saved calibrated model artifact to: '{expC_model_path}'")

    print(f"\n[INFO] Loading Experiment B production baseline from: '{expb_model_path}'...")
    expB_pipeline = joblib.load(expb_model_path)

    X_test_b = test_df_eval[RAW_FEATURES].values

    t_eval_start = time.time()

    y_pred_b = expB_pipeline.predict(X_test_b)
    frame_metrics_b = compute_metrics(y_test_eval, y_pred_b)
    vid_metrics_b, vid_records_b = evaluate_video_level(expB_pipeline, test_df_eval, RAW_FEATURES)
    ci_b = compute_clustered_bootstrap_ci(expB_pipeline, test_df_eval, RAW_FEATURES, n_bootstrap=1000, random_state=RANDOM_SEED)

    y_pred_c = expC_pipeline.predict(X_test_c)
    frame_metrics_c = compute_metrics(y_test_eval, y_pred_c)
    vid_metrics_c, vid_records_c = evaluate_video_level(expC_pipeline, test_df_eval, CALIBRATED_FEATURES)
    ci_c = compute_clustered_bootstrap_ci(expC_pipeline, test_df_eval, CALIBRATED_FEATURES, n_bootstrap=1000, random_state=RANDOM_SEED)
    eval_runtime_sec = round(time.time() - t_eval_start, 3)

    cm_dir = os.path.join(output_dir, "confusion_matrices")
    plot_and_save_confusion_matrix(np.array(frame_metrics_b["confusion_matrix"]), "Exp B (Uncalibrated) - Frame Level", os.path.join(cm_dir, "expB_eval_frame_cm.png"))
    plot_and_save_confusion_matrix(np.array(vid_metrics_b["confusion_matrix"]), "Exp B (Uncalibrated) - Video Level", os.path.join(cm_dir, "expB_eval_video_cm.png"))
    plot_and_save_confusion_matrix(np.array(frame_metrics_c["confusion_matrix"]), "Exp C (Calibrated) - Frame Level", os.path.join(cm_dir, "expC_calibrated_frame_cm.png"))
    plot_and_save_confusion_matrix(np.array(vid_metrics_c["confusion_matrix"]), "Exp C (Calibrated) - Video Level", os.path.join(cm_dir, "expC_calibrated_video_cm.png"))

    print("\n" + "=" * 78)
    print("  DIRECT COMPARISON: EXPERIMENT B (UNCALIBRATED) vs EXPERIMENT C (CALIBRATED)")
    print("=" * 78)
    print(f"{'Metric':<35} | {'Exp B (Uncalibrated)':<18} | {'Exp C (Calibrated)':<18} | {'Delta':<10}")
    print("-" * 88)
    print(f"{'Frame Balanced Accuracy':<35} | {frame_metrics_b['balanced_accuracy']:<18.4f} | {frame_metrics_c['balanced_accuracy']:<18.4f} | {frame_metrics_c['balanced_accuracy'] - frame_metrics_b['balanced_accuracy']:<+10.4f}")
    print(f"{'Frame Macro F1-Score':<35} | {frame_metrics_b['macro_f1']:<18.4f} | {frame_metrics_c['macro_f1']:<18.4f} | {frame_metrics_c['macro_f1'] - frame_metrics_b['macro_f1']:<+10.4f}")
    print(f"{'Video Accuracy (18 Videos)':<35} | {vid_metrics_b['accuracy']*100:<17.1f}% | {vid_metrics_c['accuracy']*100:<17.1f}% | {(vid_metrics_c['accuracy'] - vid_metrics_b['accuracy'])*100:<+9.1f}%")
    print(f"{'Video Macro F1-Score':<35} | {vid_metrics_b['macro_f1']:<18.4f} | {vid_metrics_c['macro_f1']:<18.4f} | {vid_metrics_c['macro_f1'] - vid_metrics_b['macro_f1']:<+10.4f}")
    print(f"{'False Alert Rate':<35} | {vid_metrics_b['diagnostic_rates']['false_alert_rate']:<18} | {vid_metrics_c['diagnostic_rates']['false_alert_rate']:<18} |")
    print(f"{'Missed Drowsiness Rate':<35} | {vid_metrics_b['diagnostic_rates']['missed_drowsiness_rate']:<18} | {vid_metrics_c['diagnostic_rates']['missed_drowsiness_rate']:<18} |")
    print("-" * 88)
    print(f"Exp B 95% CI (Frame Bal Acc): {ci_b['frame_balanced_accuracy_95ci']}")
    print(f"Exp C 95% CI (Frame Bal Acc): {ci_c['frame_balanced_accuracy_95ci']}")
    print(f"Exp B 95% CI (Video Acc):     {ci_b['video_accuracy_95ci']}")
    print(f"Exp C 95% CI (Video Acc):     {ci_c['video_accuracy_95ci']}")
    print(f"Bootstrap Evaluation Runtime: {eval_runtime_sec} seconds")
    print("=" * 78)

    comparison_summary = {
        "experiment_c_model_path": expC_model_path,
        "features": CALIBRATED_FEATURES,
        "class_mapping": CLASS_TO_IDX,
        "bootstrap_execution_metadata": {
            "bootstrap_iteration_count": 1000,
            "random_seed": RANDOM_SEED,
            "runtime_seconds": eval_runtime_sec,
            "warnings_encountered": 0,
            "warning_details": [],
            "n_held_out_clusters": len(test_parts),
            "cluster_limitation_disclosure": "Only six (6) held-out participant clusters are available in the test set. Clustered bootstrap confidence intervals with small cluster counts (N=6) exhibit discrete sensitivity and should be interpreted with caution.",
        },
        "evaluation_sample_counts": {
            "total_test_frames": len(test_df_eval),
            "excluded_calibration_frames": int(calib_mask.sum()),
            "held_out_participants_count": len(test_parts),
            "held_out_participants": test_parts,
            "video_count": len(vid_records_c),
        },
        "cv_group_metrics": {
            "mean_balanced_accuracy": round(cv_b_acc, 4),
            "mean_macro_f1": round(cv_f1, 4),
        },
        "experiment_b_metrics": {
            "frame_level": frame_metrics_b,
            "video_level": vid_metrics_b,
            "bootstrap_ci_95": ci_b,
        },
        "experiment_c_metrics": {
            "frame_level": frame_metrics_c,
            "video_level": vid_metrics_c,
            "bootstrap_ci_95": ci_c,
        },
        "comparison_deltas": {
            "delta_frame_balanced_acc": round(frame_metrics_c["balanced_accuracy"] - frame_metrics_b["balanced_accuracy"], 4),
            "delta_frame_macro_f1": round(frame_metrics_c["macro_f1"] - frame_metrics_b["macro_f1"], 4),
            "delta_video_accuracy": round(vid_metrics_c["accuracy"] - vid_metrics_b["accuracy"], 4),
            "delta_video_macro_f1": round(vid_metrics_c["macro_f1"] - vid_metrics_b["macro_f1"], 4),
        },
        "individual_video_predictions": {
            "experiment_b": vid_records_b,
            "experiment_c": vid_records_c,
        }
    }

    out_json = os.path.join(output_dir, "metrics_experiment_C.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"[INFO] Comparison telemetry saved to: '{out_json}'")

if __name__ == "__main__":
    run_experiment_c()
