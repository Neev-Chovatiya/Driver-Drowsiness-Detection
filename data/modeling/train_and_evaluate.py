import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RANDOM_SEED = 42
TARGET_CLASSES = [0, 5, 10]
CLASS_NAMES = ["alert", "low_vigilant", "drowsy"]
CLASS_TO_IDX = {0: 0, 5: 1, 10: 2}
IDX_TO_CLASS = {0: 0, 1: 5, 2: 10}
IDX_TO_NAME = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}

CORE_FEATURES = [
    "ear_left",
    "ear_right",
    "ear",
    "mar",
    "mouth_over_ear",
    "interocular_dist",
]

def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Features file not found at: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"participant_id": str, "video_label": int, "face_detected": int})

    for col in CORE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def create_participant_split(df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_SEED) -> Tuple[List[str], List[str]]:
    unique_participants = np.array(sorted(df["participant_id"].unique()))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(unique_participants, groups=unique_participants))

    train_parts = sorted(unique_participants[train_idx].tolist())
    test_parts = sorted(unique_participants[test_idx].tolist())

    return train_parts, test_parts

def plot_and_save_confusion_matrix(cm: np.ndarray, title: str, output_path: str, normalize: bool = True):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = [IDX_TO_NAME[i] for i in range(len(TARGET_CLASSES))]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Ground Truth (UTA-RLDD)", fontsize=10)
    ax.set_xlabel("Predicted State", fontsize=10)

    thresh = 0.5 if normalize else cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_text = f"{cm_display[i, j]:.2f}\n(n={cm[i, j]})" if normalize else f"{cm[i, j]}"
            color = "white" if (cm_display[i, j] if normalize else cm[i, j]) > thresh else "black"
            ax.text(j, i, val_text, ha="center", va="center", color=color, fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_p = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            "alert": {"precision": round(float(per_class_p[0]), 4), "recall": round(float(per_class_r[0]), 4), "f1": round(float(per_class_f1[0]), 4)},
            "low_vigilant": {"precision": round(float(per_class_p[1]), 4), "recall": round(float(per_class_r[1]), 4), "f1": round(float(per_class_f1[1]), 4)},
            "drowsy": {"precision": round(float(per_class_p[2]), 4), "recall": round(float(per_class_r[2]), 4), "f1": round(float(per_class_f1[2]), 4)},
        },
        "confusion_matrix": cm,
    }

def evaluate_video_level(
    model: Any,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Any]:
    video_records = []
    grouped = test_df.groupby(["participant_id", "video_filename", "video_label"])

    for (pid, vname, vlabel), group in grouped:
        X_group = group[feature_cols].values
        probs = model.predict_proba(X_group)
        mean_prob = probs.mean(axis=0)
        pred_idx = int(np.argmax(mean_prob))
        true_idx = CLASS_TO_IDX[vlabel]

        video_records.append({
            "participant_id": pid,
            "video_filename": vname,
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "probs": mean_prob.tolist(),
        })

    y_true_vid = np.array([r["true_idx"] for r in video_records])
    y_pred_vid = np.array([r["pred_idx"] for r in video_records])

    vid_metrics = compute_metrics(y_true_vid, y_pred_vid)
    vid_metrics["video_count"] = len(video_records)
    return vid_metrics

def run_experiment(
    df: pd.DataFrame,
    train_parts: List[str],
    test_parts: List[str],
    experiment_name: str,
    include_missing_face_flag: bool = False,
    output_dir: str = r"F:\DrowsinessDetector\data\modeling",
) -> Dict[str, Any]:
    print(f"\n{'='*74}")
    print(f"  RUNNING: {experiment_name.upper()}")
    print(f"{'='*74}")

    if not include_missing_face_flag:
        exp_df = df[df["face_detected"] == 1].copy()
        feature_cols = CORE_FEATURES
    else:
        exp_df = df.copy()
        exp_df["missing_face"] = (exp_df["face_detected"] == 0).astype(float)
        feature_cols = CORE_FEATURES + ["missing_face"]

    exp_df["target_idx"] = exp_df["video_label"].map(CLASS_TO_IDX)

    train_df = exp_df[exp_df["participant_id"].isin(train_parts)].copy()
    test_df = exp_df[exp_df["participant_id"].isin(test_parts)].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["target_idx"].values
    groups_train = train_df["participant_id"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["target_idx"].values

    print(f"Dataset Rows: Total={len(exp_df)} | Train={len(train_df)} | Test={len(test_df)}")
    print(f"Features: {feature_cols}")

    models = {}

    models["DummyClassifier"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])

    models["LogisticRegression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])

    models["RandomForest"] = Pipeline([
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

    models["LinearSVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(dual=False, C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=2000),
            cv=3,
        )),
    ])

    results = {
        "experiment_name": experiment_name,
        "features": feature_cols,
        "sample_counts": {
            "total_samples": len(exp_df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_participants": len(train_parts),
            "test_participants": len(test_parts),
        },
        "models": {},
    }

    gkf = GroupKFold(n_splits=4)

    for model_name, pipeline in models.items():
        print(f"\n[MODEL] Evaluating {model_name}...")
        t0 = time.time()

        cv_bal_accs = []
        cv_macro_f1s = []

        for fold_idx, (cv_tr, cv_val) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
            pipe_cv = sklearn.base.clone(pipeline)
            pipe_cv.fit(X_train[cv_tr], y_train[cv_tr])
            val_preds = pipe_cv.predict(X_train[cv_val])
            b_acc = balanced_accuracy_score(y_train[cv_val], val_preds)
            m_f1 = f1_score(y_train[cv_val], val_preds, average="macro", zero_division=0)
            cv_bal_accs.append(b_acc)
            cv_macro_f1s.append(m_f1)

        mean_cv_bal_acc = float(np.mean(cv_bal_accs))
        std_cv_bal_acc = float(np.std(cv_bal_accs))
        mean_cv_macro_f1 = float(np.mean(cv_macro_f1s))

        print(f"  CV (4-Fold GroupKFold on Dev Participants):")
        print(f"    Balanced Accuracy: {mean_cv_bal_acc:.4f} (+/- {std_cv_bal_acc:.4f})")
        print(f"    Macro F1:         {mean_cv_macro_f1:.4f}")

        pipeline.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 3)

        y_test_pred = pipeline.predict(X_test)
        frame_metrics = compute_metrics(y_test, y_test_pred)

        print(f"  Held-Out Test (Frame-Level):")
        print(f"    Accuracy:          {frame_metrics['accuracy']:.4f}")
        print(f"    Balanced Accuracy: {frame_metrics['balanced_accuracy']:.4f}")
        print(f"    Macro F1:          {frame_metrics['macro_f1']:.4f}")

        video_metrics = evaluate_video_level(pipeline, test_df, feature_cols)
        print(f"  Held-Out Test (Video-Level Aggregated):")
        print(f"    Accuracy:          {video_metrics['accuracy']:.4f}")
        print(f"    Balanced Accuracy: {video_metrics['balanced_accuracy']:.4f}")
        print(f"    Macro F1:          {video_metrics['macro_f1']:.4f}")

        cm_frame = np.array(frame_metrics["confusion_matrix"])
        cm_vid = np.array(video_metrics["confusion_matrix"])

        cm_dir = os.path.join(output_dir, "confusion_matrices")
        cm_frame_path = os.path.join(cm_dir, f"{experiment_name}_{model_name.lower()}_frame_cm.png")
        cm_vid_path = os.path.join(cm_dir, f"{experiment_name}_{model_name.lower()}_video_cm.png")

        plot_and_save_confusion_matrix(cm_frame, f"{model_name} [Frame-Level] - {experiment_name}", cm_frame_path)
        plot_and_save_confusion_matrix(cm_vid, f"{model_name} [Video-Level] - {experiment_name}", cm_vid_path)

        results["models"][model_name] = {
            "fit_time_sec": fit_time,
            "cv_group_metrics": {
                "mean_balanced_accuracy": round(mean_cv_bal_acc, 4),
                "std_balanced_accuracy": round(std_cv_bal_acc, 4),
                "mean_macro_f1": round(mean_cv_macro_f1, 4),
            },
            "frame_level_test_metrics": frame_metrics,
            "video_level_test_metrics": video_metrics,
            "cm_frame_plot": cm_frame_path,
            "cm_video_plot": cm_vid_path,
        }

    return results, models

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Baseline Model Training and Participant-Independent Evaluation")
    parser.add_argument("--features", default=r"F:\DrowsinessDetector\data\processed\features.csv", help="Path to features.csv")
    parser.add_argument("--output-dir", default=r"F:\DrowsinessDetector\data\modeling", help="Directory for modeling artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 74)
    print("  PHASE 2: PARTICIPANT-INDEPENDENT BASELINE MODELING")
    print("=" * 74)

    print(f"[INFO] Loading features dataset from: '{args.features}'...")
    df = load_dataset(args.features)
    print(f"[INFO] Loaded {len(df)} total rows across {df['participant_id'].nunique()} participants.")

    train_parts, test_parts = create_participant_split(df, test_size=0.20, random_state=RANDOM_SEED)

    split_info = {
        "random_seed": RANDOM_SEED,
        "test_size_fraction": 0.20,
        "train_participant_count": len(train_parts),
        "test_participant_count": len(test_parts),
        "train_participants": train_parts,
        "test_participants": test_parts,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    split_path = os.path.join(args.output_dir, "split_assignments.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Saved participant split assignments to: '{split_path}'")
    print(f"       Held-out Test Participants ({len(test_parts)}): {test_parts}")
    print(f"       Train/Dev Participants   ({len(train_parts)}): {train_parts}")

    res_a, models_a = run_experiment(
        df=df,
        train_parts=train_parts,
        test_parts=test_parts,
        experiment_name="expA_valid_faces",
        include_missing_face_flag=False,
        output_dir=args.output_dir,
    )

    exp_a_path = os.path.join(args.output_dir, "metrics_experiment_A.json")
    with open(exp_a_path, "w", encoding="utf-8") as f:
        json.dump(res_a, f, indent=2)
    print(f"[INFO] Saved Experiment A metrics to: '{exp_a_path}'")

    res_b, models_b = run_experiment(
        df=df,
        train_parts=train_parts,
        test_parts=test_parts,
        experiment_name="expB_with_missing_indicator",
        include_missing_face_flag=True,
        output_dir=args.output_dir,
    )

    exp_b_path = os.path.join(args.output_dir, "metrics_experiment_B.json")
    with open(exp_b_path, "w", encoding="utf-8") as f:
        json.dump(res_b, f, indent=2)
    print(f"[INFO] Saved Experiment B metrics to: '{exp_b_path}'")

    for name, model_obj in models_a.items():
        art_path = os.path.join(models_dir, f"expA_{name.lower()}.joblib")
        joblib.dump(model_obj, art_path)

    for name, model_obj in models_b.items():
        art_path = os.path.join(models_dir, f"expB_{name.lower()}.joblib")
        joblib.dump(model_obj, art_path)

    print(f"[INFO] Serialized all trained model pipelines to: '{models_dir}'")
    print("=" * 74)
    print("  PHASE 2 MODEL TRAINING AND EVALUATION COMPLETE")
    print("=" * 74)

if __name__ == "__main__":
    main()
