import os
import sys
import time
import shutil
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

LEFT_EYE_SLICE = slice(42, 48)
RIGHT_EYE_SLICE = slice(36, 42)
INNER_MOUTH_SLICE = slice(60, 68)

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    a = dist.euclidean(eye_pts[1], eye_pts[5])
    b = dist.euclidean(eye_pts[2], eye_pts[4])
    c = dist.euclidean(eye_pts[0], eye_pts[3])

    if c <= 1e-6:
        return 0.0
    return float((a + b) / (2.0 * c))

def mouth_aspect_ratio(mouth_pts: np.ndarray) -> float:
    a = dist.euclidean(mouth_pts[1], mouth_pts[7])
    b = dist.euclidean(mouth_pts[2], mouth_pts[6])
    c = dist.euclidean(mouth_pts[3], mouth_pts[5])
    d = dist.euclidean(mouth_pts[0], mouth_pts[4])

    if d <= 1e-6:
        return 0.0
    return float((a + b + c) / (2.0 * d))

LABEL_MAP = {
    "0": {"label": 0, "name": "alert", "kss": "1-5", "desc": "Alert / Normal driving"},
    "5": {"label": 5, "name": "low_vigilant", "kss": "6-7", "desc": "Low vigilance / Subtle fatigue"},
    "10": {"label": 10, "name": "drowsy", "kss": "8-9", "desc": "Drowsy / Fighting sleep"},
    "10_1": {"label": 10, "name": "drowsy", "kss": "8-9", "desc": "Drowsy part 1 (Participant 32)"},
    "10_2": {"label": 10, "name": "drowsy", "kss": "8-9", "desc": "Drowsy part 2 (Participant 32)"},
}

LABEL_SOURCE_DOC = "UTA-RLDD (Ghoddoosian et al., 2019 / arXiv:1905.04414); Video-level KSS assignment"

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}

def resolve_dataset_root(explicit_path: Optional[str] = None) -> str:
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)

    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(base_dir, "data", "row"),
        os.path.join(base_dir, "data", "raw"),
        r"F:\DrowsinessDetector\data\row",
        r"F:\DrowsinessDetector\data\raw",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "Could not locate dataset directory. Looked for 'data/row' and 'data/raw'. "
        "Please specify with --data-dir <path>."
    )

def discover_videos(dataset_root: str) -> List[Dict[str, str]]:
    discovered = []

    for root, _, files in os.walk(dataset_root):
        for fname in files:
            ext = os.path.splitext(fname)[1]
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, dataset_root)
            parts = rel_path.replace("\\", "/").split("/")

            participant_id = "unknown"
            for part in reversed(parts[:-1]):
                if part.isdigit():
                    participant_id = part
                    break

            fold = "unknown"
            for part in parts[:-1]:
                if "fold" in part.lower():
                    fold = part
                    break

            base_name = os.path.splitext(fname)[0]
            label_info = LABEL_MAP.get(base_name)

            if label_info:
                video_label = label_info["label"]
                state_label = label_info["name"]
                kss_range = label_info["kss"]
            else:
                video_label = -1
                state_label = "uncertain"
                kss_range = "unknown"

            discovered.append({
                "full_path": full_path,
                "rel_path": rel_path,
                "fold": fold,
                "participant_id": participant_id,
                "video_filename": fname,
                "base_name": base_name,
                "video_label": video_label,
                "state_label": state_label,
                "kss_range": kss_range,
            })

    discovered.sort(key=lambda x: (x["participant_id"], x["base_name"], x["video_filename"]))
    return discovered

FEATURE_CSV_COLUMNS = [
    "participant_id",
    "fold",
    "video_filename",
    "video_label",
    "state_label",
    "kss_range",
    "frame_index",
    "timestamp_sec",
    "face_detected",
    "face_left",
    "face_top",
    "face_right",
    "face_bottom",
    "ear_left",
    "ear_right",
    "ear",
    "mar",
    "mouth_over_ear",
    "interocular_dist",
    "label_source",
]

MANIFEST_COLUMNS = [
    "participant_id",
    "fold",
    "video_filename",
    "rel_path",
    "video_label",
    "state_label",
    "total_video_frames",
    "sampled_frames",
    "faces_detected",
    "rows_written",
    "status",
    "processing_time_sec",
    "error_message",
    "completed_at",
]

def load_completed_manifest(manifest_path: str) -> Set[Tuple[str, str]]:
    completed = set()
    if not os.path.isfile(manifest_path):
        return completed

    try:
        import csv
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "COMPLETED":
                    completed.add((row.get("participant_id"), row.get("video_filename")))
    except Exception as e:
        print(f"[WARN] Failed to read manifest '{manifest_path}': {e}", file=sys.stderr)

    return completed

def append_manifest_record(manifest_path: str, record: Dict[str, any]):
    import csv
    file_exists = os.path.isfile(manifest_path)
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if not file_exists or os.path.getsize(manifest_path) == 0:
            writer.writeheader()
        writer.writerow(record)
        f.flush()
        os.fsync(f.fileno())

def append_feature_rows(features_path: str, rows: List[Dict[str, any]]):
    if not rows:
        return
    import csv
    file_exists = os.path.isfile(features_path)
    os.makedirs(os.path.dirname(os.path.abspath(features_path)), exist_ok=True)

    with open(features_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_CSV_COLUMNS)
        if not file_exists or os.path.getsize(features_path) == 0:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

def process_single_video(
    video_meta: Dict[str, any],
    detector: any,
    predictor: any,
    fps_sample: float = 1.0,
    resize_width: int = 640,
    verbose: bool = True,
) -> Tuple[str, List[Dict[str, any]], Dict[str, any]]:
    video_path = video_meta["full_path"]
    t_start = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "FAILED", [], {
            "error": f"Failed to open video stream: {video_path}",
            "sampled_frames": 0,
            "faces_detected": 0,
            "total_frames": 0,
            "duration": round(time.time() - t_start, 2),
        }

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_stride = max(1, int(round(native_fps / fps_sample)))

    rows = []
    frame_idx = 0
    sampled_count = 0
    face_count = 0

    try:
        while True:
            if frame_idx % frame_stride == 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                sampled_count += 1
                timestamp_sec = round(frame_idx / native_fps, 3)

                orig_h, orig_w = frame.shape[:2]
                if resize_width > 0 and max(orig_h, orig_w) > resize_width:
                    scale = float(resize_width) / float(max(orig_h, orig_w))
                    new_w = int(round(orig_w * scale))
                    new_h = int(round(orig_h * scale))
                    proc_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    scale = 1.0
                    proc_frame = frame

                rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                rgb_contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)

                rects = detector(rgb_contiguous, 0)

                if len(rects) == 0:
                    row_data = {
                        "participant_id": video_meta["participant_id"],
                        "fold": video_meta["fold"],
                        "video_filename": video_meta["video_filename"],
                        "video_label": video_meta["video_label"],
                        "state_label": video_meta["state_label"],
                        "kss_range": video_meta["kss_range"],
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "face_detected": 0,
                        "face_left": "",
                        "face_top": "",
                        "face_right": "",
                        "face_bottom": "",
                        "ear_left": "",
                        "ear_right": "",
                        "ear": "",
                        "mar": "",
                        "mouth_over_ear": "",
                        "interocular_dist": "",
                        "label_source": LABEL_SOURCE_DOC,
                    }
                else:
                    face_count += 1

                    primary_rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

                    box_left = int(round(primary_rect.left() / scale))
                    box_top = int(round(primary_rect.top() / scale))
                    box_right = int(round(primary_rect.right() / scale))
                    box_bottom = int(round(primary_rect.bottom() / scale))

                    shape = predictor(rgb_contiguous, primary_rect)
                    coords = np.zeros((68, 2), dtype=np.float64)
                    for i in range(68):
                        pt = shape.part(i)
                        coords[i] = (pt.x, pt.y)

                    left_eye = coords[LEFT_EYE_SLICE]
                    right_eye = coords[RIGHT_EYE_SLICE]
                    inner_mouth = coords[INNER_MOUTH_SLICE]

                    ear_l = eye_aspect_ratio(left_eye)
                    ear_r = eye_aspect_ratio(right_eye)
                    ear_val = (ear_l + ear_r) / 2.0
                    mar_val = mouth_aspect_ratio(inner_mouth)
                    moe_val = (mar_val / ear_val) if ear_val > 1e-6 else 0.0

                    iod_native = dist.euclidean(coords[36], coords[45]) / scale

                    row_data = {
                        "participant_id": video_meta["participant_id"],
                        "fold": video_meta["fold"],
                        "video_filename": video_meta["video_filename"],
                        "video_label": video_meta["video_label"],
                        "state_label": video_meta["state_label"],
                        "kss_range": video_meta["kss_range"],
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "face_detected": 1,
                        "face_left": box_left,
                        "face_top": box_top,
                        "face_right": box_right,
                        "face_bottom": box_bottom,
                        "ear_left": round(ear_l, 4),
                        "ear_right": round(ear_r, 4),
                        "ear": round(ear_val, 4),
                        "mar": round(mar_val, 4),
                        "mouth_over_ear": round(moe_val, 4),
                        "interocular_dist": round(iod_native, 2),
                        "label_source": LABEL_SOURCE_DOC,
                    }

                rows.append(row_data)

            else:
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1

    except Exception as e:
        cap.release()
        return "FAILED", [], {
            "error": str(e),
            "sampled_frames": sampled_count,
            "faces_detected": face_count,
            "total_frames": total_frames,
            "duration": round(time.time() - t_start, 2),
        }

    cap.release()
    t_end = time.time()

    stats = {
        "error": "",
        "sampled_frames": sampled_count,
        "faces_detected": face_count,
        "total_frames": total_frames,
        "duration": round(t_end - t_start, 2),
    }
    return "COMPLETED", rows, stats

def check_disk_space(path: str, min_mb: int = 500) -> float:
    check_dir = path
    while not os.path.exists(check_dir):
        parent = os.path.dirname(check_dir)
        if parent == check_dir:
            break
        check_dir = parent

    total, used, free = shutil.disk_usage(check_dir)
    free_gb = free / (1024 ** 3)
    free_mb = free / (1024 ** 2)

    if free_mb < min_mb:
        raise IOError(
            f"Insufficient free disk space on '{check_dir}': {free_mb:.1f} MB available. "
            f"Safety limit requires at least {min_mb} MB."
        )
    return free_gb

def run_feature_extraction(
    data_dir: Optional[str] = None,
    output_path: str = r"F:\DrowsinessDetector\data\processed\features.csv",
    manifest_path: str = r"F:\DrowsinessDetector\data\processed\manifest.csv",
    model_path: str = "shape_predictor_68_face_landmarks.dat",
    fps_sample: float = 1.0,
    resize_width: int = 640,
    min_disk_mb: int = 500,
    limit_videos: Optional[int] = None,
    filter_participant: Optional[str] = None,
    select_videos: Optional[List[str]] = None,
    force: bool = False,
    quiet: bool = False,
):
    print("=" * 76)
    print("  DRIVER DROWSINESS DETECTION: FEATURE EXTRACTION PIPELINE (PHASE 1)")
    print("=" * 76)

    dataset_root = resolve_dataset_root(data_dir)
    print(f"[INFO] Dataset root directory: '{dataset_root}'")

    free_gb_before = check_disk_space(output_path, min_mb=min_disk_mb)
    print(f"[INFO] Storage check passed: {free_gb_before:.2f} GB available on drive.")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run setup_model.py.")
    print(f"[INFO] Loading Dlib facial landmark predictor: '{model_path}'...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    all_videos = discover_videos(dataset_root)
    print(f"[INFO] Discovered {len(all_videos)} supported video files in dataset.")

    target_videos = all_videos
    if filter_participant:
        target_videos = [v for v in target_videos if v["participant_id"] == filter_participant]
        print(f"[INFO] Filtered by participant '{filter_participant}': {len(target_videos)} video(s).")

    if select_videos:
        sel_set = {s.strip().lower() for s in select_videos}
        target_videos = [
            v for v in target_videos
            if v["video_filename"].lower() in sel_set or v["base_name"].lower() in sel_set
        ]
        print(f"[INFO] Filtered by selection: {len(target_videos)} video(s).")

    if limit_videos and limit_videos > 0:
        target_videos = target_videos[:limit_videos]
        print(f"[INFO] Video count limited to: {len(target_videos)} video(s).")

    if not target_videos:
        print("[WARN] No videos matched the specified criteria. Exiting.")
        return

    completed_set = set() if force else load_completed_manifest(manifest_path)
    if completed_set:
        print(f"[INFO] Found {len(completed_set)} previously completed video(s) in manifest.")

    total_to_process = len(target_videos)
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_rows_written = 0
    total_faces_detected = 0

    pipeline_start_time = time.time()

    print(f"[INFO] Target output:   '{output_path}'")
    print(f"[INFO] Target manifest: '{manifest_path}'")
    print("-" * 76)

    for idx, v_meta in enumerate(target_videos, 1):
        v_key = (v_meta["participant_id"], v_meta["video_filename"])

        if not force and v_key in completed_set:
            skipped_count += 1
            if not quiet:
                print(f"[{idx:02d}/{total_to_process:02d}] SKIPPED: Participant {v_meta['participant_id']} | {v_meta['video_filename']} (Already completed)")
            continue

        check_disk_space(output_path, min_mb=min_disk_mb)

        print(f"[{idx:02d}/{total_to_process:02d}] PROCESSING: Participant {v_meta['participant_id']} | "
              f"{v_meta['video_filename']} (State: {v_meta['state_label']}, Code: {v_meta['video_label']})...")

        status, rows, stats = process_single_video(
            video_meta=v_meta,
            detector=detector,
            predictor=predictor,
            fps_sample=fps_sample,
            resize_width=resize_width,
            verbose=not quiet,
        )

        completed_at_str = datetime.now().isoformat()

        if status == "COMPLETED":
            append_feature_rows(output_path, rows)

            manifest_rec = {
                "participant_id": v_meta["participant_id"],
                "fold": v_meta["fold"],
                "video_filename": v_meta["video_filename"],
                "rel_path": v_meta["rel_path"],
                "video_label": v_meta["video_label"],
                "state_label": v_meta["state_label"],
                "total_video_frames": stats["total_frames"],
                "sampled_frames": stats["sampled_frames"],
                "faces_detected": stats["faces_detected"],
                "rows_written": len(rows),
                "status": "COMPLETED",
                "processing_time_sec": stats["duration"],
                "error_message": "",
                "completed_at": completed_at_str,
            }
            append_manifest_record(manifest_path, manifest_rec)

            processed_count += 1
            total_rows_written += len(rows)
            total_faces_detected += stats["faces_detected"]
            completed_set.add(v_key)

            pct_faces = (stats["faces_detected"] / stats["sampled_frames"] * 100.0) if stats["sampled_frames"] > 0 else 0.0
            print(f"    -> SUCCESS: {len(rows)} rows written | Face rate: {pct_faces:.1f}% ({stats['faces_detected']}/{stats['sampled_frames']}) | Time: {stats['duration']}s")

        else:
            failed_count += 1
            manifest_rec = {
                "participant_id": v_meta["participant_id"],
                "fold": v_meta["fold"],
                "video_filename": v_meta["video_filename"],
                "rel_path": v_meta["rel_path"],
                "video_label": v_meta["video_label"],
                "state_label": v_meta["state_label"],
                "total_video_frames": stats["total_frames"],
                "sampled_frames": stats["sampled_frames"],
                "faces_detected": stats["faces_detected"],
                "rows_written": 0,
                "status": "FAILED",
                "processing_time_sec": stats["duration"],
                "error_message": stats["error"],
                "completed_at": completed_at_str,
            }
            append_manifest_record(manifest_path, manifest_rec)
            print(f"    -> [ERROR] Failed processing {v_meta['video_filename']}: {stats['error']}", file=sys.stderr)

    total_elapsed = round(time.time() - pipeline_start_time, 2)
    free_gb_after = check_disk_space(output_path, min_mb=0)

    features_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    manifest_bytes = os.path.getsize(manifest_path) if os.path.isfile(manifest_path) else 0

    print("=" * 76)
    print("  EXTRACTION SUMMARY REPORT")
    print("=" * 76)
    print(f"  Videos Discovered:       {len(all_videos)}")
    print(f"  Videos Targeted:         {total_to_process}")
    print(f"  Videos Processed:        {processed_count}")
    print(f"  Videos Skipped:          {skipped_count}")
    print(f"  Videos Failed:           {failed_count}")
    print(f"  Rows Written:            {total_rows_written}")
    print(f"  Total Faces Detected:    {total_faces_detected}")
    print(f"  Output CSV Size:         {features_bytes / (1024**2):.3f} MB ({features_bytes} bytes)")
    print(f"  Manifest Size:           {manifest_bytes / 1024:.2f} KB ({manifest_bytes} bytes)")
    print(f"  Free Space Before:       {free_gb_before:.3f} GB")
    print(f"  Free Space After:        {free_gb_after:.3f} GB (Delta: {free_gb_before - free_gb_after:.4f} GB)")
    print(f"  Total Time Elapsed:      {total_elapsed}s")
    print("=" * 76)

def main():
    parser = argparse.ArgumentParser(
        description="Feature Extraction Pipeline for Driver Drowsiness Detection (Phase 1)"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to raw dataset root (default: auto-detects data/row or data/raw)"
    )
    parser.add_argument(
        "--output", default=r"F:\DrowsinessDetector\data\processed\features.csv",
        help="Output CSV file path (default: data/processed/features.csv)"
    )
    parser.add_argument(
        "--manifest", default=r"F:\DrowsinessDetector\data\processed\manifest.csv",
        help="Manifest CSV file path (default: data/processed/manifest.csv)"
    )
    parser.add_argument(
        "--model", default="shape_predictor_68_face_landmarks.dat",
        help="Path to 68-point landmark predictor weights"
    )
    parser.add_argument(
        "--fps-sample", type=float, default=1.0,
        help="Frames per second to sample (default: 1.0 FPS)"
    )
    parser.add_argument(
        "--resize-width", type=int, default=640,
        help="Resize width for Dlib HOG inference (default: 640, 0 to disable)"
    )
    parser.add_argument(
        "--min-disk-mb", type=int, default=1000,
        help="Minimum free disk space threshold in MB (default: 1000 MB / 1 GB)"
    )
    parser.add_argument(
        "--limit-videos", type=int, default=None,
        help="Maximum number of videos to process in this run"
    )
    parser.add_argument(
        "--participant", type=str, default=None,
        help="Filter and process only a specific participant ID (e.g., '01')"
    )
    parser.add_argument(
        "--select-videos", type=str, default=None,
        help="Comma-separated video filenames or base names to process (e.g., '0.mov,5.mov,10.MOV')"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-processing of videos even if already completed in manifest"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose per-frame outputs"
    )

    args = parser.parse_args()

    select_list = [s.strip() for s in args.select_videos.split(",")] if args.select_videos else None

    run_feature_extraction(
        data_dir=args.data_dir,
        output_path=args.output,
        manifest_path=args.manifest,
        model_path=args.model,
        fps_sample=args.fps_sample,
        resize_width=args.resize_width,
        min_disk_mb=args.min_disk_mb,
        limit_videos=args.limit_videos,
        filter_participant=args.participant,
        select_videos=select_list,
        force=args.force,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()
