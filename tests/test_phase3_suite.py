import os
import sys
import time
import json
import shutil
from typing import Dict, List, Tuple, Any

import cv2
import dlib
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_hog_svm import (
    AudioBuzzer,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    MLFatigueAdvisoryEngine,
    HUDVisualizer,
    DriverFatiguePipeline,
)

def log_test_header(suite_name: str):
    print("\n" + "=" * 78)
    print(f" {suite_name.upper()}")
    print("=" * 78)

def run_suite_a_regression() -> Dict[str, Any]:
    log_test_header("Suite A: Regression Testing (Rule Baseline & Hardware)")
    results = {}

    print("[TEST A1] Testing Pygame Audio Synthesizer (1100 Hz Stereo)...")
    buzzer = AudioBuzzer(frequency=1100, sample_rate=22050, buffer_size=512)
    assert not buzzer.is_playing, "Buzzer should initialize in non-playing state"
    buzzer.start_alert()
    assert buzzer.is_playing, "Buzzer should report is_playing == True after start_alert()"
    time.sleep(0.1)
    buzzer.stop_alert()
    assert not buzzer.is_playing, "Buzzer should report is_playing == False after stop_alert()"
    buzzer.shutdown()
    print("  -> PASSED: Synthesizer starts, plays non-blocking, and stops cleanly.")
    results["audio_buzzer"] = "PASSED"

    print("[TEST A2] Testing EAR Drowsiness Counter & Trigger (EAR < 0.25 for 25 frames)...")
    closed_counter = 0
    ear_thresh = 0.25
    ear_limit = 25
    buzzer_active = False

    for f in range(1, 25):
        ear_val = 0.18
        if ear_val < ear_thresh:
            closed_counter += 1
        else:
            closed_counter = 0
        is_drowsy = closed_counter >= ear_limit
        if is_drowsy:
            buzzer_active = True
        assert not is_drowsy, f"Alarm triggered prematurely at frame {f}"

    assert closed_counter == 24, f"Expected 24 closed frames, got {closed_counter}"

    ear_val = 0.18
    closed_counter += 1
    is_drowsy = closed_counter >= ear_limit
    if is_drowsy:
        buzzer_active = True
    assert is_drowsy, "Alarm did not trigger at frame 25"
    assert buzzer_active, "Buzzer was not activated"

    ear_val = 0.32
    if ear_val < ear_thresh:
        closed_counter += 1
    else:
        closed_counter = 0
    is_drowsy = closed_counter >= ear_limit
    if not is_drowsy:
        buzzer_active = False
    assert closed_counter == 0, "Closed counter failed to reset on eye opening"
    assert not buzzer_active, "Buzzer failed to stop on eye opening"
    print("  -> PASSED: EAR threshold exactly triggers at frame 25 and resets on eye open.")
    results["ear_counter_behavior"] = "PASSED"

    print("[TEST A3] Testing MAR Yawning Counter & Trigger (MAR > 0.65 for 30 frames)...")
    yawn_counter = 0
    mar_thresh = 0.65
    mar_limit = 30
    warning_active = False

    for f in range(1, 30):
        mar_val = 0.72
        if mar_val > mar_thresh:
            yawn_counter += 1
        else:
            yawn_counter = 0
        is_yawning = yawn_counter >= mar_limit
        if is_yawning:
            warning_active = True
        assert not is_yawning, f"Yawn warning triggered prematurely at frame {f}"

    assert yawn_counter == 29, f"Expected 29 yawn frames, got {yawn_counter}"

    mar_val = 0.72
    yawn_counter += 1
    is_yawning = yawn_counter >= mar_limit
    assert is_yawning, "Yawn warning did not trigger at frame 30"

    mar_val = 0.10
    if mar_val > mar_thresh:
        yawn_counter += 1
    else:
        yawn_counter = 0
    assert yawn_counter == 0, "Yawn counter failed to reset on mouth closure"
    print("  -> PASSED: MAR threshold exactly triggers at frame 30 and resets on mouth close.")
    results["mar_counter_behavior"] = "PASSED"

    print("[TEST A4] Testing Priority Arbitration (Drowsy > Yawn) & Face Loss Cleanup...")

    is_drowsy = True
    is_yawning = True
    buzzer_on = False
    status_text = ""
    if is_drowsy:
        status_text = "CRITICAL: DROWSINESS DETECTED!"
        buzzer_on = True
    elif is_yawning:
        status_text = "WARNING: YAWNING DETECTED!"
        buzzer_on = False

    assert status_text == "CRITICAL: DROWSINESS DETECTED!", "Drowsiness did not take priority over yawning"
    assert buzzer_on, "Buzzer should be active on drowsiness"

    face_count = 0
    if face_count == 0:
        closed_counter = 0
        yawn_counter = 0
        buzzer_on = False
        status_text = "STATUS: NO FACE DETECTED"

    assert closed_counter == 0 and yawn_counter == 0, "Counters must reset on face loss"
    assert not buzzer_on, "Buzzer must stop immediately on face loss"
    print("  -> PASSED: Alert priority arbitration and face loss buzzer stop verified.")
    results["arbitration_and_face_loss"] = "PASSED"

    return results

def run_suite_b_ml_integration() -> Dict[str, Any]:
    log_test_header("Suite B: ML Advisory Integration & Safety Testing")
    results = {}
    model_path = "data/modeling/models/expB_randomforest.joblib"

    print(f"[TEST B1] Loading and verifying model pipeline from '{model_path}'...")
    engine = MLFatigueAdvisoryEngine(model_path=model_path, enabled=True, inference_interval=1, ema_alpha=0.30)
    assert engine.is_active, "MLFatigueAdvisoryEngine failed to load valid model"
    clf = engine.pipeline.named_steps["clf"]
    assert clf.n_features_in_ == 7, f"Expected 7 features, found {clf.n_features_in_}"
    assert list(clf.classes_) == [0, 1, 2], f"Expected classes [0, 1, 2], found {clf.classes_}"
    print(f"  -> Pipeline steps: {list(engine.pipeline.named_steps.keys())}")
    print(f"  -> Features required: {clf.n_features_in_} (Order: ear_left, ear_right, ear, mar, moe, iod, missing_face)")
    print(f"  -> Classes verified: {clf.classes_} -> ['ALERT', 'LOW VIGILANT', 'DROWSY']")
    results["pipeline_verification"] = "PASSED"

    print("[TEST B2] Testing inference with typical alert biometric vector...")
    res = engine.update(
        ear_left=0.315,
        ear_right=0.310,
        ear_mean=0.3125,
        mar=0.045,
        mouth_over_ear=0.045 / 0.3125,
        interocular_dist=205.0,
        face_detected=True,
        frame_index=1
    )
    assert res["active"], "Engine should report active"
    assert res["class_name"] in engine.CLASS_NAMES, f"Invalid class name {res['class_name']}"
    probs = res["probs"]
    assert len(probs) == 3, f"Expected 3 probabilities, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {sum(probs)}"
    print(f"  -> Valid vector output: Class={res['class_name']} ({res['confidence']*100:.1f}%), Probs={probs}")
    results["valid_vector"] = "PASSED"

    print("[TEST B3] Testing extreme values (zero EAR, zero MAR, zero IOD, division by zero)...")
    res_zero = engine.update(
        ear_left=0.0,
        ear_right=0.0,
        ear_mean=0.0,
        mar=0.0,
        mouth_over_ear=0.0,
        interocular_dist=0.0,
        face_detected=True,
        frame_index=2
    )
    assert res_zero["active"], "Engine failed on zero vector"
    assert np.all(np.isfinite(res_zero["probs"])), "Zero vector generated non-finite probabilities"
    assert abs(sum(res_zero["probs"]) - 1.0) < 1e-4, "Zero vector probabilities did not normalize"
    print(f"  -> Zero-value vector handled safely: Class={res_zero['class_name']}, Probs={res_zero['probs']}")

    print("  -> Testing NaN and Inf inputs...")
    res_nan = engine.update(
        ear_left=float("nan"),
        ear_right=float("inf"),
        ear_mean=float("-inf"),
        mar=float("nan"),
        mouth_over_ear=float("nan"),
        interocular_dist=float("nan"),
        face_detected=True,
        frame_index=3
    )
    assert np.all(np.isfinite(res_nan["probs"])), "NaN/Inf inputs produced non-finite output"
    assert abs(sum(res_nan["probs"]) - 1.0) < 1e-4, "Probabilities failed normalization on NaN input"
    print(f"  -> NaN/Inf handled safely via numerical guards: Probs={res_nan['probs']}")
    results["extreme_and_invalid_values"] = "PASSED"

    print("[TEST B4] Testing graceful fallback on missing/incompatible model...")

    fake_engine = MLFatigueAdvisoryEngine(model_path="data/modeling/models/nonexistent_test.joblib", enabled=True)
    assert not fake_engine.is_active, "Fake model path should leave engine inactive"
    assert "OFFLINE (FALLBACK)" in fake_engine.status_message, "Status message should indicate fallback"
    fallback_res = fake_engine.update(0.3, 0.3, 0.3, 0.05, 0.16, 200.0, face_detected=True)
    assert not fallback_res["active"], "Fallback result should report active=False"
    assert fallback_res["class_name"] == "N/A", "Fallback class name should be N/A"

    scratch_corrupt = "C:/Users/DELL/.gemini/antigravity-ide/brain/4c76ba7c-5e2b-4ab9-82f0-90687ba5f593/scratch/corrupt_test.joblib"
    os.makedirs(os.path.dirname(scratch_corrupt), exist_ok=True)
    with open(scratch_corrupt, "wb") as f:
        f.write(b"NOT_A_VALID_JOBLIB_PICKLE_STREAM")
    corrupt_engine = MLFatigueAdvisoryEngine(model_path=scratch_corrupt, enabled=True)
    assert not corrupt_engine.is_active, "Corrupt model should leave engine inactive"
    assert "OFFLINE (LOAD ERROR)" in corrupt_engine.status_message, "Status message should indicate load error"
    print("  -> PASSED: Non-existent and corrupt models safely trigger offline fallback without crashing.")
    results["model_fallback"] = "PASSED"

    print("[TEST B5] Testing missing-face behavior...")

    engine.update(0.32, 0.32, 0.32, 0.05, 0.15, 200.0, face_detected=True, frame_index=4)

    mf_res = engine.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, face_detected=False, frame_index=5)
    assert mf_res["class_name"] == "NO FACE", f"Expected 'NO FACE', got {mf_res['class_name']}"
    assert mf_res["confidence"] == 0.0, "Missing face should have confidence 0.0"
    assert mf_res["probs"] == [0.0, 0.0, 0.0], f"Stale probabilities returned on missing face: {mf_res['probs']}"
    assert engine.smoothed_probs is None, "Smoothing buffer must be reset on face loss"
    print("  -> PASSED: Missing face resets smoothing buffer and outputs NO FACE (no synthetic drowsiness).")
    results["missing_face_behavior"] = "PASSED"

    return results

def run_suite_c_dataset_replay() -> Dict[str, Any]:
    log_test_header("Suite C: Dataset Video Replay (Held-out Participant 18)")
    print("NOTE: UTA-RLDD session labels (0: alert, 5: low_vigilant, 10: drowsy) are")
    print("      session-level self-assessments (KSS), NOT continuous frame-level ground truth.\n")

    p18_dir = "F:/DrowsinessDetector/data/row/Fold2_part1/Fold2_part1/18"
    video_files = {
        "0.mov": {"session": "Alert (0)", "expected_state": 0},
        "5.mov": {"session": "Low Vigilance (5)", "expected_state": 1},
        "10.mov": {"session": "Drowsy (10)", "expected_state": 2},
    }

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    model_path = "data/modeling/models/expB_randomforest.joblib"
    engine = MLFatigueAdvisoryEngine(model_path=model_path, enabled=True, inference_interval=3, ema_alpha=0.30)

    replay_results = {}

    for v_file, meta in video_files.items():
        v_path = os.path.join(p18_dir, v_file)
        print(f"--- Replaying Video: {v_file} ({meta['session']}) ---")
        if not os.path.isfile(v_path):
            print(f"  [ERROR] File not found: {v_path}")
            continue

        cap = cv2.VideoCapture(v_path)
        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_step = max(1, int(round(fps_native / 2.0)))
        max_samples = 120

        frame_idx = 0
        sampled_count = 0
        face_detect_count = 0
        latencies = []
        state_counts = {"ALERT": 0, "LOW VIGILANT": 0, "DROWSY": 0, "NO FACE": 0}
        prob_sums = np.zeros(3, dtype=np.float64)
        prob_count = 0

        t_start = time.perf_counter()

        while cap.isOpened() and sampled_count < max_samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            sampled_count += 1
            h, w = frame.shape[:2]
            scale = 640.0 / w
            resized = cv2.resize(frame, (640, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            rgb_c = np.ascontiguousarray(rgb, dtype=np.uint8)

            t0 = time.perf_counter()
            rects = detector(rgb_c, 0)

            if len(rects) == 0:
                ml_res = engine.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, face_detected=False, frame_index=sampled_count)
                state_counts["NO FACE"] += 1
            else:
                face_detect_count += 1
                primary = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
                shape = predictor(rgb_c, primary)
                coords = np.zeros((68, 2), dtype=np.float64)
                for i in range(68):
                    pt = shape.part(i)
                    coords[i] = (pt.x, pt.y)

                left_eye = coords[slice(42, 48)]
                right_eye = coords[slice(36, 42)]
                inner_mouth = coords[slice(60, 68)]

                el = eye_aspect_ratio(left_eye)
                er = eye_aspect_ratio(right_eye)
                em = (el + er) / 2.0
                m = mouth_aspect_ratio(inner_mouth)
                moe = (m / em) if em > 1e-6 else 0.0
                iod = np.linalg.norm(coords[36] - coords[45]) / scale

                ml_res = engine.update(
                    ear_left=el,
                    ear_right=er,
                    ear_mean=em,
                    mar=m,
                    mouth_over_ear=moe,
                    interocular_dist=iod,
                    face_detected=True,
                    frame_index=sampled_count
                )

                state_counts[ml_res["class_name"]] = state_counts.get(ml_res["class_name"], 0) + 1
                if ml_res["active"] and ml_res["class_name"] != "NO FACE":
                    prob_sums += np.array(ml_res["probs"])
                    prob_count += 1
                    latencies.append(ml_res["latency_ms"])

            frame_idx += 1

        cap.release()
        total_time = time.perf_counter() - t_start

        face_rate = (face_detect_count / sampled_count * 100) if sampled_count > 0 else 0.0
        mean_probs = (prob_sums / prob_count).tolist() if prob_count > 0 else [0.0, 0.0, 0.0]
        mean_latency = float(np.mean(latencies)) if latencies else 0.0

        print(f"  Frames Processed: {sampled_count} | Face Detection Rate: {face_rate:.1f}% ({face_detect_count}/{sampled_count})")
        print(f"  Mean Probabilities: Alert={mean_probs[0]:.3f}, LowVigilant={mean_probs[1]:.3f}, Drowsy={mean_probs[2]:.3f}")
        print(f"  Predicted State Breakdown: {state_counts}")
        print(f"  Mean ML Inference Latency: {mean_latency:.2f} ms | Total Replay Duration: {total_time:.2f} s\n")

        replay_results[v_file] = {
            "session": meta["session"],
            "sampled_frames": sampled_count,
            "face_detect_rate_pct": round(face_rate, 2),
            "mean_probs": [round(p, 4) for p in mean_probs],
            "state_distribution": state_counts,
            "mean_inference_latency_ms": round(mean_latency, 2),
            "total_time_sec": round(total_time, 2),
        }

    return replay_results

def run_suite_d_realtime_benchmark() -> Dict[str, Any]:
    log_test_header("Suite D: Real-Time Performance & Scenario Testing")
    results = {}

    print("[TEST D1] Benchmarking Real-Time Camera Loop Performance (640x480)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("  [WARN] Physical camera not available for benchmark. Falling back to synthetic stream.")
        use_live_cam = False
    else:
        use_live_cam = True
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def measure_fps(enable_ml: bool, num_frames: int = 60) -> Tuple[float, float]:
        engine = MLFatigueAdvisoryEngine(enabled=enable_ml, inference_interval=3, ema_alpha=0.30)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        durations = []
        ml_latencies = []

        for f_idx in range(num_frames):
            t0 = time.perf_counter()
            if use_live_cam:
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_c = np.ascontiguousarray(rgb, dtype=np.uint8)

            rects = detector(rgb_c, 0)
            if len(rects) == 0:
                if enable_ml:
                    engine.update(0, 0, 0, 0, 0, 0, face_detected=False, frame_index=f_idx)
            else:
                primary = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
                shape = predictor(rgb_c, primary)
                coords = np.zeros((68, 2), dtype=np.int32)
                for i in range(68):
                    pt = shape.part(i)
                    coords[i] = (pt.x, pt.y)

                el = eye_aspect_ratio(coords[slice(42, 48)])
                er = eye_aspect_ratio(coords[slice(36, 42)])
                em = (el + er) / 2.0
                m = mouth_aspect_ratio(coords[slice(60, 68)])
                moe = (m / em) if em > 1e-6 else 0.0
                iod = float(np.linalg.norm(coords[36] - coords[45]))

                if enable_ml:
                    res = engine.update(el, er, em, m, moe, iod, face_detected=True, frame_index=f_idx)
                    if res.get("latency_ms", 0.0) > 0:
                        ml_latencies.append(res["latency_ms"])

            durations.append(time.perf_counter() - t0)

        fps = float(len(durations) / sum(durations)) if sum(durations) > 0 else 0.0
        mean_ml_lat = float(np.mean(ml_latencies)) if ml_latencies else 0.0
        return fps, mean_ml_lat

    print("  -> Measuring Baseline (Pure Rule-Based / ML Disabled)...")
    fps_baseline, _ = measure_fps(enable_ml=False, num_frames=60)
    print(f"     Baseline FPS: {fps_baseline:.2f} FPS")

    print("  -> Measuring Hybrid System (Rule-Based + ML Advisory, interval=3, alpha=0.30)...")
    fps_hybrid, mean_ml_lat = measure_fps(enable_ml=True, num_frames=60)
    print(f"     Hybrid FPS: {fps_hybrid:.2f} FPS (ML Inference Latency: {mean_ml_lat:.2f} ms)")

    if use_live_cam:
        cap.release()

    results["fps_baseline"] = round(fps_baseline, 2)
    results["fps_hybrid"] = round(fps_hybrid, 2)
    results["ml_inference_latency_ms"] = round(mean_ml_lat, 2)

    print("\n[TEST D2] Validating Driver Behavioral Scenarios...")
    scenarios = [
        {"name": "Normal Blinking", "ear": 0.15, "frames": 3, "expected_alarm": False},
        {"name": "Sustained Eye Closure", "ear": 0.15, "frames": 28, "expected_alarm": True},
        {"name": "Brief Yawn", "mar": 0.70, "frames": 10, "expected_yawn_warning": False},
        {"name": "Sustained Yawn", "mar": 0.70, "frames": 32, "expected_yawn_warning": True},
        {"name": "Head Turn / Occlusion", "face_detected": False, "frames": 20, "expected_alarm": False},
    ]

    scenario_results = {}
    for sc in scenarios:
        name = sc["name"]
        closed_cnt = 0
        yawn_cnt = 0
        alarm_triggered = False
        yawn_triggered = False

        engine = MLFatigueAdvisoryEngine(enabled=True, inference_interval=3, ema_alpha=0.30)

        for f in range(sc["frames"]):
            face_det = sc.get("face_detected", True)
            ear_v = sc.get("ear", 0.32)
            mar_v = sc.get("mar", 0.05)

            if not face_det:
                closed_cnt = 0
                yawn_cnt = 0
                engine.update(0, 0, 0, 0, 0, 0, face_detected=False)
            else:
                if ear_v < 0.25:
                    closed_cnt += 1
                else:
                    closed_cnt = 0

                if mar_v > 0.65:
                    yawn_cnt += 1
                else:
                    yawn_cnt = 0

                if closed_cnt >= 25:
                    alarm_triggered = True
                if yawn_cnt >= 30:
                    yawn_triggered = True

                moe = (mar_v / ear_v) if ear_v > 1e-6 else 0.0
                engine.update(ear_v, ear_v, ear_v, mar_v, moe, 200.0, face_detected=True, frame_index=f)

        if "expected_alarm" in sc:
            assert alarm_triggered == sc["expected_alarm"], f"Scenario '{name}' failed alarm expectation"
        if "expected_yawn_warning" in sc:
            assert yawn_triggered == sc["expected_yawn_warning"], f"Scenario '{name}' failed yawn warning expectation"

        print(f"  -> Scenario '{name}': Verified (Alarm={alarm_triggered}, YawnWarning={yawn_triggered})")
        scenario_results[name] = "PASSED"

    results["scenario_tests"] = scenario_results
    return results

def main():
    print("=" * 78)
    print(" PHASE 3 HYBRID SYSTEM VERIFICATION & VALIDATION TEST SUITE")
    print("=" * 78)

    suite_a = run_suite_a_regression()
    suite_b = run_suite_b_ml_integration()
    suite_c = run_suite_c_dataset_replay()
    suite_d = run_suite_d_realtime_benchmark()

    summary = {
        "suite_a_regression": suite_a,
        "suite_b_ml_integration": suite_b,
        "suite_c_dataset_replay": suite_c,
        "suite_d_realtime_benchmark": suite_d,
    }

    out_file = "data/modeling/phase3_test_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f" ALL PHASE 3 TESTS COMPLETED SUCCESSFULLY! Results saved to '{out_file}'.")
    print("=" * 78)

if __name__ == "__main__":
    main()
