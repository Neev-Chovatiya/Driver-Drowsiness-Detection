import os
import sys
import time
import argparse
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import joblib

class AudioBuzzer:
    def __init__(self, frequency: int = 1100, sample_rate: int = 22050, buffer_size: int = 512):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_playing = False
        self.sound = None
        self.channel = None

        self._initialize_mixer()
        self._synthesize_waveform()

    def _initialize_mixer(self):
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.mixer.init(
                frequency=self.sample_rate,
                size=-16,
                channels=2,
                buffer=self.buffer_size
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize Pygame mixer: {e}", file=sys.stderr)

    def _synthesize_waveform(self):
        duration_sec = 0.10
        n_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        amplitude = 32767 * 0.90
        mono_wave = (amplitude * np.sin(2.0 * np.pi * self.frequency * t)).astype(np.int16)

        stereo_wave = np.column_stack((mono_wave, mono_wave))
        stereo_wave = np.ascontiguousarray(stereo_wave, dtype=np.int16)

        self.sound = pygame.sndarray.make_sound(stereo_wave)

    def start_alert(self):
        if not self.is_playing and self.sound is not None:
            self.channel = self.sound.play(loops=-1)
            self.is_playing = True

    def stop_alert(self):
        if self.is_playing:
            if self.sound is not None:
                self.sound.stop()
            self.is_playing = False

    def shutdown(self):
        self.stop_alert()
        try:
            pygame.mixer.quit()
        except Exception:
            pass

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

class MLFatigueAdvisoryEngine:
    CLASS_NAMES = ["ALERT", "LOW VIGILANT", "DROWSY"]
    CLASS_COLORS = [
        (34, 187, 51),
        (0, 140, 255),
        (0, 0, 220)
    ]

    def __init__(
        self,
        model_path: str = "data/modeling/models/expB_randomforest.joblib",
        enabled: bool = True,
        inference_interval: int = 3,
        ema_alpha: float = 0.30,
        log_predictions: bool = False,
        log_file: str = "data/modeling/realtime_ml_log.csv"
    ):
        self.model_path = model_path
        self.enabled = enabled
        self.inference_interval = max(1, inference_interval)
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.log_predictions = log_predictions
        self.log_file = log_file

        self.pipeline = None
        self.is_active = False
        self.status_message = "ML: INITIALIZING"

        self.frame_counter = 0
        self.latest_raw_probs = None
        self.smoothed_probs = None
        self.predicted_class_idx = 0
        self.consecutive_missing_frames = 0
        self.last_inference_time_ms = 0.0

        if self.enabled:
            self._load_pipeline()
        else:
            self.status_message = "ML: DISABLED (RULE BASELINE)"

    def _load_pipeline(self):
        if not os.path.exists(self.model_path):
            print(f"[WARN] ML model file not found: '{self.model_path}'. Running in rule-based fallback mode.", file=sys.stderr)
            self.is_active = False
            self.status_message = "ML: OFFLINE (FALLBACK)"
            return

        try:
            self.pipeline = joblib.load(self.model_path)
            if not hasattr(self.pipeline, "predict_proba"):
                raise ValueError("Loaded pipeline does not implement predict_proba()")

            clf = self.pipeline.named_steps.get("clf")
            if clf and hasattr(clf, "n_features_in_") and clf.n_features_in_ != 7:
                raise ValueError(f"Pipeline expects {clf.n_features_in_} features, but schema requires 7.")

            if clf and hasattr(clf, "classes_"):
                if list(clf.classes_) != [0, 1, 2]:
                    raise ValueError(f"Unexpected classes {clf.classes_}, expected [0, 1, 2].")

            self.is_active = True
            self.status_message = "ML: ONLINE (ADVISORY)"
            print(f"[INFO] ML Fatigue Advisory Engine loaded successfully from '{self.model_path}'.")

            if self.log_predictions:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
                if not os.path.exists(self.log_file):
                    with open(self.log_file, "w", encoding="utf-8") as f:
                        f.write("timestamp,frame_index,ear,mar,moe,iod,p_alert,p_low_vig,p_drowsy,predicted_state,ml_active\n")

        except Exception as e:
            print(f"[ERROR] Failed to load ML model '{self.model_path}': {e}. Falling back to rule baseline.", file=sys.stderr)
            self.pipeline = None
            self.is_active = False
            self.status_message = "ML: OFFLINE (LOAD ERROR)"

    def update(
        self,
        ear_left: float,
        ear_right: float,
        ear_mean: float,
        mar: float,
        mouth_over_ear: float,
        interocular_dist: float,
        face_detected: bool,
        frame_index: int = 0
    ) -> Dict[str, Any]:
        self.frame_counter += 1

        if not self.is_active or self.pipeline is None:
            return {
                "active": False,
                "status_text": self.status_message,
                "class_name": "N/A",
                "class_idx": -1,
                "confidence": 0.0,
                "color": (130, 130, 140),
                "probs": [0.0, 0.0, 0.0],
                "latency_ms": 0.0
            }

        if not face_detected:
            self.consecutive_missing_frames += 1
            self.smoothed_probs = None
            return {
                "active": True,
                "status_text": "ML: NO FACE",
                "class_name": "NO FACE",
                "class_idx": -1,
                "confidence": 0.0,
                "color": (75, 75, 75),
                "probs": [0.0, 0.0, 0.0],
                "latency_ms": 0.0
            }

        self.consecutive_missing_frames = 0

        should_infer = (self.frame_counter % self.inference_interval == 0) or (self.smoothed_probs is None)

        if should_infer:
            t0 = time.perf_counter()

            el = float(np.clip(np.nan_to_num(ear_left, nan=0.30, posinf=0.30, neginf=0.30), 0.0, 2.0))
            er = float(np.clip(np.nan_to_num(ear_right, nan=0.30, posinf=0.30, neginf=0.30), 0.0, 2.0))
            em = float(np.clip(np.nan_to_num(ear_mean, nan=0.30, posinf=0.30, neginf=0.30), 0.0, 2.0))
            m = float(np.clip(np.nan_to_num(mar, nan=0.05, posinf=0.05, neginf=0.05), 0.0, 3.0))
            moe = float(np.clip(np.nan_to_num(mouth_over_ear, nan=0.15, posinf=0.15, neginf=0.15), 0.0, 50.0))
            iod = float(np.clip(np.nan_to_num(interocular_dist, nan=200.0, posinf=200.0, neginf=200.0), 10.0, 1000.0))
            missing_flag = 0.0

            feature_vec = np.array([[el, er, em, m, moe, iod, missing_flag]], dtype=np.float64)

            try:
                raw_probs = self.pipeline.predict_proba(feature_vec)[0]
                p_sum = np.sum(raw_probs)
                if p_sum > 1e-6 and np.all(np.isfinite(raw_probs)):
                    raw_probs = raw_probs / p_sum
                else:
                    raw_probs = np.array([0.333, 0.333, 0.334])

                self.latest_raw_probs = raw_probs

                if self.smoothed_probs is None:
                    self.smoothed_probs = raw_probs.copy()
                else:
                    self.smoothed_probs = (self.ema_alpha * raw_probs) + ((1.0 - self.ema_alpha) * self.smoothed_probs)
                    self.smoothed_probs /= np.sum(self.smoothed_probs)

                self.predicted_class_idx = int(np.argmax(self.smoothed_probs))
                self.last_inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

                if self.log_predictions and (self.frame_counter % 10 == 0):
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now().isoformat()},{frame_index},{em:.4f},{m:.4f},{moe:.4f},{iod:.2f},"
                                f"{self.smoothed_probs[0]:.4f},{self.smoothed_probs[1]:.4f},{self.smoothed_probs[2]:.4f},"
                                f"{self.CLASS_NAMES[self.predicted_class_idx]},1\n")

            except Exception as e:
                print(f"[WARN] ML inference error: {e}", file=sys.stderr)
                self.last_inference_time_ms = 0.0

        if self.smoothed_probs is None:
            self.smoothed_probs = np.array([0.333, 0.333, 0.334])

        idx = self.predicted_class_idx
        conf = float(self.smoothed_probs[idx])
        c_name = self.CLASS_NAMES[idx]
        c_color = self.CLASS_COLORS[idx]

        return {
            "active": True,
            "status_text": f"ML: {c_name} ({conf*100:.0f}%)",
            "class_name": c_name,
            "class_idx": idx,
            "confidence": conf,
            "color": c_color,
            "probs": self.smoothed_probs.tolist(),
            "latency_ms": self.last_inference_time_ms
        }

class HUDVisualizer:
    COLOR_ALERT = (34, 187, 51)
    COLOR_DROWSY = (0, 0, 220)
    COLOR_YAWN = (0, 140, 255)
    COLOR_NO_FACE = (75, 75, 75)
    COLOR_CONTOUR_EYE = (0, 255, 255)
    COLOR_CONTOUR_MOUTH = (0, 215, 255)
    COLOR_TEXT_WHITE = (255, 255, 255)
    COLOR_TEXT_DIM = (190, 190, 190)

    @staticmethod
    def draw_contours(frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray, inner_mouth: np.ndarray):
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        mouth_hull = cv2.convexHull(inner_mouth)

        cv2.drawContours(frame, [left_eye_hull], -1, HUDVisualizer.COLOR_CONTOUR_EYE, 1, lineType=cv2.LINE_AA)
        cv2.drawContours(frame, [right_eye_hull], -1, HUDVisualizer.COLOR_CONTOUR_EYE, 1, lineType=cv2.LINE_AA)
        cv2.drawContours(frame, [mouth_hull], -1, HUDVisualizer.COLOR_CONTOUR_MOUTH, 1, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_telemetry_card(
        frame: np.ndarray,
        ear: float,
        mar: float,
        closed_frames: int,
        ear_limit: int,
        yawn_frames: int,
        mar_limit: int,
        fps: float,
        ml_status: Optional[Dict[str, Any]] = None
    ):
        card_x, card_y = 16, 16
        if ml_status and ml_status.get("active"):
            card_w, card_h = 280, 188
        elif ml_status and not ml_status.get("active"):
            card_w, card_h = 280, 164
        else:
            card_w, card_h = 260, 140

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 25), -1)

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 75), 1, lineType=cv2.LINE_AA)

        cv2.putText(frame, "DRIVER TELEMETRY", (card_x + 12, card_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (100, 220, 255), 1, cv2.LINE_AA)

        ear_color = HUDVisualizer.COLOR_ALERT if ear >= 0.25 else HUDVisualizer.COLOR_DROWSY
        cv2.putText(frame, f"EAR: {ear:.3f}", (card_x + 12, card_y + 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, ear_color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"Closed: {closed_frames:02d} / {ear_limit:02d}", (card_x + 130, card_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUDVisualizer.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        mar_color = HUDVisualizer.COLOR_ALERT if mar <= 0.65 else HUDVisualizer.COLOR_YAWN
        cv2.putText(frame, f"MAR: {mar:.3f}", (card_x + 12, card_y + 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, mar_color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"Yawn:   {yawn_frames:02d} / {mar_limit:02d}", (card_x + 130, card_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUDVisualizer.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        curr_y = card_y + 106
        if ml_status and ml_status.get("active"):
            c_name = ml_status.get("class_name", "ALERT")
            c_conf = ml_status.get("confidence", 0.0)
            c_color = ml_status.get("color", (34, 187, 51))
            probs = ml_status.get("probs", [0.0, 0.0, 0.0])

            if c_name == "NO FACE":
                cv2.putText(frame, "ML Advisory: NO FACE", (card_x + 12, curr_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, (120, 120, 130), 1, cv2.LINE_AA)
                curr_y += 20
                cv2.putText(frame, "P(A:-- | LV:-- | D:--)", (card_x + 12, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)
                curr_y += 20
            else:
                cv2.putText(frame, f"ML Advisory: {c_name} ({c_conf*100:.0f}%)", (card_x + 12, curr_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, c_color, 1, cv2.LINE_AA)
                curr_y += 20
                p_text = f"P(A:{probs[0]:.2f} | LV:{probs[1]:.2f} | D:{probs[2]:.2f})"
                cv2.putText(frame, p_text, (card_x + 12, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 210, 240), 1, cv2.LINE_AA)
                curr_y += 20
        elif ml_status and not ml_status.get("active"):
            status_txt = ml_status.get("status_text", "ML: OFFLINE")
            cv2.putText(frame, status_txt, (card_x + 12, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 140), 1, cv2.LINE_AA)
            curr_y += 20

        engine_str = "HOG+RF" if (ml_status and ml_status.get("active")) else "HOG+SVM"
        cv2.putText(frame, f"FPS: {fps:.1f} | Dlib {engine_str}", (card_x + 12, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 210, 150), 1, cv2.LINE_AA)
        curr_y += 18

        cv2.putText(frame, "Press 'Q' or 'ESC' to exit", (card_x + 12, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 130), 1, cv2.LINE_AA)

    @staticmethod
    def draw_status_banner(frame: np.ndarray, status: str, banner_color: Tuple[int, int, int]):
        h, w = frame.shape[:2]
        banner_h = 52
        banner_y1 = h - banner_h
        banner_y2 = h

        cv2.rectangle(frame, (0, banner_y1), (w, banner_y2), banner_color, -1)

        cv2.line(frame, (0, banner_y1), (w, banner_y1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.85
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(status, font, font_scale, thickness)
        text_x = max(10, (w - text_w) // 2)
        text_y = banner_y1 + (banner_h + text_h) // 2 - 2

        cv2.putText(frame, status, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, status, (text_x, text_y), font, font_scale, HUDVisualizer.COLOR_TEXT_WHITE, thickness, cv2.LINE_AA)

class DriverFatiguePipeline:
    LEFT_EYE_SLICE = slice(42, 48)
    RIGHT_EYE_SLICE = slice(36, 42)
    INNER_MOUTH_SLICE = slice(60, 68)

    def __init__(
        self,
        model_path: str = "shape_predictor_68_face_landmarks.dat",
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        ear_thresh: float = 0.25,
        ear_frames_limit: int = 25,
        mar_thresh: float = 0.65,
        mar_frames_limit: int = 30,
        ml_model_path: str = "data/modeling/models/expB_randomforest.joblib",
        enable_ml: bool = True,
        ml_interval: int = 3,
        ml_alpha: float = 0.30,
        log_ml: bool = False
    ):
        self.model_path = model_path
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.ear_thresh = ear_thresh
        self.ear_frames_limit = ear_frames_limit
        self.mar_thresh = mar_thresh
        self.mar_frames_limit = mar_frames_limit

        self.ml_model_path = ml_model_path
        self.enable_ml = enable_ml
        self.ml_interval = ml_interval
        self.ml_alpha = ml_alpha
        self.log_ml = log_ml

        self.closed_frame_counter = 0
        self.yawn_frame_counter = 0

        self.detector = None
        self.predictor = None
        self.buzzer = None
        self.ml_engine = None
        self.cap = None

    def initialize_subsystems(self):
        if not os.path.exists(self.model_path):
            print(f"[FATAL] Landmark model not found: '{self.model_path}'", file=sys.stderr)
            print("        Please run: python setup_model.py", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Initializing HOG + Linear SVM Face Detector...")
        self.detector = dlib.get_frontal_face_detector()

        print(f"[INFO] Loading 68-Point Facial Landmark Predictor from '{self.model_path}'...")
        try:
            self.predictor = dlib.shape_predictor(self.model_path)
        except Exception as e:
            print(f"[FATAL] Failed to load landmark model: {e}", file=sys.stderr)
            sys.exit(1)

        print("[INFO] Initializing 2-Channel Stereo Pygame Synthesizer (1100 Hz)...")
        self.buzzer = AudioBuzzer(frequency=1100, sample_rate=22050, buffer_size=512)

        print(f"[INFO] Initializing ML Advisory Subsystem (enabled={self.enable_ml})...")
        self.ml_engine = MLFatigueAdvisoryEngine(
            model_path=self.ml_model_path,
            enabled=self.enable_ml,
            inference_interval=self.ml_interval,
            ema_alpha=self.ml_alpha,
            log_predictions=self.log_ml
        )

        print(f"[INFO] Opening Camera device index [{self.camera_index}]...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[WARN] Camera index [{self.camera_index}] failed to open. Retrying with cv2.CAP_DSHOW...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print(f"[FATAL] Could not access camera (index {self.camera_index}). Check camera connections.", file=sys.stderr)
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        print("[INFO] System online. Monitoring driver alertness...")

    def run(self):
        self.initialize_subsystems()

        window_title = "Production Driver Drowsiness & Fatigue Detector (HOG + SVM + RF)"
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        fps_timer = time.time()
        fps = 0.0
        frame_counter = 0

        try:
            while True:
                loop_start = time.time()
                frame_counter += 1

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[ERROR] Camera frame grab failed. Retrying...", file=sys.stderr)
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)

                rects = self.detector(rgb_contiguous, 0)

                status_text = "STATUS: ALERT (NORMAL)"
                status_color = HUDVisualizer.COLOR_ALERT
                current_ear = 0.0
                current_mar = 0.0
                ml_status = None

                if len(rects) == 0:
                    status_text = "STATUS: NO FACE DETECTED"
                    status_color = HUDVisualizer.COLOR_NO_FACE
                    self.closed_frame_counter = 0
                    self.yawn_frame_counter = 0
                    self.buzzer.stop_alert()

                    if self.ml_engine is not None:
                        ml_status = self.ml_engine.update(
                            ear_left=0.0,
                            ear_right=0.0,
                            ear_mean=0.0,
                            mar=0.0,
                            mouth_over_ear=0.0,
                            interocular_dist=0.0,
                            face_detected=False,
                            frame_index=frame_counter
                        )
                else:
                    primary_rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

                    shape = self.predictor(rgb_contiguous, primary_rect)
                    coords = np.zeros((68, 2), dtype=np.int32)
                    for i in range(68):
                        pt = shape.part(i)
                        coords[i] = (pt.x, pt.y)

                    left_eye = coords[self.LEFT_EYE_SLICE]
                    right_eye = coords[self.RIGHT_EYE_SLICE]
                    inner_mouth = coords[self.INNER_MOUTH_SLICE]

                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    current_ear = (left_ear + right_ear) / 2.0
                    current_mar = mouth_aspect_ratio(inner_mouth)
                    mouth_over_ear = (current_mar / current_ear) if current_ear > 1e-6 else 0.0
                    interocular_dist = float(dist.euclidean(coords[36], coords[45]))

                    if self.ml_engine is not None:
                        ml_status = self.ml_engine.update(
                            ear_left=left_ear,
                            ear_right=right_ear,
                            ear_mean=current_ear,
                            mar=current_mar,
                            mouth_over_ear=mouth_over_ear,
                            interocular_dist=interocular_dist,
                            face_detected=True,
                            frame_index=frame_counter
                        )

                    HUDVisualizer.draw_contours(frame, left_eye, right_eye, inner_mouth)

                    if current_ear < self.ear_thresh:
                        self.closed_frame_counter += 1
                    else:
                        self.closed_frame_counter = 0

                    if current_mar > self.mar_thresh:
                        self.yawn_frame_counter += 1
                    else:
                        self.yawn_frame_counter = 0

                    is_drowsy = self.closed_frame_counter >= self.ear_frames_limit
                    is_yawning = self.yawn_frame_counter >= self.mar_frames_limit

                    if is_drowsy:
                        status_text = "CRITICAL: DROWSINESS DETECTED!"
                        status_color = HUDVisualizer.COLOR_DROWSY
                        self.buzzer.start_alert()
                    elif is_yawning:
                        status_text = "WARNING: YAWNING DETECTED!"
                        status_color = HUDVisualizer.COLOR_YAWN
                        self.buzzer.stop_alert()
                    else:
                        status_text = "STATUS: ALERT (NORMAL)"
                        status_color = HUDVisualizer.COLOR_ALERT
                        self.buzzer.stop_alert()

                loop_duration = time.time() - loop_start
                if loop_duration > 0:
                    current_fps = 1.0 / loop_duration
                    fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

                HUDVisualizer.draw_telemetry_card(
                    frame=frame,
                    ear=current_ear,
                    mar=current_mar,
                    closed_frames=self.closed_frame_counter,
                    ear_limit=self.ear_frames_limit,
                    yawn_frames=self.yawn_frame_counter,
                    mar_limit=self.mar_frames_limit,
                    fps=fps,
                    ml_status=ml_status
                )
                HUDVisualizer.draw_status_banner(frame, status_text, status_color)

                cv2.imshow(window_title, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("[INFO] Shutdown signal received from user.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt detected. Shutting down gracefully...")

        finally:
            self.cleanup()

    def cleanup(self):
        print("[INFO] Releasing camera capture device...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        print("[INFO] Shutting down Pygame audio mixer...")
        if self.buzzer is not None:
            self.buzzer.shutdown()

        print("[INFO] Closing OpenCV display windows...")
        cv2.destroyAllWindows()
        print("[INFO] System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Production Real-Time Driver Drowsiness and Fatigue Detection System (HOG + Linear SVM + Random Forest)"
    )
    parser.add_argument(
        "--model", default="shape_predictor_68_face_landmarks.dat",
        help="Path to 68-point facial landmark weights file (default: shape_predictor_68_face_landmarks.dat)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera capture frame width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera capture frame height (default: 480)"
    )
    parser.add_argument(
        "--ear-thresh", type=float, default=0.25,
        help="Eye Aspect Ratio threshold triggering drowsiness (default: 0.25)"
    )
    parser.add_argument(
        "--ear-frames", type=int, default=25,
        help="Consecutive frames below EAR threshold before alarm triggers (default: 25)"
    )
    parser.add_argument(
        "--mar-thresh", type=float, default=0.65,
        help="Mouth Aspect Ratio threshold triggering yawning (default: 0.65)"
    )
    parser.add_argument(
        "--mar-frames", type=int, default=30,
        help="Consecutive frames above MAR threshold before warning triggers (default: 30)"
    )
    parser.add_argument(
        "--ml-model", default="data/modeling/models/expB_randomforest.joblib",
        help="Path to trained Random Forest model joblib file (default: data/modeling/models/expB_randomforest.joblib)"
    )
    parser.add_argument(
        "--disable-ml", action="store_true",
        help="Disable ML advisory engine and run purely on rule-based baseline"
    )
    parser.add_argument(
        "--ml-interval", type=int, default=3,
        help="Frame cadence interval for ML inference (default: 3 frames)"
    )
    parser.add_argument(
        "--ml-alpha", type=float, default=0.30,
        help="Exponential moving average smoothing alpha for ML probabilities (default: 0.30)"
    )
    parser.add_argument(
        "--log-ml", action="store_true",
        help="Log ML predictions and timestamps to CSV"
    )

    args = parser.parse_args()

    pipeline = DriverFatiguePipeline(
        model_path=args.model,
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        ear_thresh=args.ear_thresh,
        ear_frames_limit=args.ear_frames,
        mar_thresh=args.mar_thresh,
        mar_frames_limit=args.mar_frames,
        ml_model_path=args.ml_model,
        enable_ml=not args.disable_ml,
        ml_interval=args.ml_interval,
        ml_alpha=args.ml_alpha,
        log_ml=args.log_ml
    )
    pipeline.run()

if __name__ == "__main__":
    main()
