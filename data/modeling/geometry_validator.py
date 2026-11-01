import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

@dataclass
class GeometryValidatorConfig:
    min_face_width_px: float = 60.0
    min_face_height_px: float = 60.0

    min_iod_px: float = 40.0
    max_iod_px: float = 620.0

    min_ear: float = 0.00
    max_ear: float = 0.55
    min_eye_width_px: float = 8.0

    min_mar: float = 0.00
    max_mar: float = 1.80

    max_ear_asymmetry_ratio: float = 1.50

    denominator_epsilon: float = 1e-6

@dataclass
class ValidationResult:
    is_valid: bool
    status_code: str
    reasons: List[str] = field(default_factory=list)
    action: str = "PROCEED"
    sanitized_values: Optional[Dict[str, float]] = None

class GeometryValidator:
    VALIDATED_CHECKS = [
        "NonFiniteCheck",
        "FaceDetectionPresence",
        "BoundingBoxResolution",
        "InterocularDistanceRange",
        "EARPhysiologicalRange",
        "MARPhysiologicalRange",
        "DegenerateEyeSpan"
    ]

    EXPERIMENTAL_CHECKS = [
        "OcularAsymmetryRatio",
        "PoseDependentIODScale"
    ]

    def __init__(self, config: Optional[GeometryValidatorConfig] = None):
        self.config = config or GeometryValidatorConfig()

    def validate_measurements(
        self,
        ear_left: float,
        ear_right: float,
        ear_mean: float,
        mar: float,
        mouth_over_ear: float,
        interocular_dist: float,
        face_detected: bool,
        face_box: Optional[Tuple[int, int, int, int]] = None,
        horizontal_span_left: Optional[float] = None,
        horizontal_span_right: Optional[float] = None,
    ) -> ValidationResult:
        reasons = []

        if not face_detected:
            return ValidationResult(
                is_valid=False,
                status_code="NO_FACE",
                reasons=["Face not detected in frame"],
                action="NO_FACE",
            )

        raw_vals = [ear_left, ear_right, ear_mean, mar, mouth_over_ear, interocular_dist]
        if any(not np.isfinite(v) for v in raw_vals):
            reasons.append("Non-finite measurement encountered (NaN, inf, or -inf)")
            return ValidationResult(
                is_valid=False,
                status_code="NON_FINITE",
                reasons=reasons,
                action="SUSPEND_ML",
            )

        if face_box is not None:
            left, top, right, bottom = face_box
            w = right - left
            h = bottom - top
            if w < self.config.min_face_width_px or h < self.config.min_face_height_px:
                reasons.append(f"Insufficient face resolution ({w}x{h} px < {self.config.min_face_width_px}x{self.config.min_face_height_px} px)")

        if interocular_dist < self.config.min_iod_px:
            reasons.append(f"Interocular distance too small ({interocular_dist:.1f} px < {self.config.min_iod_px} px; profile/far)")
        elif interocular_dist > self.config.max_iod_px:
            reasons.append(f"Interocular distance exceeds optical bounds ({interocular_dist:.1f} px > {self.config.max_iod_px} px)")

        if ear_left < self.config.min_ear or ear_right < self.config.min_ear:
            reasons.append(f"Negative EAR measurement (left={ear_left:.3f}, right={ear_right:.3f})")
        if ear_left > self.config.max_ear or ear_right > self.config.max_ear or ear_mean > self.config.max_ear:
            reasons.append(f"EAR exceeds physiological human maximum ({max(ear_left, ear_right, ear_mean):.3f} > {self.config.max_ear})")

        if horizontal_span_left is not None and horizontal_span_left < self.config.min_eye_width_px:
            reasons.append(f"Collapsed left eye landmarks (width={horizontal_span_left:.1f} px < {self.config.min_eye_width_px} px)")
        if horizontal_span_right is not None and horizontal_span_right < self.config.min_eye_width_px:
            reasons.append(f"Collapsed right eye landmarks (width={horizontal_span_right:.1f} px < {self.config.min_eye_width_px} px)")

        if mar < self.config.min_mar:
            reasons.append(f"Negative MAR measurement ({mar:.3f})")
        elif mar > self.config.max_mar:
            reasons.append(f"MAR exceeds physiological human maximum ({mar:.3f} > {self.config.max_mar})")

        min_ear_side = min(ear_left, ear_right)
        max_ear_side = max(ear_left, ear_right)
        if min_ear_side > 0.08:
            asym_ratio = (max_ear_side - min_ear_side) / (min_ear_side + self.config.denominator_epsilon)
            if asym_ratio > self.config.max_ear_asymmetry_ratio:
                reasons.append(f"Severe ocular asymmetry (ratio={asym_ratio:.2f} > {self.config.max_ear_asymmetry_ratio}; possible wink/shadow)")

        if reasons:
            return ValidationResult(
                is_valid=False,
                status_code="INVALID_GEOMETRY",
                reasons=reasons,
                action="SUSPEND_ML",
            )

        return ValidationResult(
            is_valid=True,
            status_code="VALID",
            reasons=[],
            action="PROCEED",
        )

def run_self_test():
    print("=" * 70)
    print("  RUNNING GEOMETRY VALIDATOR SELF-TEST")
    print("=" * 70)

    validator = GeometryValidator()

    res_normal = validator.validate_measurements(
        ear_left=0.31, ear_right=0.32, ear_mean=0.315,
        mar=0.05, mouth_over_ear=0.158, interocular_dist=205.0,
        face_detected=True, face_box=(100, 100, 400, 400),
        horizontal_span_left=32.0, horizontal_span_right=33.0
    )
    assert res_normal.is_valid, f"Normal driver failed validation: {res_normal.reasons}"
    print("[PASS] Normal alert measurement verified: VALID")

    res_blink = validator.validate_measurements(
        ear_left=0.08, ear_right=0.07, ear_mean=0.075,
        mar=0.04, mouth_over_ear=0.533, interocular_dist=205.0,
        face_detected=True, face_box=(100, 100, 400, 400),
        horizontal_span_left=30.0, horizontal_span_right=31.0
    )
    assert res_blink.is_valid, f"Genuine eye closure should be valid: {res_blink.reasons}"
    print("[PASS] Genuine eye closure verified: VALID")

    res_nan = validator.validate_measurements(
        ear_left=float("nan"), ear_right=0.30, ear_mean=0.30,
        mar=0.05, mouth_over_ear=0.16, interocular_dist=200.0,
        face_detected=True
    )
    assert not res_nan.is_valid and res_nan.status_code == "NON_FINITE"
    print("[PASS] NaN input safely rejected: NON_FINITE (SUSPEND_ML)")

    res_huge_ear = validator.validate_measurements(
        ear_left=0.68, ear_right=0.72, ear_mean=0.70,
        mar=0.05, mouth_over_ear=0.07, interocular_dist=200.0,
        face_detected=True
    )
    assert not res_huge_ear.is_valid and "EAR exceeds" in res_huge_ear.reasons[0]
    print("[PASS] Implausible EAR (> 0.55) rejected: INVALID_GEOMETRY")

    res_collapsed = validator.validate_measurements(
        ear_left=0.25, ear_right=0.25, ear_mean=0.25,
        mar=0.05, mouth_over_ear=0.20, interocular_dist=200.0,
        face_detected=True, horizontal_span_left=3.2, horizontal_span_right=32.0
    )
    assert not res_collapsed.is_valid and "Collapsed left eye" in res_collapsed.reasons[0]
    print("[PASS] Collapsed landmark span rejected: INVALID_GEOMETRY")

    res_asym = validator.validate_measurements(
        ear_left=0.38, ear_right=0.11, ear_mean=0.245,
        mar=0.05, mouth_over_ear=0.20, interocular_dist=200.0,
        face_detected=True
    )
    assert not res_asym.is_valid and "Severe ocular asymmetry" in res_asym.reasons[0]
    print("[PASS] Severe eye asymmetry rejected: INVALID_GEOMETRY")

    res_lowres = validator.validate_measurements(
        ear_left=0.30, ear_right=0.30, ear_mean=0.30,
        mar=0.05, mouth_over_ear=0.16, interocular_dist=80.0,
        face_detected=True, face_box=(10, 10, 55, 55)
    )
    assert not res_lowres.is_valid and "Insufficient face resolution" in res_lowres.reasons[0]
    print("[PASS] Low-resolution face rejected: INVALID_GEOMETRY")

    print("\nAll GeometryValidator self-tests passed successfully.")

if __name__ == "__main__":
    run_self_test()
