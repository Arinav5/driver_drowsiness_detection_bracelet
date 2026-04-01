"""
utils.py — Core utilities for Driver Drowsiness & Posture Detection

Uses MediaPipe Pose (33 body landmarks) for FULL BODY posture analysis.
Reference: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

The 33 Pose Landmarks:
    0  - nose               11 - left shoulder     23 - left hip
    1  - left eye (inner)   12 - right shoulder    24 - right hip
    2  - left eye            13 - left elbow        25 - left knee
    3  - left eye (outer)   14 - right elbow       26 - right knee
    4  - right eye (inner)  15 - left wrist        27 - left ankle
    5  - right eye           16 - right wrist       28 - right ankle
    6  - right eye (outer)  17 - left pinky        29 - left heel
    7  - left ear            18 - right pinky       30 - right heel
    8  - right ear           19 - left index        31 - left foot index
    9  - mouth (left)       20 - right index       32 - right foot index
    10 - mouth (right)      21 - left thumb
                             22 - right thumb

Detects:
  1. Eye closure  — EAR (Eye Aspect Ratio) from eye landmarks 1-6
  2. Head nod     — nose (0) dropping relative to shoulder midpoint (11,12)
  3. Posture      — shoulder-ear alignment, torso lean angle
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# POSE LANDMARK INDICES (from mediapipe.solutions.pose.PoseLandmark)
# ═══════════════════════════════════════════════════════════════════
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Eye landmark groups for EAR calculation
# Using pose landmarks: inner eye corner, eye center, outer eye corner,
# plus ear and nose sides as vertical references
# NOTE: Pose has only 3 points per eye (inner, center, outer) — not 6
# like Face Mesh. We approximate EAR differently below.
LEFT_EYE_POINTS = [LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER]
RIGHT_EYE_POINTS = [RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER]


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
@dataclass
class DetectionConfig:
    """All tunable parameters for the detection system."""

    # --- Eye closure ---
    # Pose model gives visibility scores for eye landmarks.
    # When eyes are closed, visibility drops significantly.
    eye_visibility_threshold: float = 0.5
    eye_consec_frames: int = 15        # ~1 sec at 15 fps

    # --- Head nod (nose drops relative to shoulder midpoint) ---
    nod_ratio_threshold: float = 0.15  # nose-to-shoulder Y ratio change
    nod_consec_frames: int = 10

    # --- Posture: shoulder-ear vertical alignment ---
    # When driver slumps, the angle between ear-shoulder-hip deviates
    posture_angle_threshold: float = 35.0  # degrees off vertical
    posture_consec_frames: int = 20

    # --- Alert timing ---
    alert_cooldown_sec: float = 5.0

    # --- Camera ---
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # --- Serial (ESP32C3) ---
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200


# ═══════════════════════════════════════════════════════════════════
# DETECTION STATE
# ═══════════════════════════════════════════════════════════════════
@dataclass
class DetectionState:
    """Mutable state tracked across frames."""

    eye_closed_count: int = 0
    nod_count: int = 0
    posture_bad_count: int = 0

    # Baseline values (set during calibration)
    baseline_nose_shoulder_ratio: Optional[float] = None
    calibration_frames: int = 0
    calibration_ratios: list = None

    is_alerting: bool = False
    last_alert_time: float = 0.0
    frame_number: int = 0

    # Display values
    eye_visibility: float = 0.0
    nose_shoulder_ratio: float = 0.0
    posture_angle: float = 0.0
    drowsy_reason: str = "ok"

    def __post_init__(self):
        if self.calibration_ratios is None:
            self.calibration_ratios = []


# ═══════════════════════════════════════════════════════════════════
# HELPER: get landmark as (x, y) pixel coordinates
# ═══════════════════════════════════════════════════════════════════
def _lm_px(landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
    """Extract a landmark as (x_px, y_px) from normalized coords."""
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _lm_visible(landmarks, idx: int, threshold: float = 0.5) -> bool:
    """Check if a landmark meets the visibility threshold."""
    return landmarks[idx].visibility > threshold


# ═══════════════════════════════════════════════════════════════════
# EYE CLOSURE DETECTION
# ═══════════════════════════════════════════════════════════════════
def compute_eye_closure(landmarks, config: DetectionConfig) -> Tuple[float, bool]:
    """
    Detect eye closure using landmark visibility scores.

    MediaPipe Pose provides a `visibility` field for each landmark.
    When eyes are closed, the eye landmarks become less visible
    (visibility score drops). We average the visibility of all 6
    eye landmarks (3 per eye).

    Returns
    -------
    (avg_visibility, eyes_closed)
    """
    eye_indices = LEFT_EYE_POINTS + RIGHT_EYE_POINTS
    vis_sum = sum(landmarks[i].visibility for i in eye_indices)
    avg_vis = vis_sum / len(eye_indices)

    eyes_closed = avg_vis < config.eye_visibility_threshold
    return avg_vis, eyes_closed


# ═══════════════════════════════════════════════════════════════════
# HEAD NOD DETECTION
# ═══════════════════════════════════════════════════════════════════
def compute_nose_shoulder_ratio(landmarks, w: int, h: int) -> float:
    """
    Compute the vertical ratio of nose position relative to shoulders.

    ratio = (nose_y - shoulder_mid_y) / shoulder_width

    When the head drops (nodding), this ratio increases.
    Normalizing by shoulder width makes it scale-invariant.
    """
    nose_x, nose_y = _lm_px(landmarks, NOSE, w, h)
    ls_x, ls_y = _lm_px(landmarks, LEFT_SHOULDER, w, h)
    rs_x, rs_y = _lm_px(landmarks, RIGHT_SHOULDER, w, h)

    shoulder_mid_y = (ls_y + rs_y) / 2.0
    shoulder_width = abs(ls_x - rs_x)

    if shoulder_width < 1.0:
        return 0.0

    return (nose_y - shoulder_mid_y) / shoulder_width


def detect_head_nod(
    ratio: float,
    state: DetectionState,
    config: DetectionConfig,
) -> bool:
    """
    Detect head nodding by comparing current nose-shoulder ratio
    to the calibrated baseline.

    Returns True if nodding detected for enough consecutive frames.
    """
    if state.baseline_nose_shoulder_ratio is None:
        return False  # Not yet calibrated

    deviation = ratio - state.baseline_nose_shoulder_ratio

    if deviation > config.nod_ratio_threshold:
        state.nod_count += 1
    else:
        state.nod_count = max(0, state.nod_count - 1)

    return state.nod_count >= config.nod_consec_frames


# ═══════════════════════════════════════════════════════════════════
# POSTURE DETECTION (Shoulder-Ear-Hip Alignment)
# ═══════════════════════════════════════════════════════════════════
def compute_posture_angle(landmarks, w: int, h: int) -> float:
    """
    Compute the angle of the torso/neck lean.

    Uses the angle between:
      - Vector: shoulder_midpoint → ear_midpoint
      - Vertical axis (straight up)

    When seated upright, this angle is near 0°.
    When slumping forward, the angle increases.

    Returns angle in degrees.
    """
    ls_x, ls_y = _lm_px(landmarks, LEFT_SHOULDER, w, h)
    rs_x, rs_y = _lm_px(landmarks, RIGHT_SHOULDER, w, h)
    le_x, le_y = _lm_px(landmarks, LEFT_EAR, w, h)
    re_x, re_y = _lm_px(landmarks, RIGHT_EAR, w, h)

    shoulder_mid = ((ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0)
    ear_mid = ((le_x + re_x) / 2.0, (le_y + re_y) / 2.0)

    # Vector from shoulder midpoint to ear midpoint
    dx = ear_mid[0] - shoulder_mid[0]
    dy = ear_mid[1] - shoulder_mid[1]  # Note: y increases downward

    # Angle from vertical (straight up = dy is negative, dx is 0)
    # atan2 of the horizontal deviation vs the vertical component
    # Vertical vector is (0, -1) in screen coords
    angle_rad = math.atan2(abs(dx), abs(dy))
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def detect_bad_posture(
    angle: float,
    state: DetectionState,
    config: DetectionConfig,
) -> bool:
    """
    Check if the driver's posture is bad (leaning/slumping).

    Returns True if bad posture persists for enough consecutive frames.
    """
    if angle > config.posture_angle_threshold:
        state.posture_bad_count += 1
    else:
        state.posture_bad_count = max(0, state.posture_bad_count - 1)

    return state.posture_bad_count >= config.posture_consec_frames


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION (first ~30 frames to establish baseline)
# ═══════════════════════════════════════════════════════════════════
CALIBRATION_FRAME_COUNT = 30


def calibrate(state: DetectionState, ratio: float) -> bool:
    """
    Collect baseline nose-shoulder ratio over the first N frames.

    Returns True when calibration is complete.
    """
    if state.baseline_nose_shoulder_ratio is not None:
        return True  # Already calibrated

    state.calibration_ratios.append(ratio)
    state.calibration_frames += 1

    if state.calibration_frames >= CALIBRATION_FRAME_COUNT:
        state.baseline_nose_shoulder_ratio = (
            sum(state.calibration_ratios) / len(state.calibration_ratios)
        )
        logger.info(
            "Calibration complete — baseline nose/shoulder ratio: %.3f",
            state.baseline_nose_shoulder_ratio,
        )
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# MAIN EVALUATION (runs all checks)
# ═══════════════════════════════════════════════════════════════════
def evaluate_drowsiness(
    landmarks,
    frame_w: int,
    frame_h: int,
    state: DetectionState,
    config: DetectionConfig,
) -> Tuple[bool, str]:
    """
    Run all detection checks on a single frame's pose landmarks.

    Parameters
    ----------
    landmarks : list of mediapipe pose landmarks (33 total)
    frame_w, frame_h : int
    state : DetectionState (mutated in place)
    config : DetectionConfig

    Returns
    -------
    (is_drowsy: bool, reason: str)
        reason is one of: "eyes_closed", "head_nod", "bad_posture", "ok"
    """
    state.frame_number += 1

    # Check that key landmarks are visible enough to analyze
    key_landmarks = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR]
    if not all(_lm_visible(landmarks, i, 0.3) for i in key_landmarks):
        state.drowsy_reason = "landmarks_not_visible"
        return False, "landmarks_not_visible"

    # 1. Eye closure
    eye_vis, eyes_closed = compute_eye_closure(landmarks, config)
    state.eye_visibility = eye_vis

    if eyes_closed:
        state.eye_closed_count += 1
    else:
        state.eye_closed_count = 0

    # 2. Nose-shoulder ratio (for head nod)
    ns_ratio = compute_nose_shoulder_ratio(landmarks, frame_w, frame_h)
    state.nose_shoulder_ratio = ns_ratio

    # Calibrate baseline during first N frames
    is_calibrated = calibrate(state, ns_ratio)

    # 3. Head nod detection
    head_nodding = False
    if is_calibrated:
        head_nodding = detect_head_nod(ns_ratio, state, config)

    # 4. Posture angle
    p_angle = compute_posture_angle(landmarks, frame_w, frame_h)
    state.posture_angle = p_angle
    bad_posture = detect_bad_posture(p_angle, state, config)

    # 5. Drowsiness decision
    if state.eye_closed_count >= config.eye_consec_frames:
        state.drowsy_reason = "eyes_closed"
        return True, "eyes_closed"
    if head_nodding:
        state.drowsy_reason = "head_nod"
        return True, "head_nod"
    if bad_posture:
        state.drowsy_reason = "bad_posture"
        return True, "bad_posture"

    state.drowsy_reason = "ok"
    return False, "ok"


# ═══════════════════════════════════════════════════════════════════
# SERIAL COMMUNICATION (ESP32C3 bracelet)
# ═══════════════════════════════════════════════════════════════════
def open_serial(port: str, baud: int):
    """Open serial connection to ESP32C3. Returns serial object or None."""
    try:
        import serial as pyserial
        ser = pyserial.Serial(port, baud, timeout=1)
        time.sleep(2)  # ESP32 resets on serial connection
        logger.info("Serial connected on %s", port)
        return ser
    except Exception as e:
        logger.warning("Serial port %s unavailable: %s", port, e)
        logger.warning("Running in demo mode (no bracelet).")
        return None


def send_serial(ser, message: str) -> None:
    """Send a newline-terminated message to ESP32C3."""
    if ser is None:
        return
    try:
        ser.write(f"{message}\n".encode())
    except Exception:
        logger.warning("Serial write failed.")


def close_serial(ser) -> None:
    """Safely close the serial connection."""
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# DRAWING / ANNOTATION HELPERS
# ═══════════════════════════════════════════════════════════════════
def draw_landmarks_on_frame(
    frame: np.ndarray,
    landmarks,
    w: int,
    h: int,
) -> None:
    """Draw key pose landmarks and connections for debugging."""
    # Draw key body points
    key_points = {
        NOSE: (0, 255, 255),          # Yellow
        LEFT_EYE: (0, 255, 0),        # Green
        RIGHT_EYE: (0, 255, 0),
        LEFT_EAR: (255, 165, 0),      # Orange
        RIGHT_EAR: (255, 165, 0),
        LEFT_SHOULDER: (255, 0, 0),   # Blue
        RIGHT_SHOULDER: (255, 0, 0),
        LEFT_HIP: (128, 0, 128),      # Purple
        RIGHT_HIP: (128, 0, 128),
    }

    for idx, color in key_points.items():
        lm = landmarks[idx]
        if lm.visibility > 0.3:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)

    # Draw shoulder-to-ear lines (posture indicator)
    for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_EAR), (RIGHT_SHOULDER, RIGHT_EAR)]:
        s = landmarks[s_idx]
        e = landmarks[e_idx]
        if s.visibility > 0.3 and e.visibility > 0.3:
            pt1 = (int(s.x * w), int(s.y * h))
            pt2 = (int(e.x * w), int(e.y * h))
            cv2.line(frame, pt1, pt2, (0, 200, 200), 2)

    # Draw shoulder line
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    if ls.visibility > 0.3 and rs.visibility > 0.3:
        pt1 = (int(ls.x * w), int(ls.y * h))
        pt2 = (int(rs.x * w), int(rs.y * h))
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


def draw_status_overlay(
    frame: np.ndarray,
    state: DetectionState,
    is_drowsy: bool,
    is_calibrating: bool,
) -> None:
    """Draw detection status text on the video frame."""
    h, w, _ = frame.shape

    if is_calibrating:
        progress = int((state.calibration_frames / CALIBRATION_FRAME_COUNT) * 100)
        cv2.putText(frame, f"CALIBRATING... {progress}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Sit upright and look forward", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return

    color = (0, 0, 255) if is_drowsy else (0, 255, 0)

    if is_drowsy:
        label = f"DROWSY — {state.drowsy_reason.upper().replace('_', ' ')}"
    else:
        label = "Awake"

    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Eye Vis: {state.eye_visibility:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Nose/Shoulder: {state.nose_shoulder_ratio:.3f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Posture Angle: {state.posture_angle:.1f} deg", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Eyes closed: {state.eye_closed_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Nod count: {state.nod_count}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Posture bad: {state.posture_bad_count}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
