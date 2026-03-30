import sys
import time
import logging

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


def run_detection(config: DetectionConfig = None, headless: bool = False) -> None:
    """
    Main entry point — runs the full detection loop using MediaPipe Pose.

    Parameters
    ----------
    config : DetectionConfig, optional
        Tunable parameters. Uses defaults if not provided.
    headless : bool
        If True, skips all cv2.imshow display (for SSH / no-monitor setups).
    """
    if config is None:
        config = DetectionConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 55)
    logger.info("  DRIVER DROWSINESS & POSTURE DETECTION (MediaPipe Pose)")
    logger.info("=" * 55)

    # ── Serial setup ──────────────────────────────────────────────
    ser = open_serial(config.serial_port, config.baud_rate)

    # ── Camera setup ──────────────────────────────────────────────
    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        logger.error("Cannot open camera index %d", config.camera_index)
        sys.exit(1)
    logger.info("Camera opened (index %d)", config.camera_index)

    # ── MediaPipe Pose ────────────────────────────────────────────
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,            # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    logger.info("MediaPipe Pose loaded (model_complexity=1)")

    # ── State ─────────────────────────────────────────────────────
    state = DetectionState()
    logger.info("Calibrating — sit upright and look forward for ~2 seconds...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame — exiting")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            is_drowsy = False
            reason = "no_pose"
            is_calibrating = state.baseline_nose_shoulder_ratio is None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                is_drowsy, reason = evaluate_drowsiness(
                    landmarks, w, h, state, config
                )

                is_calibrating = state.baseline_nose_shoulder_ratio is None
                now = time.time()

                if not is_calibrating:
                    if is_drowsy and (now - state.last_alert_time > config.alert_cooldown_sec):
                        if not state.is_alerting:
                            logger.warning(
                                "DROWSY — %s | eye_vis=%.2f ns_ratio=%.3f "
                                "posture=%.1f° eyes=%d nods=%d posture_bad=%d",
                                reason, state.eye_visibility,
                                state.nose_shoulder_ratio, state.posture_angle,
                                state.eye_closed_count, state.nod_count,
                                state.posture_bad_count,
                            )
                            send_serial(ser, "ALERT")
                            state.is_alerting = True
                            state.last_alert_time = now

                    elif not is_drowsy and state.is_alerting:
                        logger.info("Driver appears awake — clearing alert")
                        send_serial(ser, "OK")
                        state.is_alerting = False
                        state.eye_closed_count = 0
                        state.nod_count = 0
                        state.posture_bad_count = 0

                # Draw pose skeleton + key landmarks
                if not headless:
                    # Full MediaPipe skeleton drawing
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                    )
                    # Custom overlays (posture lines, etc.)
                    draw_landmarks_on_frame(frame, landmarks, w, h)

            if not headless:
                draw_status_overlay(frame, state, is_drowsy, is_calibrating)
                cv2.imshow("Drowsiness & Posture Detector (Pose)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        logger.info("Shutting down...")
        send_serial(ser, "OK")
        close_serial(ser)
        pose.close()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("Done.")
