#!/usr/bin/env python3
"""
main.py — Entry point for the Driver Drowsiness & Posture Detection System

Uses MediaPipe Pose (33 body landmarks) for full-body posture analysis.

Usage:
    python main.py                          # Run with display
    python main.py --headless               # No display (SSH)
    python main.py --port /dev/ttyUSB0      # Custom serial port
    python main.py --camera 1               # Camera index 1
    python main.py --eye-vis 0.4            # Eye visibility threshold
    python main.py --posture-angle 30       # Posture angle threshold
"""

import argparse
from drowsy_posture_detection import run_detection, DetectionConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Driver Drowsiness & Posture Detection (MediaPipe Pose)"
    )
    p.add_argument("--headless", action="store_true",
                    help="Run without display window")
    p.add_argument("--camera", type=int, default=0,
                    help="Camera device index (default: 0)")
    p.add_argument("--port", type=str, default="/dev/ttyACM0",
                    help="Serial port for ESP32C3 (default: /dev/ttyACM0)")
    p.add_argument("--eye-vis", type=float, default=0.5,
                    help="Eye visibility threshold — lower = more sensitive (default: 0.5)")
    p.add_argument("--eye-frames", type=int, default=15,
                    help="Consecutive closed-eye frames to trigger (default: 15)")
    p.add_argument("--nod-ratio", type=float, default=0.15,
                    help="Nose-shoulder ratio change for nod detection (default: 0.15)")
    p.add_argument("--posture-angle", type=float, default=35.0,
                    help="Ear-shoulder angle threshold in degrees (default: 35.0)")
    p.add_argument("--cooldown", type=float, default=5.0,
                    help="Seconds between repeated alerts (default: 5.0)")
    p.add_argument("--width", type=int, default=640,
                    help="Camera frame width (default: 640)")
    p.add_argument("--height", type=int, default=480,
                    help="Camera frame height (default: 480)")
    return p.parse_args()


def main():
    args = parse_args()

    config = DetectionConfig(
        camera_index=args.camera,
        serial_port=args.port,
        eye_visibility_threshold=args.eye_vis,
        eye_consec_frames=args.eye_frames,
        nod_ratio_threshold=args.nod_ratio,
        posture_angle_threshold=args.posture_angle,
        alert_cooldown_sec=args.cooldown,
        frame_width=args.width,
        frame_height=args.height,
    )

    run_detection(config=config, headless=args.headless)


if __name__ == "__main__":
    main()
