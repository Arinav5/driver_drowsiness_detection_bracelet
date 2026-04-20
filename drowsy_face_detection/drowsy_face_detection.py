import cv2
import mediapipe as mp
import numpy as np
import time


# ---------------------------------
# Helpers
# ---------------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_face_point(landmarks, idx, img_w, img_h):
    lm = landmarks[idx]
    return (int(lm.x * img_w), int(lm.y * img_h))


def get_eye_points(landmarks, eye_indices, img_w, img_h):
    return [get_face_point(landmarks, idx, img_w, img_h) for idx in eye_indices]


# ---------------------------------
# MediaPipe setup
# ---------------------------------
mp_face_mesh = mp.solutions.face_mesh

# Face landmarks for eyes
left_eye_indices = [362, 385, 387, 263, 373, 380]
right_eye_indices = [33, 160, 158, 133, 153, 144]

# Thresholds
CLOSED_THRESHOLD = 0.17
CLOSED_TIME_THRESHOLD = 3.0  # 3 seconds


# ---------------------------------
# Webcam
# ---------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

closed_start_time = None

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)

        eye_status = "No face detected"

        # ---------------------------------
        # EYE DETECTION ONLY
        # ---------------------------------
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark

            left_eye = get_eye_points(face_landmarks, left_eye_indices, img_w, img_h)
            right_eye = get_eye_points(face_landmarks, right_eye_indices, img_w, img_h)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # draw eye points
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # check if both eyes are closed
            if left_ear < CLOSED_THRESHOLD and right_ear < CLOSED_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()

                closed_duration = time.time() - closed_start_time
                eye_status = f"Eyes Closed: {closed_duration:.1f}s"

                # big alert after 3 seconds
                if closed_duration >= CLOSED_TIME_THRESHOLD:
                    cv2.rectangle(frame, (40, 100), (img_w - 40, 250), (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "WAKE UP!",
                        (80, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.2,
                        (255, 255, 255),
                        5
                    )
                    cv2.putText(
                        frame,
                        "EYES CLOSED",
                        (80, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        4
                    )
            else:
                closed_start_time = None
                eye_status = "Eyes Open"

            cv2.putText(
                frame,
                f"EAR: {avg_ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        else:
            closed_start_time = None

        # ---------------------------------
        # Status text
        # ---------------------------------
        cv2.putText(
            frame,
            eye_status,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow("Drowsy Eye Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()