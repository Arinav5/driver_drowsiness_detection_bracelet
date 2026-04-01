import cv2
import numpy as np
import mediapipe as mp

# MediaPipe FaceMesh setup
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Key eye landmarks
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

# Load the image
image = cv2.imread("test-open-eyes.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = image.shape

# Create FaceMesh object
with mp_facemesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw the chosen eye landmarks
        for idx in all_chosen_idxs:
            lm = landmarks[idx]
            coord = denormalize_coordinates(lm.x, lm.y, img_w, img_h)
            if coord:  # Make sure it's not None
                cv2.circle(image, coord, 3, (0, 255, 0), -1)  # Green dots

# Convert back to BGR for OpenCV display
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Show the image in a window
cv2.imshow("Eye Landmarks", image_bgr)
cv2.waitKey(0)  # Wait until you press a key
cv2.destroyAllWindows()
