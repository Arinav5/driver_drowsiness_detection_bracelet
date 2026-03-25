import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lines(edges):
    return cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        50,
        minLineLength=50,
        maxLineGap=100
    )

def average_lane_lines(frame, lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -0.5:
            left_lines.append((slope, intercept))
        elif slope > 0.5:
            right_lines.append((slope, intercept))

    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    def make_line(line_params):
        if len(line_params) == 0:
            return None
        slope, intercept = np.mean(line_params, axis=0)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    return make_line(left_lines), make_line(right_lines)

def draw_line(img, line, color=(0, 255, 0), thickness=5):
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

cap = cv2.VideoCapture(0)

score = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    edges = detect_edges(frame)
    cropped = region_of_interest(edges)
    lines = detect_lines(cropped)

    left_lane, right_lane = average_lane_lines(frame, lines)

    output = frame.copy()
    draw_line(output, left_lane)
    draw_line(output, right_lane)

    frame_center = output.shape[1] // 2

    if left_lane and right_lane:
        left_x = left_lane[0]
        right_x = right_lane[0]
        lane_center = (left_x + right_x) // 2
        offset = frame_center - lane_center

        cv2.line(output, (frame_center, 480), (frame_center, 400), (255, 0, 0), 3)
        cv2.line(output, (lane_center, 480), (lane_center, 400), (0, 0, 255), 3)

        abs_offset = abs(offset)

        if abs_offset > 80:
            score -= 2
            status = "Lane Departure Risk"
        elif abs_offset > 40:
            score -= 1
            status = "Drifting"
        else:
            status = "Good Lane Keeping"

        cv2.putText(output, f"Offset: {offset}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, f"Status: {status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(output, f"Score: {score}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Lane Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Score:", max(score, 0))