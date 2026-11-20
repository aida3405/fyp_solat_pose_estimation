import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

# ==========================================
# 1. HELPER FUNCTION: CALCULATE JOINT ANGLE
# ==========================================

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# ==========================================
# 1. DOWNLOAD MODEL
# ==========================================

model_path = "pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("Downloading MediaPipe model...")

    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

    urllib.request.urlretrieve(url, model_path)

    print("Download complete!")


# ==========================================
# 2. INITIALISE MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)


# ==========================================
# 3. BODY LANDMARK CONNECTIONS
# ==========================================

connections = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Upper body
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28)   # Right leg
]


# ==========================================
# 4. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

print("Starting pose detection...")
print("Press Q to exit.")


# ==========================================
# 5. MAIN PROCESSING LOOP
# ==========================================

with PoseLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Cannot access webcam!")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert frame into MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Update timestamp
        frame_timestamp_ms += 33

        # Detect body landmarks
        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        # Check whether a person is detected
        if results.pose_landmarks:

            landmarks = results.pose_landmarks[0]

            height, width, _ = frame.shape

            #==========================================
            # Get left arm landmarks
            #==========================================
            left_shoulder = [
                landmarks[11].x * width,
                landmarks[11].y * height
            ]

            left_elbow = [
                landmarks[13].x * width,
                landmarks[13].y * height
            ]

            left_wrist = [
                landmarks[15].x * width,
                landmarks[15].y * height
            ]

            # Calculate elbow angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            #==========================================
            # Get right arm landmarks
            #==========================================
            right_shoulder = [
                landmarks[12].x * width,
                landmarks[12].y * height
            ]

            right_elbow = [
                landmarks[14].x * width,
                landmarks[14].y * height
            ]

            right_wrist = [
                landmarks[16].x * width,
                landmarks[16].y * height
            ]

            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            #==========================================
            # Calculate left hip angle
            #==========================================
            left_hip_shoulder = [
                landmarks[11].x * width,
                landmarks[11].y * height
            ]

            left_hip = [
                landmarks[23].x * width,
                landmarks[23].y * height
            ]

            left_knee = [
                landmarks[25].x * width,
                landmarks[25].y * height
            ]

            left_hip_angle = calculate_angle(left_hip_shoulder, left_hip, left_knee)

            #==========================================
            # Calculate right hip angle
            #==========================================
            right_hip_shoulder = [
                landmarks[12].x * width,
                landmarks[12].y * height
            ]

            right_hip = [
                landmarks[24].x * width,
                landmarks[24].y * height
            ]

            right_knee = [
                landmarks[26].x * width,
                landmarks[26].y * height
            ]

            right_hip_angle = calculate_angle(right_hip_shoulder, right_hip, right_knee)

            #==========================================
            # Display angles
            #=========================================
            cv2.putText(
                frame,
                f"Left Elbow: {int(left_angle)} degrees",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Elbow: {int(right_angle)} degrees",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Left Hip: {int(left_hip_angle)} degrees",
                (20,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Hip: {int(right_hip_angle)} degrees",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Draw landmark connections
            for start, end in connections:

                point1 = landmarks[start]
                point2 = landmarks[end]

                if (
                    point1.visibility > 0.5
                    and point2.visibility > 0.5
                ):

                    x1 = int(point1.x * width)
                    y1 = int(point1.y * height)

                    x2 = int(point2.x * width)
                    y2 = int(point2.y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

            # Draw body landmark points
            for landmark in landmarks:

                if landmark.visibility > 0.5:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 0, 255),
                        -1
                    )

            cv2.putText(
                frame,
                "Person Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No Person Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Display webcam
        cv2.imshow(
            "AI Prayer Guidance System",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ==========================================
# 6. CLEAN UP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Program closed.")
