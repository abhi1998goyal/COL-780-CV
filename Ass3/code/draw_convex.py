import os
import cv2
import mediapipe as mp
import numpy as np

print("MediaPipe version:", mp.__version__)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, 
                      max_num_hands=2,
                      min_detection_confidence=0.2,
                      min_tracking_confidence=0.2)

# Read the image
img = cv2.imread('Final_dataset/closed3/train/1617.jpg')
#img = cv2.flip(img, 1)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process the image to detect hand landmarks
results = hands.process(imgRGB)

# Check if hands are detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract landmark points
        landmark_points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            landmark_points.append([x, y])

        # Convert landmark points to numpy array
        landmark_points = np.array(landmark_points)

        # Draw green circles at each landmark position
        for point in landmark_points:
            cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)

        # Compute and draw the convex hull
        hull = cv2.convexHull(landmark_points, returnPoints=True)
        cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)

# Display the image with highlighted landmarks and convex hull
cv2.imshow('Hand Landmarks with Convex Hull', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
