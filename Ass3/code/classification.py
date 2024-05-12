import joblib
import train_svm
import hog
import cv2
import cropped_rectangle
import train_svm_classify as sv
import mediapipe as mp
import numpy as np
import Landmark as lm
from sklearn.preprocessing import StandardScaler



mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, 
                      max_num_hands=2,
                      min_detection_confidence=0.2,
                      min_tracking_confidence=0.2)

clf = joblib.load('svm_model_oc2.pkl')
img = cv2.imread('Final_dataset/my_opn.jpg')

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process the image to detect hand landmarks
results = hands.process(imgRGB)
landmarks=[]

# Check if hands are detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract landmark points
        landmark_points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            landmark_points.append([x, y])

        landmark_points = np.array(landmark_points)
        landmarks.append(landmark_points)



for landmark in landmarks:
    lm_features = lm.generate_feature_vectors(landmark)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(lm_features)
    prediction, decision_score = sv.test_hand(clf, np.array(lm_features))
    print(f'desicion score {decision_score} \n')
    
    color = (0, 255, 255)   #opn #yelow
    if prediction == 0:
        color = (255, 0, 255)  #cls #purple

    for point in landmark:
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1) 
        
    x_min, y_min = np.min(landmark, axis=0)
    x_max, y_max = np.max(landmark, axis=0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

resized_img = cv2.resize(img, (810, 1080))
cv2.imshow('Hand Detection', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()