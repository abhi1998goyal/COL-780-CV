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

def class_hog(image_path):

    clf = joblib.load('svm_model_hog_oc1.pkl')
    #img = cv2.imread('Final_dataset/my_opn2.jpg')
    img = cv2.imread(image_path)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #hog_features = hog.calc_img_hog2(img)

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
        x, y, w, h = cv2.boundingRect(landmark) 
        crop = img[y:y+h, x:x+w]
    # lm_features = lm.generate_feature_vectors(landmark)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(lm_features)
        prediction, decision_score = sv.test_hand(clf, np.array(hog.calc_img_hog2(crop)))
        print(f'desicion score {decision_score} \n')
        
        color = (0, 255, 255)   #opn #yelow
        if prediction == 0:
            color = (255, 0, 255)  #cls #purple

        for point in landmark:
                cv2.circle(img, tuple(point), 5, (255, 0, 0), -1) 
            
        x_min, y_min = np.min(landmark, axis=0)
        x_max, y_max = np.max(landmark, axis=0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 5)

    resized_img = cv2.resize(img, (810, 810))
    cv2.imshow('Hand Detection', resized_img)
    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()