import cv2
import joblib
import numpy as np
import mediapipe as mp
import Landmark as lm 
# Load the trained SVM model svm_model_hog_oc1.pkl
#svm_model = joblib.load('svm_model_oc2.pkl')

svm_model = joblib.load('svm_model_hog_oc1.pkl')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, 
                       max_num_hands=1,
                       min_detection_confidence=0.2,
                       min_tracking_confidence=0.2)

# Function to preprocess the hand region and classify the gesture
def detect_hand_gesture(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame_rgb)

    hand_landmarks = lm.get_landmarks(frame)

   # if results.multi_hand_landmarks:
    #    for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand region using landmarks
            # Example: Get bounding box of the hand
            #bbox = get_hand_bbox(hand_landmarks, frame.shape)

            # Crop the hand region
            #hand_region = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            # Preprocess the hand region (resize, normalize, etc.)
            # Example: Apply any preprocessing steps

            # Extract features from the preprocessed hand region

    features = lm.generate_feature_vectors(hand_landmarks)
    prediction=1

    if(len(features))>0:

            # Classify the gesture using the SVM model
        prediction = svm_model.predict(np.array(features).reshape(1,-1))

            # Return True if "close" gesture is detected, False otherwise
    return prediction == 0  # Assuming "close" gesture is labeled as class 0

    #return False  # No hand detected

# Function to extract features from the hand region
#def extract_features(hand_region):
    # Implement feature extraction logic here
    # Example: Compute HOG features, landmarks, or any other relevant features
    # Return the extracted features as a numpy array
    #img_lm.reshape(1, -1)

# Function to get bounding box of the hand
def get_hand_bbox(hand_landmarks, frame_shape):
    # Example: Get bounding box based on hand landmarks
    # You may need to adjust this based on your specific application
    bbox = [int(hand_landmarks[0].x * frame_shape[1]), 
            int(hand_landmarks[0].y * frame_shape[0]),
            200, 200]  # Example: Bounding box width and height
    return bbox

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform hand gesture detection and classification
    if detect_hand_gesture(frame):
        # Close the window if "close" gesture is detected
        cv2.destroyAllWindows()
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
