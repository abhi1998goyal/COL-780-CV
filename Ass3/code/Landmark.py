import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, 
                      max_num_hands=1,
                      min_detection_confidence=0.2,
                      min_tracking_confidence=0.2)

# def extract_hand_landmarks(image_path):
#     # Initialize MediaPipe Hands
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
#                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

#     # Read the input image
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Extract hand landmarks
#     results = hands.process(image_rgb)
#     if results.multi_hand_landmarks:
#         return [hand_landmarks.landmark for hand_landmarks in results.multi_hand_landmarks]
#     else:
#         return None

# import numpy as np

def extract_features(landmarks):
    features = []
    
    # 1. Distance between landmarks
    distances = []
    finger_pairs = [(4, 6), (8, 12), (12, 16), (14, 19)] 

    for pair in finger_pairs:
        dist = np.linalg.norm(landmarks[pair[0]] - landmarks[pair[1]])
        features.append(abs(dist))
    
    # 2. Hand parameters (e.g., width, height, area)
    # hand_bbox = cv2.boundingRect(np.array(landmarks))
    # hand_width = hand_bbox[2]
    # hand_height = hand_bbox[3]
    # hand_area = hand_width * hand_height
    # features.extend([hand_width, hand_height, hand_area])
    
    # 3. Hand aspect ratio
    # aspect_ratio = hand_width / hand_height
    # features.append(aspect_ratio)
    
    # 4. Hand skeleton features (joint angles)
    # joint_angles = []
    # for i in range(len(landmarks)):
    #     for j in range(i+1, len(landmarks)):
    #         # Calculate the vector between two landmarks
    #         vector = np.array([landmarks[j][0] - landmarks[i][0], landmarks[j][1] - landmarks[i][1]])
            
    #         # Calculate the angle of the vector
    #         angle = np.arctan2(vector[1], vector[0]) * 180.0 / np.pi
            
    #         # Normalize angle to be between 0 and 180 degrees
    #         if angle < 0:
    #             angle += 180.0
            
    #         joint_angles.append(angle)
    # features.extend(joint_angles)

    # hull = cv2.convexHull(np.array(landmarks), returnPoints=True)
    # perimeter = cv2.arcLength(hull, True)
    # features.append(perimeter)

    # area = cv2.contourArea(hull)
    # features.append(area)
    
    return features


def generate_feature_vectors(landmarks):
    # Generate feature vectors for each set of hand landmarks
    features=[]
    if(len(landmarks)>0):
   # for landmarks in landmarks_list:
        features = extract_features(landmarks)
        
    return features

#feature_vectors = generate_feature_vectors(landmarks)




def get_landmarks(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process the image to detect hand landmarks
    results = hands.process(imgRGB)
    
    landmark_points = []
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark points
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                landmark_points.append([x, y])
            
            # Convert landmark points to numpy array
    landmark_points = np.array(landmark_points)
    return landmark_points
