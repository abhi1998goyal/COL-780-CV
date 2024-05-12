from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import cv2
import hog
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import joblib
import Landmark as lm
from sklearn.preprocessing import StandardScaler

def extract_Landmark_features_from_folder(folder_path, label):
    lm_features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            lm_features = lm.generate_feature_vectors(lm.get_landmarks(image))
            if(len(lm_features)>0):
               lm_features_list.append(lm_features)
    return lm_features_list


def visualize_svm(clf,X_train, y_train):
    # Apply PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    # Plotting the data points
    plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], color='blue', label='Hands')
    plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], color='red', label='Non-Hands')

    # Getting support vectors and transforming them using PCA
    support_vectors = clf.support_vectors_
    support_vectors_pca = pca.transform(support_vectors)
    plt.scatter(support_vectors_pca[:, 0], support_vectors_pca[:, 1], s=100, facecolors='none', edgecolors='green', label='Support Vectors')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Training Data with Support Vectors')
    plt.legend()
    plt.show()

def run_train_cls_hand_opn_hand():

    cls_hands_features = extract_Landmark_features_from_folder('Final_dataset/closed_hands', label=1)
    print("D1***************************")
    opn_hands_features = extract_Landmark_features_from_folder('Final_dataset/open_hands', label=0)[0:2000]
    print("D2***************************")
    # aug_hands_features = extract_hog_features_from_folder2('Final_dataset/Augmented_Hands_5', label=1)
    # print("D3***************************")

    X = opn_hands_features + cls_hands_features
    #+ aug_hands_features
    y = [1] * len(opn_hands_features) + [0] * len(cls_hands_features) 
    #+ [1]*len(aug_hands_features)

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


    clf = SVC()
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'svm_model_oc2.pkl')

    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)

#visualize_svm(clf, np.array(X_train), np.array(y_train))

def test_hand(clf,img_lm):
    prediction = clf.predict(img_lm.reshape(1, -1))
    decision_score = clf.decision_function(img_lm.reshape(1, -1))
    return prediction, decision_score


#run_train_cls_hand_opn_hand()