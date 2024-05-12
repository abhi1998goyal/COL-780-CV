from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import cv2
import hog
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import joblib

def extract_hog_features_from_folder1(folder_path, label):
    hog_features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            hog_features = hog.calc_img_hog1(image)
            hog_features_list.append(hog_features)
    return hog_features_list


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


def extract_hog_features_from_folder2(folder_path, label):
    hog_features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            hog_features = hog.calc_img_hog2(image)
            hog_features_list.append(hog_features)
    return hog_features_list

def run_train_on_hand_non_hand():

    hands_features = extract_hog_features_from_folder1('Final_dataset/Hands', label=1)
    print("D1***************************")
    non_hands_features = extract_hog_features_from_folder2('Final_dataset/NonHands', label=0)
    print("D2***************************")
    aug_hands_features = extract_hog_features_from_folder2('Final_dataset/Augmented_Hands_5', label=1)
    print("D3***************************")

    X = hands_features + non_hands_features + aug_hands_features
    y = [1] * len(hands_features) + [0] * len(non_hands_features) + [1]*len(aug_hands_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    clf = SVC()
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'svm_model_5.pkl')

#accuracy = clf.score(X_test, y_test)
#print("Accuracy:", accuracy)

#visualize_svm(clf, np.array(X_train), np.array(y_train))

def test_hand(clf,img_hog):
    prediction = clf.predict(img_hog.reshape(1, -1))
    decision_score = clf.decision_function(img_hog.reshape(1, -1))
    return prediction, decision_score


#run_train_on_hand_non_hand()