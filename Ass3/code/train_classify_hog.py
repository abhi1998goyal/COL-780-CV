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

def extract_hog_features_from_folder1(folder_path, label):
    hog_features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            hog_features = hog.calc_img_hog1(image)
            if(len(hog_features)>0):
               hog_features_list.append(hog_features)
    return hog_features_list

def extract_hog_features_from_folder2(folder_path, label):
    hog_features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            hog_features = hog.calc_img_hog2(image)
            hog_features_list.append(hog_features)
    return hog_features_list


def run_train_cls_hand_opn_hand():

    cls_hands_features = extract_hog_features_from_folder1('Final_dataset/closed_hands', label=1)
    print("D1***************************")
    opn_hands_features = extract_hog_features_from_folder1('Final_dataset/open_hands', label=0)
    print("D2***************************")
    aug_opn_features = extract_hog_features_from_folder2('Final_dataset/open_hands/color_aug', label=0)
    print("D3***************************")
    aug_cls_features = extract_hog_features_from_folder2('Final_dataset/closed_hands/color_aug', label=1)
    print("D4***************************")
    # aug_hands_features = extract_hog_features_from_folder2('Final_dataset/Augmented_Hands_5', label=1)
    # print("D3***************************")

    X = opn_hands_features + aug_opn_features + cls_hands_features + aug_cls_features
    #+ aug_hands_features
    y = [1] * len(opn_hands_features+aug_opn_features) + [0] * len(cls_hands_features+aug_cls_features) 
    #+ [1]*len(aug_hands_features)

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    clf = SVC()
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'svm_model_hog_oc1.pkl')

    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)

#visualize_svm(clf, np.array(X_train), np.array(y_train))

def test_hand(clf,img_lm):
    prediction = clf.predict(img_lm.reshape(1, -1))
    decision_score = clf.decision_function(img_lm.reshape(1, -1))
    return prediction, decision_score


run_train_cls_hand_opn_hand()