from sklearn.metrics import classification_report, roc_auc_score
import os
import cv2
import joblib
import Landmark as lm
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import hog
# Load test images from the "open" and "close" folders



def load_test_images(folder_path):
    test_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            test_images.append(image)
    return test_images

def class_hog2(open_path,cls_path):
    #test_folder_open = 'Final_dataset/Test/open'
    test_folder_open = open_path

    #test_folder_close = 'Final_dataset/Test/close'
    test_folder_close = cls_path

    clf = joblib.load('svm_model_hog_oc1.pkl')


    test_images_open = load_test_images(test_folder_open)
    test_images_close = load_test_images(test_folder_close)

    # Extract features from test images
    test_features_open = [hog.calc_img_hog1(image) for image in test_images_open]
    test_features_close = [hog.calc_img_hog1(image) for image in test_images_close]

    test_features_open = [features for features in test_features_open if np.array(features).size>0]
    test_features_close = [features for features in test_features_close if np.array(features).size>0]


    # Scale test features using the same scaler used during training
    # test_features_open_scaled = scaler.transform(test_features_open)
    # test_features_close_scaled = scaler.transform(test_features_close)

    # Predict labels for test images
    predictions_open=[]
    for tfo in test_features_open:
        #if(tfo.size>0):
        #clf.predict(img_lm.reshape(1, -1))
        predictions_open.append(clf.predict(np.array(tfo).reshape(1,-1)))

    predictions_close=[]
    for tfc in test_features_close:
    # if(tfc.size>0):
        predictions_close.append(clf.predict(np.array(tfc).reshape(1,-1)))

    # Compute evaluation metrics
    y_true_open = np.ones(len(test_features_open))  # Assuming "open" class is labeled as 0
    y_true_close = np.zeros(len(test_features_close))  # Assuming "close" class is labeled as 1

    y_true = np.concatenate([y_true_open, y_true_close])
    predictions = np.concatenate([predictions_open, predictions_close])

    print("Classification Report:")
    print(classification_report(y_true, predictions))

    # Compute ROC AUC score
    y_scores_open = clf.decision_function(test_features_open)
    y_scores_close = clf.decision_function(test_features_close)

    y_scores = np.concatenate([y_scores_open, y_scores_close])
    print("ROC AUC Score:", roc_auc_score(y_true, y_scores))




    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def class__hog2(path):
     clf = joblib.load('svm_model_hog_oc1.pkl')
     test_images_open = load_test_images(path)
     for img in test_images_open:
         tfo=hog.calc_img_hog1(img)
         if(clf.predict(np.array(tfo).reshape(1,-1))==1):
            cv2.imwrite('test/open',img)
         else:
            cv2.imwrite('test/close',img)
     