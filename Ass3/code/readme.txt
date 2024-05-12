To Run the project on test folder
python main.py Final_dataset/Test/open Final_dataset/Test/close 

To Run for just one picture 
Uncomment marked line in main.py file. Comment all other lines main function. Run 
ch.class_hog('path to image') only in main function


To train the SVM
run train_classify_hog.py with cleaned dataset in open_hands and closed_hands folders

To augment the dataset
run augment.py

