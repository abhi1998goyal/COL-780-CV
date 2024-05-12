import sys,os
import classification_hog2 as ch2
import classification_hog as ch

def main():
    #open_path = sys.argv[1]
   # cls_path = sys.argv[2]


    #ch2.class_hog2(open_path,cls_path)
    ch.class_hog('Final_dataset/my_opn.jpg') #uncomment it to just run one picture
    #ch2.class__hog2('Test Dataset\Test Dataset\combined')
    #ch.class_hog('Test Dataset/Test Dataset/closed1/image_15.jpg') 

if __name__ == "__main__":
    main()