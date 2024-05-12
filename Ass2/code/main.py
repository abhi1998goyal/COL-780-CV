
import prob1
import prob2
import sys,os


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py 1/2 <img dir> <output path>")
        return

    option = int(sys.argv[1])
    img_dir = sys.argv[2]
    output_path = sys.argv[3]


    if(option==1):
        prob1.prob1(img_dir,output_path)
    elif(option==2):
        prob2.prob2(img_dir,output_path)
    else:
        print("Invalid input")
    

if __name__ == "__main__":
    main()