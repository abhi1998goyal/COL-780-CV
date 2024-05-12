import sys 
import train_rcnn as frcn
import train_ditr as ditr

def main():
    if(sys.args[0]==1):
       frcn.train()
    if(sys.args[0]==2):
       ditr.train()


if __name__ == "__main__":
    main()