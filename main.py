from preprocess import Preprocess

import pandas as pd


'''class AdaptiveTraining:

    def init(self):
        #define class variables

    def preprocess(self,df):'''



def main():
    filename=input("Enter the path to data--")
    pp=input("For preprocessing enter 1 and For no preprocessing enter 2--")
    print("The path ot file is {}".format(filename))
    df=pd.read_csv(filename)

    if (pp==1):







if __name__=="__main__": main()