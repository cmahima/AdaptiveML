import numpy as np
from scipy.stats import kurtosis, skew


class Preprocess:

    def init(self):
        return

    def var(self,data):
        return np.var(data)

    def avg(self,data):
        return np.mean(data)

    def kurtosis(self,data):
        return kurtosis(data)

    def skewness(self,data):
        return skew(data)

def main():
    process=Preprocess()

if __name__=="__main__": main()