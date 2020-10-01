import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import math


class FFT:
    def __init__(self, file,minfreq=2, maxfreq=50,sr=256):
        self.filename=file
        self.min_freq=int(minfreq)
        self.max_freq=int(maxfreq)
        self.srate=int(sr)


    def gettfdata(self):
        df = pd.read_csv(self.filename)
        [m, n] = df.shape
        df = pd.DataFrame(df.values, columns=range(n))
        classes=df[n-1]
        df=df.drop(labels=n-1, axis=1)
        [m, n] = df.shape

        num_frex = 2
        frex = np.linspace(self.min_freq, self.max_freq, num_frex)
        time1 = np.arange(-1.5, 1.5, 1 / self.srate)
        half_wave = round((len(time1) - 1) / 2)
        # FFT parameters
        nKern = len(time1)
        nData = m
        nConv = nKern + nData - 1
        tf= [[[0] * m] * len(frex)] * n
        tf = np.asarray(tf, dtype=float)
        channels = range(n)
        for cyclei in range(0, n):
            dataX = fft(df[channels[cyclei]].to_numpy(), nConv)
            for i in range(0, len(frex)):
                s = 8 / (2 * math.pi * frex[i])
                cmw = np.multiply(np.exp(np.multiply(2 * complex(0, 1) * math.pi * frex[i], time1)),
                                  np.exp(np.divide(-time1 ** 2, (2 * s ** 2))))
                cmwX = fft(cmw, nConv)
                cmwX = np.divide(cmwX, max(cmwX))
                as1 = ifft(np.multiply(cmwX, dataX), nConv)
                as1 = as1[half_wave:len(as1) - half_wave + 1]
                as1 = np.reshape(as1, m)
                mag = np.absolute(as1) ** 2
                tf[cyclei, i, :] = np.absolute(as1) ** 2

        return tf,classes

def main():
    transform=FFT(minfreq=2, maxfreq=50,sr=256)
    #transform.gettfdata()

if __name__=="__main__": main()
