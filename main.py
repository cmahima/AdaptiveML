from preprocess import Preprocess
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from fft import FFT


class AdaptiveTraining:

    def __init__(self,root):
        self.filename="No File"
        self.lay = []
        self.sr=256

        self.highfreq=ttk.Entry()
        self.lowfreq=ttk.Entry()
        self.samprate=ttk.Entry()
        self.time_win=ttk.Entry()
        self.flag=False
        self.tfdata=[]
        self.features=[]
        self.method=[]

        self.frame_header = ttk.Frame(root)
        self.frame_header.grid(row=2,sticky ='sw')
        ttk.Label(self.frame_header, text = 'The file uploaded is {}'.format(self.filename)).grid(row = 0, column = 0)


        self.frame_content = ttk.Frame(root)
        self.frame_content.grid(row=4,sticky = 'sw')


        root.title('Adaptive Application')
        root.geometry("1000x800")


        ttk.Button(self.frame_content, text='Upload File', command=self.UploadAction).grid(row=1, column=0, ipadx=5, ipady=5,sticky = 'sw')
        ttk.Button(self.frame_content, text='Preprocess', command=self.preprocess).grid(row=3, column=0, ipadx=5, ipady=5,sticky = 'sw')

    def UploadAction(self,event=None):
        self.filename = filedialog.askopenfilename()
        ttk.Label(self.frame_header, text = 'The file uploaded is {}'.format(self.filename), wraplength = 1000).grid(row = 0, column = 0)



    def preprocess(self):
        subwin=Toplevel()
        subwin.geometry("500x200")

        self.lay.append(subwin)
        subwin.title("Preprocessing")
        ttk.Button(subwin, text='Perform FFT', command=self.FFT).grid(row=0, column=0, ipadx=5, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find variance', command=lambda:self.timewindow("var")).grid(row=1, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find average', command=lambda:self.timewindow("avg")).grid(row=2, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find kurtosis', command=lambda:self.timewindow("kurtosis")).grid(row=3, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find skewness', command=lambda:self.timewindow("skewness")).grid(row=4, column=0, ipadx=4, ipady=5,sticky = 'e')



    def FFT(self):
        fft=Toplevel()
        fft.geometry("500x200")

        self.lay.append(fft)
        fft.title("Fast Fourier Transform Specifications")
        ttk.Label(fft, text = 'Enter lower frequency').grid(row = 0, column = 0, padx = 5, sticky = 'sw')
        ttk.Label(fft, text = 'Enter upper frequency').grid(row = 1, column = 0, padx = 5, sticky = 'sw')
        ttk.Label(fft, text = 'Enter sampling rate').grid(row = 2, column = 0, padx = 5, sticky = 'sw')
        self.lowfreq = ttk.Entry(fft, width=24, font = ('Arial', 10))
        self.highfreq = ttk.Entry(fft, width=24, font = ('Arial', 10))
        self.samprate = ttk.Entry(fft, width=24, font = ('Arial', 10))

        self.lowfreq.grid(row = 0, column = 1, padx = 5)
        self.highfreq.grid(row = 1, column = 1, padx = 5)
        self.samprate.grid(row=2, column=1, padx=5)
        ttk.Button(fft,text="Perform FFT",command=self.getfftparams).grid(row = 3, column = 0, padx = 5, sticky = 'sw')

        btn=Button(fft,text="Save",command=lambda:[fft.destroy(),messagebox.showinfo(title = 'Saved', message = 'The data is saved in tfdata.csv files')])
        btn.grid(row = 3, column = 1, padx = 5, sticky = 'sw')


    def getfftparams(self):
        self.flag=True
        lowerf=(self.lowfreq.get())
        upperf=(self.highfreq.get())
        self.sr=(self.samprate.get())
        data=FFT(self.filename,lowerf,upperf,self.sr)
        self.tfdata=data.gettfdata()


    def timewindow(self,method):
        time=Toplevel()
        time.geometry("500x200")

        ttk.Label(time, text = 'Enter time window in seconds').grid(row = 0, column = 0, padx = 5, sticky = 'sw')
        ttk.Label(time, text = 'secs').grid(row = 0, column = 2, padx = 5, sticky = 'sw')

        self.time_win = ttk.Entry(time, width=24, font=('Arial', 10))
        self.time_win.grid(row = 0, column = 1, padx = 5)


        ttk.Button(time,text="Done",command=lambda: self.gettimeparams(method)).grid(row = 1, column = 0, padx = 5, sticky = 'sw')


        btn=Button(time,text="Save",command=lambda:[time.destroy(),messagebox.showinfo(title = 'Saved', message = 'Time Window Saved')])
        btn.grid(row = 1, column = 1, padx = 5, sticky = 'sw')

    def gettimeparams(self,method ):
        time_window=self.time_win.get()
        entries= int(int(self.sr) * float(time_window))
        k=0
        process=Preprocess()
        if self.flag== True:
            l=len(self.tfdata)
            n=len(self.tfdata[0])
            m=len(self.tfdata[0][0])
            #meth=method.header
            for i in range(l):
                for j in range (n):
                    while k<m/entries:
                        #self.features.append(process.meth(self.tfdata[i][j][k*entries:(k+1)*entries]))
                        self.features.append(getattr(process, method)(self.tfdata[i][j][k*entries:(k+1)*entries]))
                        k+=1

        else:
            df = pd.read_csv(self.filename)
            [m, n] = df.shape
            df = pd.DataFrame(df.values, columns=range(n))
            for i in range(n):
                k=0
                while k<m/entries:
                    self.features.append(getattr(process, method)(df[i].iloc[k*entries:(k+1)*entries]))
                    k+=1
        #messagebox.showinfo(title = 'Final time window', message = 'Time window is {}'.format(time_window))





def main():
    root=Tk()
    app=AdaptiveTraining(root)
    root.mainloop()
'''    filename=input("Enter the path to data--")
    pp=input("For preprocessing enter 1 and For no preprocessing enter 2--")
    print("The path ot file is {}".format(filename))
    df=pd.read_csv(filename)

    if (pp==1):
        return'''


if __name__=="__main__": main()