from preprocess import Preprocess
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from fft import FFT
from mlmodel import Model


class AdaptiveTraining:

    def __init__(self,root):
        self.filename="No File"
        self.lay = []
        self.sr=256

        self.highfreq=ttk.Entry()
        self.lowfreq=ttk.Entry()
        self.samprate=ttk.Entry()
        self.time_win=ttk.Entry()
        self.cv=ttk.Entry()
        self.fftflag=False
        self.preflag=False
        self.tfdata=[]
        self.features=[]
        self.method=[]
        self.labesl=[]
        self.classes_new=[]

        self.final_df= pd.DataFrame()

        self.frame_header = ttk.Frame(root)
        self.frame_header.grid(row=2,sticky ='sw')
        ttk.Label(self.frame_header, text = 'The file uploaded is {}'.format(self.filename)).grid(row = 0, column = 0)


        self.frame_content = ttk.Frame(root)
        self.frame_content.grid(row=4,sticky = 'sw')


        root.title('Adaptive Application')
        root.geometry("500x400")


        ttk.Button(self.frame_content, text='Upload File', command=self.UploadAction).grid(row=1, column=0, ipadx=5, ipady=5,sticky = 'sw')
        ttk.Button(self.frame_content, text='Preprocess', command=self.preprocess).grid(row=3, column=0, ipadx=5, ipady=5,sticky = 'sw')
        ttk.Button(self.frame_content, text='Apply ML models', command=self.MLModel).grid(row=6, column=0, ipadx=5, ipady=5,sticky = 'sw')

    def UploadAction(self,event=None):
        self.filename = filedialog.askopenfilename()
        ttk.Label(self.frame_header, text = 'The file uploaded is {}'.format(self.filename), wraplength = 1000).grid(row = 0, column = 0)



    def preprocess(self):
        self.preflag=True
        subwin=Toplevel()
        subwin.geometry("400x200")

        self.lay.append(subwin)
        subwin.title("Preprocessing")
        ttk.Button(subwin, text='Perform FFT', command=self.FFT).grid(row=0, column=0, ipadx=5, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find variance', command=lambda:self.timewindow("var")).grid(row=1, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find average', command=lambda:self.timewindow("avg")).grid(row=2, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find kurtosis', command=lambda:self.timewindow("kurtosis")).grid(row=3, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Find skewness', command=lambda:self.timewindow("skewness")).grid(row=4, column=0, ipadx=4, ipady=5,sticky = 'e')
        ttk.Button(subwin, text='Done', command=lambda:subwin.destroy()).grid(row=5, column=2, ipadx=4, ipady=5,sticky = 'e')



    def FFT(self):
        fft=Toplevel()
        fft.geometry("400x200")

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
        self.fftflag=True
        lowerf=(self.lowfreq.get())
        upperf=(self.highfreq.get())
        self.sr=(self.samprate.get())
        data=FFT(self.filename,lowerf,upperf,self.sr)
        self.tfdata,self.classes=data.gettfdata()


    def timewindow(self,method):
        time=Toplevel()
        time.geometry("400x200")

        ttk.Label(time, text = 'Enter time window in seconds').grid(row = 0, column = 0, padx = 5, sticky = 'sw')
        ttk.Label(time, text = 'secs').grid(row = 0, column = 2, padx = 5, sticky = 'sw')

        self.time_win = ttk.Entry(time, width=24, font=('Arial', 10))
        self.time_win.grid(row = 0, column = 1, padx = 5)


        ttk.Button(time,text="Done",command=lambda: self.gettimeparams(method)).grid(row = 1, column = 0, padx = 5, sticky = 'sw')


        btn=Button(time,text="Save",command=lambda:[time.destroy()])
        btn.grid(row = 1, column = 1, padx = 5, sticky = 'sw')

    def gettimeparams(self,method ):
        time_window=self.time_win.get()
        entries= int(int(self.sr) * float(time_window))
        k=0
        process=Preprocess()
        if self.fftflag== True:
            l=len(self.tfdata)
            n=len(self.tfdata[0])
            m=len(self.tfdata[0][0])
            for i in range(l):
                for j in range (n):
                    k=0
                    self.classes_new = []
                    while k<m/entries:
                        if (all(x==self.classes[k*entries] for x in self.classes[k*entries:(k+1)*entries])):
                            self.features.append(getattr(process, method)(self.tfdata[i][j][k*entries:(k+1)*entries]))
                            self.classes_new.append(self.classes[k*entries])
                            k+=1
                        elif (all(x == self.classes[k * entries] for x in
                                  self.classes[k * entries:(k + 1) * entries]) == False):
                            k += 1

                    self.final_df[method+"_fft_{}_{}".format(i,j)]=self.features
                    self.features = []


        else:
            df = pd.read_csv(self.filename)
            [m, n] = df.shape
            df = pd.DataFrame(df.values, columns=range(n))
            self.classes= df[n-1]
            df=df.drop(labels=n-1, axis=1)
            [m, n] = df.shape
            for i in range(n):
                k=0
                self.classes_new=[]
                while k<m/entries:
                    if (all(x == self.classes[k * entries] for x in self.classes[k * entries:(k + 1) * entries])):
                        self.features.append(getattr(process, method)(df[i].iloc[k*entries:(k+1)*entries]))
                        self.classes_new.append(self.classes[k * entries])
                        k+=1
                    elif(all(x == self.classes[k * entries] for x in self.classes[k * entries:(k + 1) * entries])==False):k+=1
                self.final_df[method+"_{}".format(i)]=self.features
                self.features=[]
        self.classes = self.classes_new
        print(self.final_df)
        print((self.classes))
        #messagebox.showinfo(title = 'Final time window', message = 'Time window is {}'.format(time_window))

    def MLModel(self):
        model=Toplevel()
        model.geometry("400x200")

        model.title("Machine Learning Model")
        ttk.Label(model, text = 'Enter number of folds for cross validation').grid(row = 0, column = 0, padx = 5, sticky = 'sw')
        self.cv= ttk.Entry(model, width=14, font=('Arial', 10))
        self.cv.grid(row = 0, column = 1, padx = 5)
        ttk.Button(model, text="K Nearest Neighbors", command=lambda:[self.apply_model("KNN")]).grid(row = 1, column = 0, padx = 5, sticky = 'sw')
        ttk.Button(model, text="Decision Tree", command=lambda:[self.apply_model("DecisionTree")]).grid(row = 2, column = 0, padx = 5, sticky = 'sw')
        ttk.Button(model, text="Logistic Regression", command=lambda:[self.apply_model("LR")]).grid(row = 3, column = 0, padx = 5, sticky = 'sw')
        ttk.Button(model, text="Random Forest", command=lambda:[self.apply_model("RF")]).grid(row = 4, column = 0, padx = 5, sticky = 'sw')
        ttk.Button(model, text="Support Vector Machine", command=lambda:[self.apply_model("SVM")]).grid(row = 5, column = 0, padx = 5, sticky = 'sw')


    def apply_model(self,model):
        model_obj=Model()
        if (self.preflag == True):
            df=self.final_df
            classes=self.classes
        elif(self.preflag == False):
            df = pd.read_csv(self.filename)
            [m, n] = df.shape
            df = pd.DataFrame(df.values, columns=range(n))
            classes= df[n-1]
            df=df.drop(labels=n-1, axis=1)
        cross_val=self.cv.get()
        m,s=getattr(model_obj, model)(df,classes,cross_val)
        print('Accuracy of {}:'.format(model))
        print('%.3f%% (+/-%.3f)' % (m, s))


def main():
    root=Tk()
    app=AdaptiveTraining(root)
    root.mainloop()



if __name__=="__main__": main()