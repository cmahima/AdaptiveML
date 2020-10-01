from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std



class Model():

    def __init__(self):
        return

    def summarize_results(self,scores):
        m, s = mean(scores), std(scores)
        return m,s


    def KNN(self,datax,datay,folds):
        neigh = KNeighborsClassifier(n_neighbors=6)
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(neigh, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)

    def DecisionTree(self,datax,datay,folds):
        dt = DecisionTreeClassifier()
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(dt, datax, datay, cv=cv,scoring='accuracy')
        return self.summarize_results(scores)

    def LR(self,datax,datay,folds):
        logreg = LogisticRegression(C=10, penalty='l2')
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(logreg, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)


    def SVM(self,datax,datay,folds):
        svclassifier = SVC(kernel='linear')
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(svclassifier, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)

    def MLP(self,datax,datay,folds):
        mlp = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=1000, alpha=0.0001, learning_rate_init=0.01,
                            activation='logistic')
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(mlp, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)

    def RF(self,datax,datay,folds):
        clf = RandomForestClassifier(n_estimators=100)
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(clf, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)


def main():
    model=Model()


if __name__=="__main__": main()