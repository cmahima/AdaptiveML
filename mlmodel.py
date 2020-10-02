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
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import svm




class Model():

    def __init__(self):
        return

    def summarize_results(self,scores):
        m, s = mean(scores), std(scores)
        return m,s


    def KNN(self,datax,datay,folds):
        neigh = KNeighborsClassifier()
        params = {
            'n_neighbors': [5,6,7,8],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        grid_kn = GridSearchCV(estimator=neigh,
                               param_grid=params,
                               scoring='accuracy',
                               cv=5,
                               verbose=1,
                               n_jobs=-1)
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        print(grid_kn)
        scores = cross_val_score(grid_kn, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)

    def DecisionTree(self,datax,datay,folds):
        params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
        grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(grid_search_cv, datax, datay, cv=cv,scoring='accuracy')
        return self.summarize_results(scores)

    def LR(self,datax,datay,folds):
        param_grid = {'penalty': ['l1', 'l2'],'C': np.logspace(-4, 4, 20),'solver': ['liblinear']}
        grid_search_cv = GridSearchCV(LogisticRegression(), param_grid, verbose=1, cv=3)
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(grid_search_cv, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)


    def SVM(self,datax,datay,folds):
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC()
        folds=int(folds)
        clf = GridSearchCV(svc, parameters)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(clf, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)

    def MLP(self,datax,datay,folds):
        mlp = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=1000, alpha=0.0001, learning_rate_init=0.01,
                            activation='logistic')
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(mlp, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)

    def RF(self,datax,datay,folds):
        model_params = {
            'n_estimators': [50, 150, 250],
            'min_samples_split': [2, 4, 6]
        }
        rf_model = RandomForestClassifier(random_state=1)
        clf = GridSearchCV(rf_model, model_params, cv=3)
        folds=int(folds)
        cv = ShuffleSplit(n_splits=folds, random_state=0)
        scores = cross_val_score(clf, datax, datay, cv=cv, scoring='accuracy')
        return self.summarize_results(scores)


def main():
    model=Model()


if __name__=="__main__": main()