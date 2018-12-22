
##
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

##

def run_svm(X_train, y_train, X_test, y_test):


    # fit the different models
    for kernel in ('linear', 'rbf', 'poly', 'sigmoid'):

        clf = SVC(kernel=kernel, random_state=42, gamma='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('score', kernel, clf.score(X_test, y_test))
        print('accuracy', kernel, accuracy_score(y_test, y_pred))
        print('correct cases', kernel, accuracy_score(y_test, y_pred, ...
               normalize=False))




###

def opt_kernel(X_train, y_train, X_test, y_test, kernel):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print('train score', grid.score(X_train, y_train))
    print('test score', grid.score(X_test, y_test))
    print('best parameters:', grid.best_params_)
    for param, score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
        print(param, score)






##

opt_kernel(X_train, y_train, X_test, y_test, 'linear')
#run_svm(X_train, y_train, X_test, y_test)


##



