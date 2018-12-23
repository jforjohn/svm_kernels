#############################################################
#############################################################
#############################################################


import numpy as np
#import cvxopt
#import cvxopt.solvers
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools




if __name__ == "__main__":
    import pylab as pl

    rng = np.random.RandomState(0)


    def plot_confusion_matrix(cm, classes, kernel_name, fig_num,
                              normalize=False,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        title = kernel_name + ' Confusion matrix'
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.figure(fig_num + 10)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()


    def generate_data_set1():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set2():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def generate_data_set3():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def run_svm_dataset(dataset_index):
        random.seed(42)
        X1, y1, X2, y2 = globals().get('generate_data_set' + str(dataset_index))()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        # stack all the data to plot
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))


        # fit the different models
        for fig_num, kernel in enumerate(('linear', 'rbf', 'poly', 'sigmoid')):
            clf = SVC(kernel=kernel, random_state=42, gamma='auto')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print('===============')
            print(kernel)
            print('score:', clf.score(X_test, y_test))
            print('accuracy:', accuracy_score(y_test, y_pred))
            print('correct cases:', accuracy_score(y_test, y_pred, normalize=False))
            plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=set(y_test),
                                  kernel_name=kernel, fig_num=fig_num)
            print()


            plt.figure(fig_num)
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.coolwarm,
                        edgecolor='k', s=20)

            # Circle out the test data
            plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                        zorder=10, edgecolor='w')

            plt.axis('tight')
            x_min = X[:, 0].min() - 1
            x_max = X[:, 0].max() + 1
            y_min = X[:, 1].min() - 1
            y_max = X[:, 1].max() + 1

            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(XX.shape)
            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.coolwarm)
            plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                        linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

            plt.title(kernel + ' Hyperplane')
        plt.show()


        return


        # Write here your SVM code and choose a linear kernel
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        #

    '''
    def run_svm_dataset2():
        X1, y1, X2, y2 = generate_data_set2()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)


        # stack all the data to plot
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))


        # fit the different models
        for fig_num, kernel in enumerate(('linear', 'rbf', 'poly', 'sigmoid')):
            clf = SVC(kernel=kernel, random_state=42, gamma='auto')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print('score', kernel, clf.score(X_test, y_test))
            print('accuracy', kernel, accuracy_score(y_test, y_pred))
            print('correct cases', kernel, accuracy_score(y_test, y_pred, normalize=False))

            plt.figure(fig_num)
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.coolwarm,
                        edgecolor='k', s=20)

            # Circle out the test data
            plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                        zorder=10, edgecolor='k')

            plt.axis('tight')
            x_min = X[:, 0].min() - 1
            x_max = X[:, 0].max() + 1
            y_min = X[:, 1].min() - 1
            y_max = X[:, 1].max() + 1

            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(XX.shape)
            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.coolwarm)
            plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                        linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

            plt.title(kernel)
        plt.show()

        return

        #
        # Write here your SVM code and choose a linear kernel with the best C parameter
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        #



    def run_svm_dataset3():
        X1, y1, X2, y2 = generate_data_set3()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)



        # stack all the data to plot
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))



        # fit the different models
        for fig_num, kernel in enumerate(('linear', 'rbf', 'poly', 'sigmoid')):
            clf = SVC(kernel=kernel, random_state=42, gamma='auto')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print('score', kernel, clf.score(X_test, y_test))
            print('accuracy', kernel, accuracy_score(y_test, y_pred))
            print('correct cases', kernel, accuracy_score(y_test, y_pred, normalize=False))

            plt.figure(fig_num)
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.coolwarm,
                        edgecolor='k', s=20)

            # Circle out the test data
            plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                        zorder=10, edgecolor='k')

            plt.axis('tight')
            x_min = X[:, 0].min()-1
            x_max = X[:, 0].max()+1
            y_min = X[:, 1].min()-1
            y_max = X[:, 1].max()+1

            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(XX.shape)
            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.coolwarm)
            plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                        linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

            plt.title(kernel)
        plt.show()

        return

        #### 
        # Write here your SVM code and use a gaussian kernel 
        # plot the graph with the support_vectors_
        # print on the console the number of correct predictions and the total of predictions
        ####
    '''


#############################################################
#############################################################
#############################################################

# EXECUTE SVM with THIS DATASETS
    # set seed for reproducibility
    #random.seed(40)
    rng = np.random.RandomState(0)


    run_svm_dataset(1)   # data distribution 1
    #run_svm_dataset(2)   # data distribution 2
    #run_svm_dataset(3)   # data distribution 3

#############################################################
#############################################################
#############################################################
