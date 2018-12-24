##
from MyPreprocessing import MyPreprocessing
from scipy.io.arff import loadarff
import numpy as np
from config_loader import load
import argparse
import sys
from os import walk
from scipy.stats import friedmanchisquare
import seaborn as sns
from exercise1_svm import mainExercise1
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from MidpointNormalize import MidpointNormalize



import numpy as np
import pandas as pd
from time import time
import scikit_posthocs as sp
import matplotlib.pyplot as plt

def getProcessedData(path, dataset, filename, raw=False):
    try:
        Xtrain, meta = loadarff(path + filename)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" % (dataset, path))
        sys.exit(1)

    # Preprocessing
    preprocess = MyPreprocessing(raw)
    preprocess.fit(Xtrain)
    df = preprocess.new_df
    labels = preprocess.labels_
    #labels_fac = preprocess.labels_fac
    return df, labels #, labels_fac

def labelFactorization(ytrain_lbl, ytest_lbl):
    labels = np.hstack((ytrain_lbl, ytest_lbl))
    labels_fac = pd.factorize(labels)[0]
    return labels_fac[:ytrain_lbl.size], labels_fac[ytrain_lbl.size:]


def plotHeatmap(scores, gamma_range, C_range, dataset, fig_num):
    plt.figure(10+fig_num, figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.9))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(dataset + ', ' + kernel + ':' + ' Validation accuracy')

##
if __name__ == '__main__':
    ##
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="svm.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    dataset = config.get('svm', 'dataset')
    svm_exercise = config.get('svm', 'svm_exercise')
    kernels = config.get('svm', 'kernels').split('-')
    tuning = config.get('svm', 'tuning')

    if svm_exercise == '1':
        datasets = dataset.split('-')
        correctDataset = True
        for dataset in datasets:
            if dataset not in ['1', '2', '3']:
                correctDataset = False
        if not correctDataset:
            raise ValueError("Dataset for svm_exercise=1 should be 1,2,3 (separating by '-')")
        mainExercise1(kernels, datasets)

    elif svm_exercise == '2':
        path = 'datasetsCBR/' + dataset + '/'  # + dataset + '.fold.00000'
        filenames = [filename for _, _, filename in walk(path)][0]

        df_results = pd.DataFrame()
        accum_acc_lst = []
        accum_time_lst = []
        row_names = []
        C = 0
        gamma = 0
        for kernel in kernels:
            print()
            print('======================================')
            print(kernel)
            print()
            best_params = {}
            acc_lst = []
            time_lst = []
            start = time()
            if tuning == 'gridsearch':
                df_train, ytrain_lbl = getProcessedData(path, dataset, filenames[1])
                df_test, ytest_lbl = getProcessedData(path, dataset, filenames[0])
                ytrain, ytest = labelFactorization(ytrain_lbl, ytest_lbl)
                X = pd.concat([df_train, df_test],
                               axis=0).reset_index(drop=True)
                y= np.hstack((ytrain, ytest))

                n_features = X.shape[1]
                gamma_min = 1.0 / n_features
                gamma_max = 1.0 / (n_features * X.values.std())

                '''
                gamma_range = [gamma_min - gamma_mean, gamma_min,
                               gamma_min + gamma_mean, gamma_max,
                               gamma_max + gamma_mean]
                '''
                gamma_range = np.logspace(-5, 1.2, 10)
                C_range = np.logspace(-1, 5, 10)
                param_grid = dict(C=C_range, gamma=gamma_range)
                clf = GridSearchCV(SVC(kernel=kernel),
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   error_score='raise',
                                   iid=False)
                clf.fit(X, y)

                mean_test_score = clf.cv_results_['mean_test_score']
                scores = mean_test_score.reshape(len(C_range),
                                                 len(gamma_range))
                best_params = clf.best_params_
                ind = clf.cv_results_['rank_test_score'][0]
                print('Best params:', best_params, mean_test_score[ind])
                print()
                plotHeatmap(scores, gamma_range, C_range, dataset, len(row_names))

            for f_ind in range(0,len(filenames), 2):
                test_file = filenames[f_ind]
                train_file = filenames[f_ind+1]
                # raw = False
                df_train, ytrain_lbl = getProcessedData(path, dataset, train_file)
                df_test, ytest_lbl = getProcessedData(path, dataset, test_file)
                ytrain, ytest = labelFactorization(ytrain_lbl, ytest_lbl)

                # fix missing columns because of NaNs and one hot encoding without dummy_na
                if df_train.shape[1] != df_test.shape[1]:
                    missing_cols = set(df_test.columns) - set(df_train.columns)
                    for col in missing_cols:
                        df_train[col] = np.zeros([df_train.shape[0],1])

                    missing_cols = set(df_train.columns) - set(df_test.columns)
                    for col in missing_cols:
                        print(df_train.shape, df_test.shape)
                        df_test[col] = np.zeros([df_test.shape[0],1])

                C = best_params.get('C', 42)
                # gamma from best params or 1/n_features: a
                gamma = best_params.get('gamma', 1.0/df_train.shape[1])

                clf = SVC(kernel=kernel,
                          random_state=42,
                          gamma=gamma,
                          C=C)
                clf.fit(df_train, ytrain)

                acc = clf.score(df_test, ytest)
                acc_lst.append(acc)
                duration = time() - start
                time_lst.append(duration)

            row_name = 'kernel:%s/C:%.2f/gamma:%.5f' % (kernel,
                                                        C,
                                                        gamma)
            row_names.append(row_name)
            accum_acc_lst.append(acc_lst)
            accum_time_lst.append(time_lst)
            duration_time = sum(time_lst)
            accuracy = sum(acc_lst)
            df = pd.DataFrame([[duration_time/10, accuracy/10]],
                              index=[row_name],
                              columns=['Time', 'Accuracy'])
            df_results = pd.concat([df_results, df], axis=0)
            print(df_results)

        df_acc = pd.DataFrame(accum_acc_lst, index=row_names)
        df_time = pd.DataFrame(accum_time_lst, index=row_names)
        print('Accuracy:')
        print(df_acc)
        print()


        print('Time:')
        print(df_time)
        print()

        print('Results:')
        print(df_results.values.tolist())
        print('Accuracy')
        print(accum_acc_lst)
        print('Time')
        print(accum_time_lst)
        print()
        print('Row names')
        print(row_names)

        if len(accum_acc_lst) > 2:
            print("Friedman test on 'accuracy'")
            stat, p = friedmanchisquare(*accum_acc_lst)
            print(stat, p)
            print("Friedman test on 'time'")
            stat, p = friedmanchisquare(*accum_time_lst)
            print(stat, p)

            res = sp.posthoc_nemenyi_friedman(np.array(accum_acc_lst).T)
            res = res.rename(columns={i:row_names[i] for i in res.columns},
                             index={i:row_names[i] for i in res.columns})
            plt.figure(1)
            cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
            heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
            sp.sign_plot(res, **heatmap_args)
            plt.title(dataset + ': '+ 'Nemenyi p-values matrix')

            plt.figure(2)
            for acc, row in zip(accum_acc_lst, row_names):
                # Subset to the airline

                # Draw the density plot
                sns.distplot(acc, hist = False, kde = True,
                            kde_kws = {'linewidth': 3},
                            label = row)

            # Plot formatting
            plt.legend(prop={'size': 16}, title = 'Model')
            plt.title(dataset + ': '+ 'Density Plot with Multiple Models')
            plt.xlabel('Accuracy')
            plt.ylabel('Density')

            plt.show()

