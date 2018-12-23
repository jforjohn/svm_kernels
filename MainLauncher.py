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
    labels_fac = preprocess.labels_fac
    return df, labels, labels_fac

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

        ##
        for filename in filenames:
            #accuracy = 0
            start = time()
            for f_ind in range(0,len(filenames), 2):
                test_file = filenames[f_ind]
                train_file = filenames[f_ind+1]
                # raw = False
                df_train, ytrain, ytrain_fac = getProcessedData(path, dataset, train_file)
                df_test, ytest, _ = getProcessedData(path, dataset, test_file)

                if df_train.shape[1] != df_test.shape[1]:
                    missing_cols = set(df_test.columns) - set(df_train.columns)
                    for col in missing_cols:
                        df_train[col] = np.zeros([df_train.shape[0],1])

                    missing_cols = set(df_train.columns) - set(df_test.columns)
                    for col in missing_cols:
                        print(df_train.shape, df_test.shape)
                        df_test[col] = np.zeros([df_test.shape[0],1])

                clf = MyIBL(n_neighbors=n_neighbor,
                            ibl_algo=ib,
                            voting=voting,
                            distance=distance
                            )
                print(df_train.shape, df_test.shape)
                clf.fit(df_train, ytrain)
                train_obj.append(clf)
                pred = clf.predict(df_test, ytest)

                size_fold = df_test.shape[0]
                acc = clf.classificationTest['correct'] / size_fold
                acc_lst.append(acc)
                duration = time() - start
                time_lst.append(duration)
                missclassification_rate += clf.classificationTest['incorrect'] /size_fold

            row_name = 'k=' + str(n_neighbor) + '/' + distance + '/' + voting
            row_names.append(row_name)
            accum_acc_lst.append(acc_lst)
            accum_time_lst.append(time_lst)
            duration_time = sum(time_lst)
            accuracy = sum(acc_lst)
            df = pd.DataFrame([[duration_time/10, accuracy/10, missclassification_rate/10, cd_len/10]],
                              index=[row_name],
                              columns=['Time', 'Accuracy', 'MisclassRate', 'CDpercentage'])
            df_results = pd.concat([df_results, df], axis=0)
            print(df_results)

        df_acc = pd.DataFrame(accum_acc_lst, index=row_names)
        df_time = pd.DataFrame(accum_time_lst, index=row_names)
        print('Accuracy:')
        print(df_acc)
        print()
        if run == 'all':
            stat, p = friedmanchisquare(*accum_acc_lst)
            print(stat, p)
        print()

        print('Time:')
        print(df_time)
        print()
        if run == 'all':
            stat, p = friedmanchisquare(*accum_time_lst)
            print(stat, p)
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
        '''
        res = sp.posthoc_nemenyi_friedman(np.array(accum_acc_lst).T)
        res = res.rename(columns={i:rows[i] for i in res.columns}, index={i:rows[i] for i in res.columns})
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        sp.sign_plot(res, **heatmap_args)
        plt.title('Nemenyi p-values matrix')
        
        for acc, row in zip(accum_acc_lst,rows):
        # Subset to the airline
        
        # Draw the density plot
        sns.distplot(acc, hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = row)
        
        # Plot formatting
        plt.legend(prop={'size': 16}, title = 'Model')
        plt.title('Density Plot with Multiple Models')
        plt.xlabel('Accuracy')
        plt.ylabel('Density')
        '''

