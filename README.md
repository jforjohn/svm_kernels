# svm_kernels
This is a project where we try and compare different models using SVM kernels.

The folder called svm_kernels comes with a  
- requirements.txt for downloading possible dependencies (pip install -r requirements.txt)  
- svm.cfg configuration file in which you can define the specs of the algorithm you want to run  

When you define what you want to run in the configuration file you just run the MainLauncher.py file.  

Concerning the configuration file at the clustering section:
- dataset: the name of the dataset which is combined with svm_exercise  
    - In case of svm_exercise=1 the options are 1,2,3 or combination of 1-2-3  
    - In case of svm_exercise=2 type the name of the dataset file from datasetCBR directory without the .arff extension, which is in the same directory as this file  
- svm_exercise: 1 or 2, the exercises of this project  
- kernels: the kernel to apply e.g. linear-rbf-sigmoid  
- tuning: the type of tuning when svm_exercise=2}. Available options are gridsearch or none.  