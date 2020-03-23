# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:34:02 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importKernelPrincipalComponentAnalysisDataset(kernelPrincipalComponentAnalysisDatasetFileName):
    
    kernelPrincipalComponentAnalysisDataset = pd.read_csv(kernelPrincipalComponentAnalysisDatasetFileName)
    X = kernelPrincipalComponentAnalysisDataset.iloc[:, [2,3]].values
    y = kernelPrincipalComponentAnalysisDataset.iloc[:, 4].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveKernelPrincipalComponentAnalysisStandardScaler(kernelPrincipalComponentAnalysisStandardScalar):
    
    #Write KernelPrincipalComponentAnalysisStandardScaler in a picke file
    with open("KernelPrincipalComponentAnalysisStandardScaler.pkl",'wb') as KernelPrincipalComponentAnalysisStandardScaler_Pickle:
        pickle.dump(kernelPrincipalComponentAnalysisStandardScalar, KernelPrincipalComponentAnalysisStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save KernelPrincipalComponentAnalysisModel as a pickle file.
"""
def saveKernelPrincipalComponentAnalysisModel(kernelPrincipalComponentAnalysisModel):
    
    #Write KernelPrincipalComponentAnalysisModel as a picke file
    with open("KernelPrincipalComponentAnalysisModel.pkl",'wb') as KernelPrincipalComponentAnalysisModel_Pickle:
        pickle.dump(kernelPrincipalComponentAnalysisModel, KernelPrincipalComponentAnalysisModel_Pickle, protocol = 2)

"""
read KernelPrincipalComponentAnalysisStandardScalar from pickel file
"""
def readKernelPrincipalComponentAnalysisStandardScaler():
    
    #load KernelPrincipalComponentAnalysisStandardScaler object
    with open("KernelPrincipalComponentAnalysisStandardScaler.pkl","rb") as KernelPrincipalComponentAnalysisStandardScaler:
        kernelPrincipalComponentAnalysisStandardScalar = pickle.load(KernelPrincipalComponentAnalysisStandardScaler)
    
    return kernelPrincipalComponentAnalysisStandardScalar

"""
read KernelPrincipalComponentAnalysisModel from pickle file
"""
def readKernelPrincipalComponentAnalysisModel():
    
    #load KernelPrincipalComponentAnalysisModel model
    with open("KernelPrincipalComponentAnalysisModel.pkl","rb") as KernelPrincipalComponentAnalysisModel:
        kernelPrincipalComponentAnalysisModel = pickle.load(KernelPrincipalComponentAnalysisModel)
    
    return kernelPrincipalComponentAnalysisModel

"""
read X_train from pickle file
"""
def readKernelPrincipalComponentAnalysisXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readKernelPrincipalComponentAnalysisXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readKernelPrincipalComponentAnalysisYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readKernelPrincipalComponentAnalysisYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveKernelPrincipalComponentAnalysisYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readKernelPrincipalComponentAnalysisYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred

def saveTrainingAndTestingDatasetKernelPrincipleComponentAnalysis(X_train_Kernel_PrincipleComponentAnalysis, X_test_Kernel_PrincipleComponentAnalysis):
    
    #Write X_train_Kernel_PrincipleComponentAnalysis in a picke file
    with open("X_train_Kernel_PrincipleComponentAnalysis.pkl",'wb') as X_train_Kernel_PrincipleComponentAnalysis_Pickle:
        pickle.dump(X_train_Kernel_PrincipleComponentAnalysis, X_train_Kernel_PrincipleComponentAnalysis_Pickle, protocol = 2)
    
    #Write X_test_Kernel_PrincipleComponentAnalysis in a picke file
    with open("X_test_Kernel_PrincipleComponentAnalysis.pkl",'wb') as X_test_Kernel_PrincipleComponentAnalysis_Pickle:
        pickle.dump(X_test_Kernel_PrincipleComponentAnalysis, X_test_Kernel_PrincipleComponentAnalysis_Pickle, protocol = 2)

"""
read X_train_PCA from pickle file
"""
def readKernelPrincipalComponentAnalysisXTrainPCA():
    
    #load X_train_PCA
    with open("X_train_Kernel_PrincipleComponentAnalysis.pkl","rb") as X_train_Kernel_PCA_pickle:
        X_train_Kernel_PCA = pickle.load(X_train_Kernel_PCA_pickle)
    
    return X_train_Kernel_PCA

"""
read X_test_PCA from pickle file
"""
def readKernelPrincipalComponentAnalysisXTestPCA():
    
    #load X_test_Kernel_PCA
    with open("X_test_Kernel_PrincipleComponentAnalysis.pkl","rb") as X_test_Kernel_PCA_pickle:
        X_test_Kernel_PCA = pickle.load(X_test_Kernel_PCA_pickle)
    
    return X_test_Kernel_PCA

def saveKernelPCA(kernelPCA):
    
    #Write KernelPCA in a picke file
    with open("KernelPCA.pkl",'wb') as KernelPCA_Pickle:
        pickle.dump(kernelPCA, KernelPCA_Pickle, protocol = 2)
        
"""
read PCA from pickle file
"""
def readKernelPCA():
    
    #load PCA
    with open("KernelPCA.pkl","rb") as KernelPCA_pickle:
        kernelPCA = pickle.load(KernelPCA_pickle)
    
    return kernelPCA
