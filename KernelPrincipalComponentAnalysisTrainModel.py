# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:37:35 2020

@author: Santosh Sah
"""

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from KernelPrincipalComponentAnalysisUtils import (saveKernelPrincipalComponentAnalysisModel, readKernelPrincipalComponentAnalysisXTrain, readKernelPrincipalComponentAnalysisYTrain,
                                             saveTrainingAndTestingDatasetKernelPrincipleComponentAnalysis, readKernelPrincipalComponentAnalysisXTrainPCA,
                                             readKernelPrincipalComponentAnalysisXTest, saveKernelPCA)

"""
Train KernelPrincipalComponentAnalysis model 
"""
def trainKernelPrincipalComponentAnalysisModel():
    
    X_train = readKernelPrincipalComponentAnalysisXTrainPCA()
    y_train = readKernelPrincipalComponentAnalysisYTrain()
        
    kernelPrincipalComponentAnalysis = LogisticRegression(random_state = 1234)
    kernelPrincipalComponentAnalysis.fit(X_train, y_train)
    
    saveKernelPrincipalComponentAnalysisModel(kernelPrincipalComponentAnalysis)

def selectedFeatureComponentsForModel():
    
    X_train = readKernelPrincipalComponentAnalysisXTrain()
    X_test = readKernelPrincipalComponentAnalysisXTest()
    
    kernelPCA = KernelPCA(n_components = 2, kernel = "rbf")
    kernelPCA.fit(X_train)
    
    X_train = kernelPCA.transform(X_train)
    X_test = kernelPCA.transform(X_test)
    
    saveKernelPCA(kernelPCA)
    saveTrainingAndTestingDatasetKernelPrincipleComponentAnalysis(X_train, X_test)

if __name__ == "__main__":
    #selectedFeatureComponentsForModel()
    trainKernelPrincipalComponentAnalysisModel()    
