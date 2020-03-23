# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:46:59 2020

@author: Santosh Sah
"""

from sklearn.preprocessing import StandardScaler
from KernelPrincipalComponentAnalysisUtils import (importKernelPrincipalComponentAnalysisDataset, saveTrainingAndTestingDataset, 
                                                   saveKernelPrincipalComponentAnalysisStandardScaler)

def preprocess():
    
    X_train, X_test, y_train, y_test = importKernelPrincipalComponentAnalysisDataset("Kernel_Principal_Component_Analysis_Social_Network_Ads.csv")
    
    kernelPrincipalComponentAnalysisStandardScalar = StandardScaler()
    
    kernelPrincipalComponentAnalysisStandardScalar.fit(X_train)
    saveKernelPrincipalComponentAnalysisStandardScaler(kernelPrincipalComponentAnalysisStandardScalar)
    
    X_train = kernelPrincipalComponentAnalysisStandardScalar.transform(X_train)
    X_test = kernelPrincipalComponentAnalysisStandardScalar.transform(X_test)
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()