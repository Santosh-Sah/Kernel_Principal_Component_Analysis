# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:42:18 2020

@author: Santosh Sah
"""

from KernelPrincipalComponentAnalysisUtils import (readKernelPrincipalComponentAnalysisXTestPCA, readKernelPrincipalComponentAnalysisModel,
                                     saveKernelPrincipalComponentAnalysisYPred)

"""
test the model on testing dataset
"""
def testLogisticRegressionModel():
    
    X_test = readKernelPrincipalComponentAnalysisXTestPCA()
    
    kernelPrincipalComponentAnalysisModel = readKernelPrincipalComponentAnalysisModel()
    
    y_pred = kernelPrincipalComponentAnalysisModel.predict(X_test)
    saveKernelPrincipalComponentAnalysisYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testLogisticRegressionModel()