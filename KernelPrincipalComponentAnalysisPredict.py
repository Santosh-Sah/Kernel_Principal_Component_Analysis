# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:51:12 2020

@author: Santosh Sah
"""

import pandas as pd
from KernelPrincipalComponentAnalysisUtils import readKernelPrincipalComponentAnalysisModel, readKernelPrincipalComponentAnalysisStandardScaler,readKernelPCA

def predict():
    
    kernelPrincipalComponentAnalysis = readKernelPrincipalComponentAnalysisModel()
    kernelPrincipalComponentAnalysisStandardScaler = readKernelPrincipalComponentAnalysisStandardScaler()
    kernelPCA = readKernelPCA()

    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(kernelPCA.transform(kernelPrincipalComponentAnalysisStandardScaler.transform(inputValue)))
    
    predictedValue = kernelPrincipalComponentAnalysis.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()