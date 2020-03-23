# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:55:42 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from KernelPrincipalComponentAnalysisUtils import (readKernelPrincipalComponentAnalysisYTest, readKernelPrincipalComponentAnalysisYPred)

"""

calculating KernelPrincipalComponentAnalysis confussion matrix

"""
def testKernelPrincipalComponentAnalysisConfussionMatrix():
    
    y_test = readKernelPrincipalComponentAnalysisYTest()
    y_pred = readKernelPrincipalComponentAnalysisYPred()
    
    kernelPrincipalComponentAnalysisConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(kernelPrincipalComponentAnalysisConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[54  4]
    [ 4 18]]
    
    """
"""
calculating accuracy score

"""

def testKernelPrincipalComponentAnalysisAccuracy():
    
    y_test = readKernelPrincipalComponentAnalysisYTest()
    y_pred = readKernelPrincipalComponentAnalysisYPred()
    
    kernelPrincipalComponentAnalysisConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(kernelPrincipalComponentAnalysisConfussionAccuracy) #.90%

"""
calculating classification report

"""

def testKernelPrincipalComponentAnalysisClassificationReport():
    
    y_test = readKernelPrincipalComponentAnalysisYTest()
    y_pred = readKernelPrincipalComponentAnalysisYPred()
    
    kernelPrincipalComponentAnalysisConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(kernelPrincipalComponentAnalysisConfussionClassificationReport)
    
    """
               precision    recall  f1-score   support

          0       0.93      0.93      0.93        58
          1       0.82      0.82      0.82        22

avg / total       0.90      0.90      0.90        80

    """
    
if __name__ == "__main__":
    #testKernelPrincipalComponentAnalysisConfussionMatrix()
    #testKernelPrincipalComponentAnalysisAccuracy()
    testKernelPrincipalComponentAnalysisClassificationReport()