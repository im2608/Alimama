'''
Created on Mar 8, 2018

@author: Heng.Zhang
'''

from global_variables import *
import numpy as np
import pandas as pd
import scipy
import pywFM
from pyfm import pylibfm

def main():
    print(getCurrentTime(), "running...")
    dok_train = np.load(r"%s\..\input\sparse_train.npy" % runningPath)[()]# train 中包括 '2018-09-17'-- '2018-09-23' 的数据
    dok_verify = np.load(r"%s\..\input\sparse_verify.npy" % runningPath)[()] # verify 中只包含了 '2018-09-24' 的数据
    dok_test = np.load(r"%s\..\input\sparse_test.npy" % runningPath)[()]
    
    train_label = pd.read_csv(r'%s\..\input\train_label.txt' % runningPath)
    verify_label = pd.read_csv(r'%s\..\input\verify_label.txt' % runningPath)
    
    fm = pylibfm.FM(num_factors=50, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")
    
    fm.fit(dok_train, train_label['is_trade'])
    
    Y_prediced = fm.predict(dok_verify)
    
    pyfm_logloss = -np.sum(verify_label * np.log(Y_prediced) + 
                          (1 - verify_label) * np.log(1 - Y_prediced))/ Y_prediced.shape[0] 
    print(getCurrentTime(), "lgb logloss %.6f" %(pyfm_logloss))

    
    return 

if (__name__ == '__main__'):
    main()

    