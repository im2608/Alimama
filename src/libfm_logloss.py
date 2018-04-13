'''
Created on Apr 3, 2018

@author: Heng.Zhang
'''
from global_variables import *
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss


def logloss():
    libfm_verify_pred = pd.read_csv(r'%s\..\output\libfm.csv' % runningPath, header=None)
    libfm_verify_pred.columns = ['predicted_score']
    verify_label = pd.read_csv(r'%s\..\input\verify_label.txt' % runningPath)

    print("Validation log loss: %.6f" % log_loss(verify_label['is_trade'], libfm_verify_pred['predicted_score']))

    return

def gen_predictio():
    libfm_pred = pd.read_csv(r'%s\..\output\libfm.csv' % runningPath, header=None)
    libfm_pred.columns = ['predicted_score']
    test_instanceid = pd.read_csv(r'%s\..\input\test_instanceid.txt' % runningPath)
    test_instanceid['predicted_score'] = libfm_pred['predicted_score']
    test_instanceid.to_csv(r'%s\..\output\libfm_prediction.csv' % runningPath, index=False,  sep=' ', encoding='utf-8')
    return

if (__name__ == '__main__'):
    logloss()



