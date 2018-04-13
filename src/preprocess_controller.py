'''
Created on Mar 1, 2018

@author: Heng.Zhang
'''

import subprocess  
import time

from global_variables import *
from features import *
from shop_features import *
from item_features import *
from onehot_features import *
from commonfunc import *
import json
import scipy
from pygame.examples.prevent_display_stretching import running
import os

runningSubProcesses = {}


def submiteOneSubProcess(dffilename):
    cmdLine = "python preprocess_cocurrent.py f=%s" % (dffilename)
    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(dffilename, time.time())] = sub
    print("running cmd line: %s" % cmdLine)
    time.sleep(1)
    return


def waitSubprocesses():
    for start_end_date_str in runningSubProcesses:
        sub = runningSubProcesses[start_end_date_str]
        ret = subprocess.Popen.poll(sub)
        if ret == 0:
            runningSubProcesses.pop(start_end_date_str)
            return start_end_date_str
        elif ret is None:
            time.sleep(1) # running
        else:
            runningSubProcesses.pop(start_end_date_str)
            return start_end_date_str
    return (0, 0)

def libfm_and_onehot_cocurrent():
    print(getCurrentTime(), " running...")
    sparse_mat_col_idx_dict = {'idx' : 0}
    test_date = ['2018-09-24', '2018-09-25']
#     test_date = ['2018-09-24']
    train_date = ['2018-09-17', '2018-09-18', '2018-09-19', '2018-09-20', '2018-09-21', '2018-09-22', '2018-09-23', '2018-09-24']  
#     train_date = ['2018-09-17', '2018-09-18']

    dftest = pd.read_csv(r'%s\..\input\test.txt' % runningPath)
    df = pd.read_csv(r'%s\..\input\train.txt' % runningPath)    
  
    X_test = []
    for each_date in test_date:
        print(getCurrentTime(), 'extracting test features on %s' % each_date)
        each_df_of_date, float_features = extract_features_libfm(dftest[dftest['date'] == each_date], sparse_mat_col_idx_dict)
        X_test.append(each_df_of_date)
    dftest = pd.concat(X_test, axis=0, ignore_index=True)
    dftest[['instance_id']].to_csv(r'%s\..\input\test_label.txt' % (runningPath), index=False)
   
    # index 记录了原始instance id 顺序，按照index排序则恢复到原始顺序
    dftest = dftest.sort_values('index',  axis=0, ascending=True)
    del dftest['index']
   
    dftest.to_csv(r'%s\..\input\cocurrent\test_features.csv' % (runningPath), index=False)
   
    print(getCurrentTime(), "dftest after all extracting feature shape",  dftest.shape)
    
    train_instance_id = []
  
    for each_date in train_date:
        print(getCurrentTime(), 'extracting train features on %s' % each_date)
        train_date_file = r'%s\..\input\cocurrent\train_features_%s.csv'  % (runningPath, each_date)
        tmp_df = df[df['date'] == each_date]
        train_instance_id.append(tmp_df[['instance_id', 'is_trade']])
        each_df_of_date, float_features = extract_features_libfm(tmp_df, sparse_mat_col_idx_dict)
        each_df_of_date.to_csv(train_date_file, index=False)
        
    train_only_df = pd.concat(train_instance_id[:-1], axis=0, ignore_index=True)
    train_only_df.to_csv(r'%s\..\input\train_label.csv' % (runningPath), index=False)
    
    train_instance_id[:-1].to_csv(r'%s\..\input\verify_label.csv' % (runningPath), index=False)
    
    train_total_df = pd.concat(train_instance_id, axis=0, ignore_index=True)
    train_total_df.to_csv(r'%s\..\input\train_total_label.csv' % (runningPath), index=False)
    
    del train_total_df
    del train_only_df
   
    print(getCurrentTime(), "df after all extracting feature shape", df.shape)
    for each in float_features:
        if (each not in float_features):
            sparse_mat_col_idx_dict[each] = sparse_mat_col_idx_dict['idx']
            sparse_mat_col_idx_dict['idx'] += 1
           
    with open(r'%s\..\input\cocurrent\float_features.txt' % runningPath, 'w') as float_features_file:
        float_features_file.write(",".join(list(float_features)))
   
    for each_labled in LAB_ENCODE.keys():
        create_onehot_col_idx(df, each_labled, sparse_mat_col_idx_dict)
        create_onehot_col_idx(dftest, each_labled, sparse_mat_col_idx_dict)
   
    # 经过以上步骤，确定了稀疏矩阵有多少列
    print(getCurrentTime(), "sparse_mat_col_idx_dict is %d" % (sparse_mat_col_idx_dict['idx']))
       
    with open(r'%s\..\input\cocurrent\col_idx_dict.txt' % runningPath, 'w') as col_idx_file:
        sparse_mat_col_idx_dict.pop('idx')
        json.dump(sparse_mat_col_idx_dict, col_idx_file)
# 
    for each in train_date:
        submiteOneSubProcess('train_features_%s.csv' % each)
  
    submiteOneSubProcess('test_features.csv')
 
    while True:
        start_end_date_str = waitSubprocesses()
        if ((start_end_date_str[0] != 0 and start_end_date_str[1] != 0)):
                    print("after waitSubprocesses, subprocess [%s] finished, took %d seconds, runningSubProcesses len is %d" % 
                          (start_end_date_str[0], time.time() - start_end_date_str[1], len(runningSubProcesses)))
        if (len(runningSubProcesses) == 0):
            break

    sparse_train = None
    print(getCurrentTime(), "combining sparse files..")
    for each_date in train_date[:-1]:
        sparse_filename = "train_features_%s.csv.sparse.npy" % (each_date)
        print(getCurrentTime(), "loading %s" % (sparse_filename))
        dok_train = np.load(r"%s\..\input\cocurrent\%s" % (runningPath, sparse_filename))[()]
        if (sparse_train is None):
            sparse_train = dok_train
        else:
            sparse_train = scipy.sparse.vstack([sparse_train, dok_train])

    print(getCurrentTime(), "combining done, saving files..")
    np.save(r"%s\..\input\train.sparse" % (runningPath), sparse_train)

#     sparse_filename = "train_features_%s.csv.sparse.npy" % (train_date[-1])
#     dok_train = np.load(r"%s\..\input\cocurrent\%s" % (runningPath, sparse_filename))[()]
#     sparse_train = scipy.sparse.vstack([sparse_train, dok_train])
#     np.save(r"%s\..\input\train_total.sparse" % (runningPath), sparse_train)

#     combin_train_libfm_cmd = "copy"
#     for each_date in train_date[:-1]:
#         if (len(combin_train_libfm_cmd) > 4):
#             combin_train_libfm_cmd += r'+%s\..\input\cocurrent\train_features_%s.csv.libfm' % (runningPath, each)
#         else:
#             combin_train_libfm_cmd += r' %s\..\input\cocurrent\train_features_%s.csv.libfm' % (runningPath, each)
#             
#     combin_train_total_libfm_cmd = combin_train_libfm_cmd.copy()
#     combin_train_libfm_cmd += r' %s\..\input\cocurrent\train_features.libfm' % (runningPath, each)
#     
#     combin_train_total_libfm_cmd += r'+%s\..\input\cocurrent\train_features_%s.csv.libfm' % (runningPath, train_date[-1])

    return

def main():
    libfm_and_onehot_cocurrent()
   
    return

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':
    main()