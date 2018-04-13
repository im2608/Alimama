'''
Created on Apr 12, 2018

@author: Heng.Zhang
'''


import numpy as np
from global_variables import *
from features import *
from shop_features import *
from item_features import *
from onehot_features import *



# 得到onehot之后，每个列的索引
def create_onehot_col_idx(df, colname, sparse_mat_col_idx_dict):
    col_val_list = df[colname].unique()
    for each_val in col_val_list:
        col_name = colname + np.str(each_val)
        if (col_name not in sparse_mat_col_idx_dict):
            sparse_mat_col_idx_dict[col_name] = sparse_mat_col_idx_dict['idx']
            sparse_mat_col_idx_dict['idx'] += 1
    return


def extract_features_libfm(df, sparse_mat_col_idx_dict):
    old_features = set(df.columns)
    df = shop_features(df)
    df = user_features(df)
    df = item_features(df)

    new_features = set(np.sort(list(set(df.columns) - old_features)))
    for each in new_features:
        if (each not in sparse_mat_col_idx_dict):
            sparse_mat_col_idx_dict[each] = sparse_mat_col_idx_dict['idx']
            sparse_mat_col_idx_dict['idx'] += 1

    return df, new_features
