'''
Created on Mar 1, 2018

@author: Heng.Zhang
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from global_variables import *
from features import *
from shop_features import *
from item_features import *
from onehot_features import *
import scipy
from visualize_feature import *
from commonfunc import *
import json

pd.options.display.float_format = '{:.2f}'.format

from scipy.sparse import dok_matrix

def split_item_property_list(item_property_lis_str, property_dict):
    property_list = item_property_lis_str.split(";")
    for each_prop in property_list:
        if (each_prop not in property_dict):
            property_dict[each_prop] = property_dict['idx']
            property_dict['idx'] += 1

def onehot_item_property_list(df, dftest):
    property_dict = {'idx':0}
    df['item_property_list'].apply(split_item_property_list, args=(property_dict,))  #  得到每个property对应的列索引
    dftest['item_property_list'].apply(split_item_property_list, args=(property_dict,))  #  得到每个property对应的列索引

    prop_dok_train = dok_matrix((df.shape[0], len(property_dict)), dtype=np.bool) # 创建一个稀疏矩阵
    prop_dok_test = dok_matrix((dftest.shape[0], len(property_dict)), dtype=np.bool)

    print("property_dict len is ", len(property_dict))
    
    for sample_idx in range(df.shape[0]):
        property_list = df.iloc[sample_idx]['item_property_list'].split(";")
        for each_prop in property_list:
            prop_dok_train[sample_idx, property_dict[each_prop]] = True
        
        if (sample_idx % 10000 == 0):
            print("%d train lines read\r" % (sample_idx), end="")

    for sample_idx in range(dftest.shape[0]):
        property_list = dftest.iloc[sample_idx]['item_property_list'].split(";")
        for each_prop in property_list:
            prop_dok_test[sample_idx, property_dict[each_prop]] = True

        if (sample_idx % 10000 == 0):
            print("%d test lines read\r" % (sample_idx), end="")

    np.save(r"%s\..\input\item_property_sparse_train" % runningPath, prop_dok_train)
    np.save(r"%s\..\input\item_property_sparse_test" % runningPath, prop_dok_test)

    return


#####################################################################################################################
#####################################################################################################################
####################################################################################################################
def set_sparse_matrix(df, sparse_dok, sparse_mat_col_idx_dict, onehot_col_list, float_col_list):
    rows = df.shape[0]
    for row_idx in df.index:
        if (row_idx % 1000 == 0):
            print(getCurrentTime(), "%d / %d lines read\r" % (row_idx, rows), end="")

        for onehot_col_name in onehot_col_list:
            tmp = onehot_col_name + np.str(df.loc[row_idx, onehot_col_name])
            sparse_dok[row_idx, sparse_mat_col_idx_dict[tmp]] = 1

        for float_col_name in float_col_list:
            sparse_dok[row_idx, sparse_mat_col_idx_dict[float_col_name]] = df.loc[row_idx, float_col_name]

        property_list = df.iloc[row_idx]['item_property_list'].split(";")
        for each_prop in property_list:
            sparse_dok[row_idx, sparse_mat_col_idx_dict[each_prop]] = 1


def write_libfmtxtfile_and_dok(each_sample, output_libfm_file, sparse_mat_col_idx_dict, args_apply):
    if (each_sample.name % 1000 == 0):
        l1000 = time.clock()
        print(getCurrentTime(), "%d / %d lines read, p1 %d, p2 %d, p3 %d (3-1 %d), time used %d\r" % 
              (each_sample.name, 
               args_apply['total'], args_apply['p1_time'], args_apply['p2_time'], args_apply['p3_time'], args_apply['p3_1_time'],
               l1000 - args_apply['L1000']), end="")

    s = time.clock()

    if ('is_trade' in each_sample):
        each_line = ["%d" % each_sample['is_trade']]
    else:
        each_line = ["%d" % 0]

    sparse_dok = args_apply['dok']

#     for onehot_col_name in onehot_col_list:
#         col_idx = sparse_mat_col_idx_dict[onehot_col_name + np.str(each_sample[onehot_col_name])]
#         sparse_dok[each_sample.name, col_idx] = 1
#         each_line.append("%d:1" % col_idx)

    p1_e = time.clock()
    
    args_apply['p1_time'] += p1_e - s;

#     property_list = each_sample['item_property_list'].split(";")
#     for each_prop in property_list:
#         col_idx = sparse_mat_col_idx_dict[each_prop]
#         each_line.append("%d:1" % col_idx)
#         sparse_dok[each_sample.name, col_idx] = 1
        
    p2_e = time.clock()
    args_apply['p2_time'] += p2_e - p1_e;

    # 用户在点击某个 item/shop 时，之前 1， 2， 3 小时内还点击过哪些其他的 item/shop 以及次数  
#     user_id = each_sample['user_id']
#     timestamp = each_sample['context_timestamp']
#     for hour in [1,2,3]:
#         timestamp_hour = timestamp - hour * 3600
#         p3_0 = time.clock()        
#         
#         df_tuple = (user_id, timestamp_hour, timestamp)
#         if (df_tuple not in args_apply):
#             user_opt_hour = df[(df['user_id'] == user_id)& (df['context_timestamp'] >= timestamp_hour)&(df['context_timestamp'] < timestamp)]
#             args_apply[df_tuple] = user_opt_hour
#         else:
#             user_opt_hour =  args_apply[df_tuple]
# 
#         if (user_opt_hour.shape[0] > 0):
#             for colname in ['item_id', 'shop_id']:
#                 user_clicked_cnt = user_opt_hour[['user_id', colname]].groupby(colname).size()
#                 for each_item in user_clicked_cnt.index:  
#                     col_idx = sparse_mat_col_idx_dict["user_clicked_%s_befroe_%dh_%s" % (colname, hour, each_item)]
#                     each_line.append("%d:%d" % (col_idx, user_clicked_cnt[each_item]))
#                     sparse_dok[each_sample.name, col_idx] = user_clicked_cnt[each_item]

    float_features = args_apply['float_features']
    for each in float_features:
        value = each_sample[each]
        if (value == 0):
            continue
  
        col_idx = sparse_mat_col_idx_dict[each]
        if (type(value) == float):
            each_line.append("%d:%.4f" % (col_idx, value))
        else:
            each_line.append("%d:%d" % (col_idx, value))

        sparse_dok[each_sample.name, col_idx] = value

    for each_labeld in LAB_ENCODE.keys():
        value = each_sample[each_labeld]
        if (value == 0):
            continue

        col_idx = sparse_mat_col_idx_dict["%s%s" % (each_labeld, str(value))]
        each_line.append("%d:%.1f" % (col_idx, LAB_ENCODE[each_labeld][value]))
        sparse_dok[each_sample.name, col_idx] = LAB_ENCODE[each_labeld][value]

    output_libfm_file.write("%s\n" % " ".join(each_line))

    args_apply['idx'] += 1
    return

# column 在onehot之后，在稀疏矩阵中的 列索引
# onehot_col_list = ['user_id', 'shop_id', 
#                    'item_id', 'item_brand_id', 
#                    'item_city_id']
#                    'item_price_level', 'item_sales_level',
#                    'item_pv_level', 'context_page_id', 
#                    'hour', 'item_category_list',
#                    'user_gender_id', 'user_age_level', 'user_occupation_id',
#                    'user_star_level', 'shop_review_num_level', 'shop_star_level',
#                    'shop_score_service', 'shop_review_positive_rate',
#                    'shop_score_delivery', 'shop_score_description']

# onehot_col_list = ['user_id', 'shop_id', 
#                    'item_id', 'item_brand_id', 
#                    'item_city_id']

# 用户在点击某个 item/shop 时，之前 1， 2， 3 小时内还点击过哪些其他的 item/shop 以及次数 , 记录这些列的索引
def user_clicked_info(df, sparse_mat_col_idx_dict):
    cols = ['item_id', 'shop_id']
    for colname in cols:
        col_vals = df[colname].unique()
        for each_val in col_vals:
            colname_before_hours = ["user_clicked_%s_befroe_%dh_%s" % (colname, hour, each_val) for hour in [1,2,3]]
            for each_hour_col in colname_before_hours:
                if (each_hour_col not in sparse_mat_col_idx_dict):
                    sparse_mat_col_idx_dict[each_hour_col] = sparse_mat_col_idx_dict['idx']
                    sparse_mat_col_idx_dict['idx'] += 1
    return


def onehot_columns_to_libfmtextfile_and_sparsemat(dffile):

    with open(r'%s\..\input\cocurrent\col_idx_dict.txt' % runningPath, 'r') as col_idx_file:
        sparse_mat_col_idx_dict = json.load(col_idx_file)

    df = pd.read_csv(r'%s\..\input\cocurrent\%s'  % (runningPath, dffile))

    dok_sparse = dok_matrix((df.shape[0], len(sparse_mat_col_idx_dict)), dtype=np.float) 

    float_features = []
    with open(r'%s\..\input\cocurrent\float_features.txt' % runningPath, 'r') as float_features_file:
        line = float_features_file.read()
        float_features = line.split(",")

    args_apply = {'idx':0, 'total':df.shape[0], 'dok':dok_sparse,
                 'p1_time':0, 'p2_time':0, 'p3_time':0, 'p3_1_time':0, 'L1000':0, 
                 'float_features':float_features, 'labeled_features':labeled_features, 
                 }

    output_libfm_file = open(r"%s\..\input\cocurrent\%s.libfm" % (runningPath, dffile), mode='w')

    df.apply(write_libfmtxtfile_and_dok, axis=1, args=(output_libfm_file, sparse_mat_col_idx_dict, args_apply, ))# 在行上apply

    print(getCurrentTime(), "writting sparse matrix to %s\..\input\%s.sparse" % (runningPath, dffile))    
    np.save(r"%s\..\input\cocurrent\%s.sparse" % (runningPath, dffile), dok_sparse)
    print(getCurrentTime(), "writting sparse matrix to %s\..\input\%s.sparse done!" % (runningPath, dffile))
    return
    
def onehot_columns(dftrain_total, dftest):
    sparse_mat_col_idx_dict = {'idx':0}

    # column 在onehot之后，在稀疏矩阵中的 列索引
    onehot_col_list = ['item_id',
                       'item_brand_id', 
                       'item_city_id',          'user_gender_id', 
                       'user_age_level',        'user_occupation_id', 
                       'user_star_level',       'context_page_id',
                       'hour',                  'shop_star_level',
                       'shop_review_num_level', 'shop_id', 'item_category_list']

    for onehot_col_name in onehot_col_list:
        create_onehot_col_idx(dftrain_total, onehot_col_name,sparse_mat_col_idx_dict)
        create_onehot_col_idx(dftest, onehot_col_name,sparse_mat_col_idx_dict)

     #  得到每个property的列索引
    dftrain_total['item_property_list'].apply(split_item_property_list, args=(sparse_mat_col_idx_dict,))
    dftest['item_property_list'].apply(split_item_property_list, args=(sparse_mat_col_idx_dict,)) 
    
    # 这些列已经计算完毕的特征值，不需要onehont编码，直接copy到稀疏矩阵中
    float_col_list = ['shop_review_positive_rate', 'shop_score_service', 
                       'shop_score_delivery', 'shop_score_description']
#     float_col_list.extend(new_features_name)
    for float_col in float_col_list:
        sparse_mat_col_idx_dict[float_col] = sparse_mat_col_idx_dict['idx']
        sparse_mat_col_idx_dict['idx'] += 1 

    print(getCurrentTime(), "sparse_mat_col_idx_dict len is %d" % (sparse_mat_col_idx_dict['idx']))
    
    df_train = dftrain_total[dftrain_total['date'] != '2018-09-24']
    df_train.index = range(df_train.shape[0])
    
    df_verify = dftrain_total[dftrain_total['date'] == '2018-09-24']
    df_verify.index = range(df_verify.shape[0])

    # 创建稀疏矩阵, 将train 与 train verify 分成两个稀疏矩阵存储，否则从train 中抽取 train_verify 太慢
    # dok_total_train 则是 train 转换成的稀疏矩阵， 在训练完确定模型参数后，再用dok_total_train重新训练一次
#     dok_total_train = dok_matrix((dftrain_total.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float)
    dok_train = dok_matrix((df_train.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float) 
    dok_verify = dok_matrix((df_verify.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float) 
    dok_test = dok_matrix((dftest.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float)

    df_list =       [df_train,  df_verify,  dftest]
    sparse_matrix = [dok_train, dok_verify, dok_test]
    for df, dok in zip(df_list, sparse_matrix):
        set_sparse_matrix(df, dok, sparse_mat_col_idx_dict, onehot_col_list, float_col_list)

    print(getCurrentTime(), "here are %d nonzeor values in train %s, %d nonzero values in verify %s, %d nonzero values in test %s" %
                            (dok_train.nnz, dok_train.shape, dok_verify.nnz, dok_verify.shape, dok_test.nnz, dok_test.shape))

    np.save(r"%s\..\input\sparse_train_simple" % (runningPath), dok_train)
    np.save(r"%s\..\input\sparse_verify_simple" % (runningPath), dok_verify)
    np.save(r"%s\..\input\sparse_test_simple" % (runningPath), dok_test)

    df_train[['instance_id', 'is_trade']].to_csv(r"%s\..\input\train_label.txt" % (runningPath), index=False)
    df_verify[['instance_id', 'is_trade']].to_csv(r"%s\..\input\verify_label.txt" % (runningPath), index=False)
    dftest[['instance_id']].to_csv(r"%s\..\input\test_instanceid.txt" % runningPath, index=False)
    
    dok_total_train = scipy.sparse.vstack([dok_train, dok_verify])
    np.save(r"%s\..\input\sparse_train_total_simple" % (runningPath), dok_total_train)
    
    print(getCurrentTime(), "done")

    return 0

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# 按出现次数处理-1值
def allocate_neg1(df):
    colnames = ['user_occupation_id', 'user_age_level', 'user_gender_id', 'item_brand_id', 'item_city_id', 'item_sales_level', 'user_star_level']
    
    for col in colnames:
        neg1s = df[df[col] == -1].shape[0]
        neg1_idx = df[df[col] == -1].index
        vals = list(df[col].unique())
        vals.remove(-1)
        
        val_nums = [df[df[col] == each_val].shape[0] for each_val in vals]  # 每个值得数量
        num_allocate_neg1 = np.round(neg1s  * (val_nums / np.sum(val_nums)))  # 每个值得数量占所有值得比例 * -1 的数量
        num_allocate_neg1 = np.array(num_allocate_neg1, dtype=np.int)
        total = 0
        for each_num, each_val in zip(num_allocate_neg1, vals):
            df.loc[neg1_idx[total:total+each_num], col] = each_val
            total += each_num

        print(col, 'done')
        
    colsetto0 = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery','shop_score_description']
    for col in colsetto0:
        df.loc[df[df[col] == -1].index, col] = 0
        
    return df
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# 将doulbe值取到小数点后两位
def round_ratio(df):
    double_colname = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
    for each in double_colname:
        df[each] = np.round(np.round(df[each] * 100, 2)/100, 2)
     
    
    return df
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def extract_features(df):
    old_features = set(df.columns)
#     df = onehot_features(df)
    df = shop_features(df)
    df = user_features(df)
    df = item_features(df)

    new_features = list(np.sort(list(set(df.columns) - old_features)))

    if ('is_trade' in df.columns):
        new_features.append('is_trade')

    new_features.append('instance_id')

    return df[new_features]

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def single_preprocess(df, dftest):
    train_date = ['2018-09-17', '2018-09-18', '2018-09-19', '2018-09-20', '2018-09-21', '2018-09-22', '2018-09-23', '2018-09-24']
    X_train = []
    original_features = set(dftest.columns)
    for each_date in train_date:
        each_df_of_date  = extract_features(df[df['date'] == each_date])
        print(getCurrentTime(), "df %s after extracting feature shape %s" % (each_date, each_df_of_date.shape))
        X_train.append(each_df_of_date)
    
    df = pd.concat(X_train, axis=0, ignore_index=True)
    print(getCurrentTime(), "df after all extracting feature shape", df.shape)

    test_date = ['2018-09-24', '2018-09-25']
    X_test = []
    for each_date in test_date:
        each_df_of_date = extract_features(dftest[dftest['date'] == each_date])
        print(getCurrentTime(), "dftest %s after extracting feature shape %s" % (each_date, each_df_of_date.shape))
        X_test.append(each_df_of_date)
    dftest = pd.concat(X_train, axis=0, ignore_index=True)    
    print(getCurrentTime(), "df after all extracting feature shape",  dftest.shape)
    
    new_features = set(dftest.columns) - original_features

    return


def cocurrent_preprocess(df, dftest):
    original_features = set(dftest.columns)

    date = sys.argv[1].split("=")[1]
    print(getCurrentTime(), "running for date ", date)
    df = df[df['date'] == date]
    dftest = dftest[dftest['date'] == date]

    df = extract_features(df)
    dftest = extract_features(dftest)
    new_features = set(dftest.columns) - original_features
    
    print(getCurrentTime(), "after %s, extract_features, df shape %s, dftest shape %s, new features %d" % 
          (date, df.shape, dftest.shape, len(new_features)))

    onehot_columns(df, dftest, list(new_features), date)   
    return

def cocurrent_preprocess_output():
    date = sys.argv[1].split("=")[1]
    dffilename = sys.argv[2].split("=")[1]
    print(getCurrentTime(), "running for date %s, %s" % (date, dffilename))
    df = pd.read_csv(r'%s\..\input\%s.txt' % (runningPath, dffilename))
    
    df = df[df['date'] == date]
    df.index = range(df.shape[0])

    df = extract_features(df)
    
    print(getCurrentTime(), "after extracting features %s, %s, shape %s" % (date, dffilename, df.shape))
    df.to_csv(r'%s\..\input\%s_feature_%s.txt' % (runningPath, dffilename, date), index=False)

    return


def rem_1st_item_category(each_item_category_list):
    item_cat_list = each_item_category_list.split(";")
    return item_cat_list[-1]

def handle_item_category(df, outputfilename):
    df['item_category'] = df['item_category_list'].apply(rem_1st_item_category)
    del df['item_category_list']
    
    df.rename(columns={'item_category':'item_category_list'}, inplace=True)
    df.loc[df['item_category_list'] == '2642175453151805566;6233669177166538628', 'item_category_list'] = '6233669177166538628'
    df.loc[df['item_category_list'] == '2642175453151805566;8868887661186419229', 'item_category_list'] = '8868887661186419229'
    df.to_csv(r'%s\..\input\%s' % (runningPath,outputfilename), index=False)
    return

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def level_label_encode(df, outputfilename):
    for colname, col_lab_encode in LAB_ENCODE.items():
        print(getCurrentTime(), 'handling %s %s' % (outputfilename, colname))
        df[colname + "_lab_encode"] = df[colname].map(col_lab_encode)
        df[colname + "_lab_encode"].fillna(0, inplace=True)

    df.to_csv(r'%s\..\input\%s' % (runningPath,outputfilename), index=False)
    return

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':
    dffile = sys.argv[1].split("=")[1]
    print(getCurrentTime(), "running for %s " %(dffile))

    onehot_columns_to_libfmtextfile_and_sparsemat(dffile)


    