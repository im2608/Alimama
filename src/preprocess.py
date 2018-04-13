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

pd.options.display.float_format = '{:.2f}'.format

def set_subplot(df, colname, how_display=plt.bar):
    value_trade_dict = dict()
    values = df[colname].unique()
    for each in values:
        value_trade_dict[each] = df[(df[colname] == each) & (df['is_trade'] == 1)].shape[0]
    plt.xlabel(colname)
    plt.grid(True)
    x = []
    y = []
    for k in np.sort(list(value_trade_dict.keys())):
        x.append(k)
        y.append(value_trade_dict[k])
    
    plt.xticks(x, rotation=90)
    plt.yticks(y)
    how_display(x, y)
    return

def shop_trade(df):    
    plt.subplot(231)
    set_subplot(df, 'shop_review_num_level', how_display=plt.scatter)
    
    plt.subplot(232)
    set_subplot(df, 'shop_review_positive_rate', how_display=plt.scatter)
    
    plt.subplot(233)
    set_subplot(df, 'shop_star_level', how_display=plt.scatter)
    
    plt.subplot(234)
    set_subplot(df, 'shop_score_service', how_display=plt.scatter)
    
    plt.subplot(235)
    set_subplot(df, 'shop_score_delivery', how_display=plt.scatter)
    
    plt.subplot(236)
    set_subplot(df, 'shop_score_description', how_display=plt.scatter)
    
    plt.show()
     
    return


def user_trade(df):
    plt.subplot(221)
    set_subplot(df, 'user_star_level')
    
    plt.subplot(222)
    set_subplot(df, 'user_age_level')
    
    plt.subplot(223)
    set_subplot(df, 'user_occupation_id')
    
    plt.subplot(224)
    set_subplot(df, 'user_gender_id')
    
    plt.show()
    return


def context_trade(df):
    set_subplot(df, 'context_page_id')
    
    plt.show()
    
    return


def ad_trade(df):
    plt.subplot(221)
    set_subplot(df, 'item_price_level')
     
    plt.subplot(222)
    set_subplot(df, 'item_sales_level')
     
    plt.subplot(223)
    set_subplot(df, 'item_collected_level')
     
    plt.subplot(224)
    set_subplot(df, 'item_pv_level')
    
    # item_city_id 与 is_trade 关系不大
#     plt.subplot(325)
#     set_subplot(df, 'item_city_id', how_display=plt.scatter)
    
    plt.show()
    return


def hour_trade(df):
    set_subplot(df, 'hour')
    plt.show()
    return


def get_trade_num(df, colname, need_trade = True):
    col_val = df[colname].unique()
    col_val_trade_num = {}
    
    for each in col_val:
        if (need_trade):
            col_val_trade_num[each] = df[(df[colname]==each)&(df['is_trade']==1)].shape[0]
        else:
            col_val_trade_num[each] = df[(df[colname]==each)].shape[0]
        
    return col_val_trade_num


# 删除  predict_category_property 中的-1
def handle_predict_category_property(predict_category_property):
    removed_nag1_category_property = ""
    if (predict_category_property == "-1"):
        return removed_nag1_category_property
    
    category_property = predict_category_property.split(";")
    for each_cat_prop in category_property:
        category, props = each_cat_prop.split(":")        
        if (props == "-1"):
            continue

        props = props.split(",")
        if (len(removed_nag1_category_property) > 0):
            removed_nag1_category_property += ";"

        removed_nag1_category_property += "%s:%s" % (category, ",".join(props))

    return removed_nag1_category_property


# 删除 item_category_list 中的-1
def handle_propertylist(propertylist):
    removed_nag1_property = ""
    
    if (propertylist == "-1" ):
        return removed_nag1_property
    
    properties = propertylist.split(";")
    for each in properties:
        if (each == -1):
            continue 
        
        if (len(removed_nag1_property) > 0):
            removed_nag1_property += ";"
            
        removed_nag1_property += each

    return removed_nag1_property

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# 将 context_timestamp 转换成 date， time
def handle_timestamp(timeStamp, prog):
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    date = dateArray.strftime("%Y-%m-%d")
    hour = dateArray.strftime("%H")
    prog['idx'] += 1
    
    if (prog['idx'] % 10000 == 0):
        print("handled %d lines" % prog['idx'])
    return date, hour

def preprocess_timestame(df):
    prog={'idx':0}
    date_hour = df['context_timestamp'].apply(handle_timestamp, args=(prog,)) # date_hour is a Series
    date_hour = date_hour.apply(pd.Series)
    date_hour.rename(columns={0:'date', 1:'hour'}, inplace=True)
    del df['context_timestamp']
    return pd.concat((df, date_hour), axis=1)
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# 计算 predict_category_property 对 item_category_list， item_property_list 的 f1 评分
# 统计 cat f1, prop f1 在各个值时 is_trade = 1/0 的数量， 查看是否对查询词的cat， prop预测的越准， is_trade 就越高
# 但是通过计算， cat f1, prop f1 的评分与 is_trade 无关, 所以 predict_category_property 没有用处 -- 2018-03-14
def calculate_cat_prop_f1(each_sample, prog):
    prog['idx'] += 1
    if (each_sample['predict_category_property'] == '-1'):
        if (prog['idx'] % 10000 == 0):
            print("%d lines handeld" % prog['idx'])

        return 0, 0

    predicted_cat = set()
    predicted_prop = set()
    
    predict_category= each_sample['predict_category_property'].split(";")
    for each_cat in predict_category:
        category, props = each_cat.split(":")
        predicted_cat.add(category)
        predicted_prop.add(props)
        
    item_cats = set(each_sample['item_category_list'].split(";"))
    item_props = set(each_sample['item_property_list'].split(";"))
    
    cat_f1 = 0
    
    if (len(predicted_cat) > 0):
        cat_hit_cnt = len(predicted_cat & item_cats)
        cat_p = cat_hit_cnt / len(predicted_cat)
        cat_r = cat_hit_cnt / len(item_cats)
        if (cat_hit_cnt > 0):
            cat_f1 = 2 * cat_p * cat_r / (cat_p + cat_r)
        
    prop_f1 = 0
    if (len(predicted_prop) > 0):
        prop_hit_cnt = len(predicted_prop & item_props)
        prop_p = prop_hit_cnt / len(predicted_prop)
        prop_r = prop_hit_cnt / len(item_props)
        if (prop_hit_cnt):
            prop_f1 = 2 * prop_p * prop_r/ (prop_p + prop_r)
            
    if (prog['idx'] % 10000 == 0):
        print("%d lines handeld" % prog['idx'])

    return round(cat_f1, 2), round(prop_f1, 2)
 
def predicted_cat_prop_f1(df):
    prog = {'idx': 0}
    cat_prop_f1 = df.apply(calculate_cat_prop_f1, axis=1, args = (prog, )) # 在行上apply
    cat_prop_f1 = cat_prop_f1.apply(pd.Series)
    
    cat_prop_f1.to_csv(r"%s\..\input\cat_prop_f1.csv" % runningPath, index=False)
    
    return 

# 统计 cat f1, prop f1 在各个值时 is_trade = 1/0 的数量
def f1_trade(each_f1, cat_trade_dict, prop_trade_dict, prog):
    if (each_f1['cat_f1'] not in cat_trade_dict):
        cat_trade_dict[each_f1['cat_f1']] = 1
    else:
        cat_trade_dict[each_f1['cat_f1']] += 1
        
    if (each_f1['prop_f1'] not in prop_trade_dict):
        prop_trade_dict[each_f1['prop_f1']] = 1
    else:
        prop_trade_dict[each_f1['prop_f1']] += 1
        
    prog['idx'] += 1
    if (prog['idx'] % 10000 == 0):
        print("%d lines handeld" % prog['idx'])

    return


def predict_subplot(f1_trade_dict, xaxisname, x_upper):
    x = []
    y = []
    f1 = np.sort(list(f1_trade_dict.keys()))
    for each_f1 in f1:
        x.append(each_f1)
        y.append(f1_trade_dict[each_f1])

    plt.xlim(0, x_upper)
    plt.grid(True)
    plt.xticks(x, rotation=90)
    plt.yticks(y) 
    plt.scatter(x, y)    
    plt.xlabel(xaxisname)
    return


# 
def cat_prop_f1_trade(df):
    cat_prop_f1 = pd.read_csv(r"%s\..\input\cat_prop_f1.csv" % runningPath)
    
    cat_prop_f1['cat_f1'] = np.round(cat_prop_f1['cat_f1'] *100, 2) / 100
    cat_prop_f1['prop_f1'] = np.round(cat_prop_f1['prop_f1'] *100, 2) / 100
    
    trade_idx = df[df['is_trade'] == 1].index
    not_trade_idx = df[df['is_trade'] == 0].index
    
    cat_f1_trade_dict = {}
    cat_f1_not_trade_dict = {}
    prop_f1_trade_dict = {}
    prop_f1_not_trade_dict = {}
    
    porg = {'idx':0}
    
    # 得到在cat， prop 某个f1值下，is_trade = 1的数量
    cat_prop_f1.iloc[trade_idx].apply(f1_trade, axis=1, args=(cat_f1_trade_dict, prop_f1_trade_dict, porg)) # 在行上apply
    
    # 得到在cat， prop 某个f1值下，is_trade != 1的数量
    cat_prop_f1.iloc[not_trade_idx].apply(f1_trade, axis=1, args=(cat_f1_not_trade_dict, prop_f1_not_trade_dict, porg, )) # 在行上apply
    
    plt.subplot(221)
    predict_subplot(cat_f1_trade_dict, "category f1 is_trade =1", 1)
    
    plt.subplot(222)
    predict_subplot(cat_f1_not_trade_dict, "category f1 is_trade = 0", 1)
    
    plt.subplot(223)
    predict_subplot(prop_f1_trade_dict, "property f1 is_trade = 1", 0.25)
    
    plt.subplot(224)
    predict_subplot(prop_f1_not_trade_dict, "property f1 is_trade = 0", 0.25)
    
    plt.show()
    

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


def split_item_category_list(item_category_list_str, category_set):
    category_arr = item_category_list_str.split(";")
    for each_cat in category_arr:
        category_set.add("item_cat_" + each_cat)
        
def onehot_encode_category_list(each_sample):
    category_arr = each_sample['item_category_list'].split(";")
    for each_cat in category_arr:
        each_sample["item_cat_" + each_cat] = 1
        
    return each_sample

# item_cat_7908382889764677758 在 train， test 上每个item都有， 删除

# 'item_cat_2011981573061447208', 'item_cat_1968056100269760729',
#        'item_cat_5755694407684602296', 'item_cat_5799347067982556520',
#        'item_cat_509660095530134768', 'item_cat_3203673979138763595',
#        'item_cat_6233669177166538628', 'item_cat_2642175453151805566',
#        'item_cat_4879721024980945592', 'item_cat_22731265849056483',
#        'item_cat_8277336076276184272', 'item_cat_2436715285093487584',
#        'item_cat_8710739180200009128', 'item_cat_7258015885215914736'


def onehot_item_category_list(df):
    category_set = set()
    df['item_category_list'].apply(split_item_category_list, args=(category_set,))
    
    print("category_set len ", len(category_set)) 
    
    category_property_dok = dok_matrix(np.zeros((df.shape[0], len(category_set))), dtype=np.bool)
    
#     df = pd.concat((df, category_property_df), axis=1, ignore_index=True) # 增加新的列
        
    print("after concat shape is ", df.shape)
        
    return df.apply(onehot_encode_category_list, axis=1) # 在行上apply
    
def item_cat_trade(df):
    item_cat_trade_dict = dict()

    item_cat_list = ['item_cat_8868887661186419229', 'item_cat_2011981573061447208', 'item_cat_1968056100269760729',
                    'item_cat_5755694407684602296', 'item_cat_5799347067982556520',
                    'item_cat_509660095530134768', 'item_cat_3203673979138763595',
                    'item_cat_6233669177166538628', 'item_cat_2642175453151805566',
                    'item_cat_4879721024980945592', 'item_cat_22731265849056483',
                    'item_cat_8277336076276184272', 'item_cat_2436715285093487584',
                    'item_cat_8710739180200009128', 'item_cat_7258015885215914736']
    
    for each_cat in item_cat_list:
        item_cat_trade_dict[each_cat] = df[(df[each_cat] == 1) & (df['is_trade'] == 1)].shape[0]

    plt.xlabel('item_category_list')
    plt.grid(True)
    x = []
    y = []
    
    for k, v in item_cat_trade_dict.items():
        x.append(k)
        y.append(v)
    
    plt.xticks(range(len(x)), x, rotation=90)
    plt.yticks(y, rotation=0)
    plt.bar(x, y)

    plt.show()
    
    return


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

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
#####################################################################################################################
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

    if (args_apply['is_train']):
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

    labeled_features = args_apply['labeled_features']
    for each_labeld in labeled_features:
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

def onehot_columns_to_libfmtextfile_and_sparsemat(dftrain_total, dftest):
    sparse_mat_col_idx_dict = {'idx':0}

    float_features = None

    test_date = ['2018-09-24', '2018-09-25']
    test_date = ['2018-09-24']
    X_test = []
    for each_date in test_date:
        print(getCurrentTime(), 'extracting test features on %s' % each_date)
        each_df_of_date, float_features = extract_features_libfm(dftest[dftest['date'] == each_date], sparse_mat_col_idx_dict)
        X_test.append(each_df_of_date)
    dftest = pd.concat(X_test, axis=0, ignore_index=True)
    
    # index 记录了原始instance id 顺序，按照index排序则恢复到原始顺序
    dftest = dftest.sort_values('index',  axis=0, ascending=True)
    del dftest['index']
    
    print(getCurrentTime(), "dftest after all extracting feature shape",  dftest.shape)
  
    train_date = ['2018-09-17', '2018-09-18', '2018-09-19', '2018-09-20', '2018-09-21', '2018-09-22', '2018-09-23', '2018-09-24']
    train_date = ['2018-09-17']
    X_train = []
    for each_date in train_date:
        print(getCurrentTime(), 'extracting train features on %s' % each_date)
        tmp_df = dftrain_total[dftrain_total['date'] == each_date]
        each_df_of_date, float_features = extract_features_libfm(tmp_df, sparse_mat_col_idx_dict)
        X_train.append(each_df_of_date)
    dftrain_total = pd.concat(X_train, axis=0, ignore_index=True)
    print(getCurrentTime(), "df after all extracting feature shape", dftrain_total.shape)
    for each in float_features:
        if (each not in sparse_mat_col_idx_dict):
            sparse_mat_col_idx_dict[each] = sparse_mat_col_idx_dict['idx']
            sparse_mat_col_idx_dict['idx'] += 1


#     for onehot_col_name in onehot_col_list:
#         create_onehot_col_idx(dftrain_total, onehot_col_name,sparse_mat_col_idx_dict)
#         create_onehot_col_idx(dftest, onehot_col_name,sparse_mat_col_idx_dict)
    
    labeled_features = set(['item_price_level', 'item_sales_level',
                        'item_pv_level', 'context_page_id', 
                        'hour', 'item_category_list',
                        'user_gender_id', 'user_age_level', 'user_occupation_id',
                        'user_star_level', 'shop_review_num_level', 'shop_star_level',
                        'shop_score_service', 'shop_review_positive_rate',
                        'shop_score_delivery', 'shop_score_description'])

    for each_labled in labeled_features:
        create_onehot_col_idx(dftrain_total, each_labled, sparse_mat_col_idx_dict)
        create_onehot_col_idx(dftest, each_labled, sparse_mat_col_idx_dict)

     #  得到每个property的列索引
#     dftrain_total['item_property_list'].apply(split_item_property_list, args=(sparse_mat_col_idx_dict,))
#     dftest['item_property_list'].apply(split_item_property_list, args=(sparse_mat_col_idx_dict,))

    # 用户在点击某个 item/shop 时，之前 1， 2， 3 小时内还点击过哪些其他的 item/shop 以及次数 , 记录这些列的索引
#     user_clicked_info(dftrain_total,sparse_mat_col_idx_dict)
#     user_clicked_info(dftest,sparse_mat_col_idx_dict)

    # 经过以上步骤，确定了稀疏矩阵有多少列
    print(getCurrentTime(), "sparse_mat_col_idx_dict len is %d" % (sparse_mat_col_idx_dict['idx']))

    df_train = dftrain_total[dftrain_total['date'] != '2018-09-24']
    df_train.index = range(df_train.shape[0])

    df_verify = dftrain_total[dftrain_total['date'] == '2018-09-24']
    df_verify.index = range(df_verify.shape[0])

    # 创建稀疏矩阵, 将train 与 train verify 分成两个稀疏矩阵存储，否则从train 中抽取 train_verify 太慢    
    dok_train = dok_matrix((df_train.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float) 
    dok_verify = dok_matrix((df_verify.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float) 
    dok_test = dok_matrix((dftest.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float)

    df_list =       [dftest,   df_train,  df_verify]
    sparse_matrix = [dok_test, dok_train, dok_verify]
    is_train = [False, True, True]
    sparse_filenames = ['sparse_test', 'sparse_train', 'sparse_verify']
    libfm_filenamess = ['libfm_test.txt', 'libfm_train.txt', 'libfm_verify.txt']

    for each_df, each_sparse_mat, b_train, sparse_filename, libfm_filename in zip(df_list, sparse_matrix, is_train, sparse_filenames, libfm_filenamess):
        print(getCurrentTime(), "handling %s  " % sparse_filename)    
        args_apply = {'idx':0, 'total':each_df.shape[0], 'is_train':b_train, 'dok':each_sparse_mat,
                      'p1_time':0, 'p2_time':0, 'p3_time':0, 'p3_1_time':0, 'L1000':0, 
                      'float_features':float_features, 'labeled_features':labeled_features, 
                      }

        output_libfm_file = open(r"%s\..\input\%s" % (runningPath, libfm_filename), mode='w')

        each_df.apply(write_libfmtxtfile_and_dok, axis=1, args=(output_libfm_file, sparse_mat_col_idx_dict, args_apply, ))# 在行上apply

        print(getCurrentTime(), "writting sparse matrix to %s\..\input\%s" % (runningPath, sparse_filename))
        np.save(r"%s\..\input\%s" % (runningPath, sparse_filename), each_sparse_mat)

#     dok_total_train 则是 train 转换成的稀疏矩阵， 在训练完确定模型参数后，再用dok_total_train重新训练一次
    print(getCurrentTime(), "combining tran and verify to Total..")
    dok_total_train = scipy.sparse.vstack([dok_train, dok_verify])
    print(getCurrentTime(), "saving sparse_total..")
    np.save(r"%s\..\input\sparse_train_total" % (runningPath), dok_total_train)

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
    print(getCurrentTime(), " running...")
    dftest = pd.read_csv(r'%s\..\input\test.txt' % runningPath)
    df = pd.read_csv(r'%s\..\input\train.txt' % runningPath)
    
#     single_preprocess(df, dftest)

#     handle_item_category(df, 'train.txt')
#     handle_item_category(dftest, 'test.txt')
#     level_label_encode(df, 'train.txt')
#     level_label_encode(dftest, 'test.txt')
    
#     df = round_ratio(df)
#     dftest = round_ratio(dftest)
#     df = allocate_neg1(df)
#     dftest = allocate_neg1(dftest)
#     df = preprocess_timestame(df)
#     dftest = preprocess_timestame(dftest)

#     df.to_csv(r'F:\doc\ML\TianChi\Alimama\input\train.txt', index=False)
#     dftest.to_csv(r'F:\doc\ML\TianChi\Alimama\input\test.txt', index=False)

#     user_trade_ratio(df)    
#     hour_trade_ratio(df)
#     ad_trade_ratio(df)
#     context_page_trade_ratio(df)
#     item_category_trade_ratio(df)
#     shop_trade_ratio(df)     
#     predicted_cat_prop_f1(df)
#     cat_prop_f1_trade(df)

#     onehot_columns(df, dftest)
    onehot_columns_to_libfmtextfile_and_sparsemat(df, dftest)
#     shop_trade(df)
#     hour_trade(df)
#     item_cat_trade(df)
#     ad_trade(df)

#     cocurrent_preprocess_output()


    