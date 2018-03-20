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
# 得到onehot之后，每个列的索引
def create_onehot_col_idx(df, colname, sparse_mat_col_idx_dict):
    col_val_list = df[colname].unique()
    for each_val in col_val_list:
        col_name = colname + np.str(each_val)
        if (col_name not in sparse_mat_col_idx_dict):
            sparse_mat_col_idx_dict[col_name] = sparse_mat_col_idx_dict['idx']
            sparse_mat_col_idx_dict['idx'] += 1
    return

def onehot_col_idx_predict_category_property(predict_category_property):
    if (predict_category_property is None):
        return
    
    category_property = predict_category_property.split(";")
    for each_cat_prop in category_property:
        category, props = each_cat_prop.split(":")        
        if (props == "-1"):
            continue


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
    
def onehot_columns(dftrain_total, dftest):
    sparse_mat_col_idx_dict = {'idx':0}
    
    # column 在onehot之后，在稀疏矩阵中的 列索引
    onehot_col_list = ['item_id',               'item_brand_id', 
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
    
    # 这些列不需要onehont编码，直接copy到稀疏矩阵中
    float_col_list = ['shop_review_positive_rate', 'shop_score_service', 
                      'shop_score_delivery', 'shop_score_description',]

    for float_col in float_col_list:
        sparse_mat_col_idx_dict[float_col] = sparse_mat_col_idx_dict['idx']
        sparse_mat_col_idx_dict['idx'] += 1 

    print(getCurrentTime(), "sparse_mat_col_idx_dict len is ", sparse_mat_col_idx_dict['idx']) # 76712
    
    df_train = dftrain_total[dftrain_total['date'] != '2018-09-24']
    df_train.index = range(df_train.shape[0])
    
    df_verify = dftrain_total[dftrain_total['date'] == '2018-09-24']
    df_verify.index = range(df_verify.shape[0])

    # 创建稀疏矩阵, 将train 与 train verify 分成两个稀疏矩阵存储，否则从train 中抽取 train_verify 太慢
    # dok_total_train 则是 train 转换成的稀疏矩阵， 在训练完确定模型参数后，再用dok_total_train重新训练一次
    dok_total_train = dok_matrix((dftrain_total.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float)
    dok_train = dok_matrix((df_train.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float) 
    dok_verify = dok_matrix((df_verify.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float) 
    dok_test = dok_matrix((dftest.shape[0], sparse_mat_col_idx_dict['idx'] + 1), dtype=np.float)

    df_list =       [df_train,  df_verify,  dftest,   dftrain_total]
    sparse_matrix = [dok_train, dok_verify, dok_test, dok_total_train]
    for df, dok in zip(df_list, sparse_matrix):
        set_sparse_matrix(df, dok, sparse_mat_col_idx_dict, onehot_col_list, float_col_list)

    print(getCurrentTime(), "here are %d nonzeor values in train, %d nonzero values in verify, %d nonzero values in test" %
                            (dok_train.nnz, dok_verify.nnz, dok_test.nnz))

    np.save(r"%s\..\input\sparse_train" % runningPath, dok_train)
    np.save(r"%s\..\input\sparse_verify" % runningPath, dok_verify)
    np.save(r"%s\..\input\sparse_test" % runningPath, dok_test)
    np.save(r"%s\..\input\sparse_train_total" % runningPath, dok_total_train)
    
    df_train['is_trade'].to_csv(r"%s\..\input\train_label.txt" % runningPath, index=False)
    df_verify['is_trade'].to_csv(r"%s\..\input\verify_label.txt" % runningPath, index=False)
    dftrain_total['is_trade'].to_csv(r"%s\..\input\train_label_total.txt" % runningPath, index=False)
    
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



if __name__ == '__main__':
    dftest = pd.read_csv(r'%s\..\input\test.txt' % runningPath)
    df = pd.read_csv(r'%s\..\input\train.txt' % runningPath)
    
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
#     shop_trafe_ratio(df)     
#     predicted_cat_prop_f1(df)
#     cat_prop_f1_trade(df)

    onehot_columns(df, dftest)
#     shop_trade(df)
#     hour_trade(df)
#     item_cat_trade(df)
#     ad_trade(df)

    
    
    