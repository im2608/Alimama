'''
Created on Mar 1, 2018

@author: Heng.Zhang
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

runningPath = sys.path[0]

def set_subplot(df, colname):
    value_trade_dict = dict()
    values = df[colname].unique()
    for each in values:
        value_trade_dict[each] = df[(df[colname] == each) & (df['is_trade'] == 1)].shape[0]
    plt.xlabel(colname)
    plt.grid(True)
    x = []
    y = []
    for k, v in value_trade_dict.items():
        x.append(k)
        y.append(v)
    
    plt.xticks(x)
    plt.yticks(y)
    plt.bar(x, y)
    return

def shop_trade(df):    
    plt.subplot(231)
    set_subplot(df, 'shop_review_num_level')
    
    plt.subplot(232)
    set_subplot(df, 'shop_review_positive_rate')
    
    plt.subplot(233)
    set_subplot(df, 'shop_star_level')
    
    plt.subplot(234)
    set_subplot(df, 'shop_score_service')
    
    plt.subplot(235)
    set_subplot(df, 'shop_score_delivery')
    
    plt.subplot(236)
    set_subplot(df, 'shop_score_description')
    
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
def handle_timestamp(timeStamp):
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    date = dateArray.strftime("%Y-%m-%d")
    hour = dateArray.strftime("%H")
    return date, hour

def preprocess_timestame(df):
    date_hour = df['context_timestamp'].apply(handle_timestamp) # date_hour is a Series
    date_hour = date_hour.apply(pd.Series)
    date_hour.rename(columns={0:'date', 1:'hour'}, inplace=True)
    return pd.concat((df, date_hour), axis=1)
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

if __name__ == '__main__':
    dftest = pd.read_csv(r'%s\..\input\test.txt' % runningPath)
    df = pd.read_csv(r'%s\..\input\train.txt' % runningPath)
#     onehot_item_category_property_list(df, 'item_category_list', 'item_cat_')
    onehot_item_property_list(df, dftest)
    
#     hour_trade(df)
#     item_cat_trade(df)
    
    