'''
Created on Mar 15, 2018

@author: Heng.Zhang
'''

import pandas as pd 

import numpy as np
from global_variables import *
# 
# def click_count(df, prefix_col):
#     def click_on_column_at_hour(df, prefix_col, colname, click_at_hour):
#         # prefix_col 在 colname 上各个小时的点击数量
#         click_on_col_at_hour = df[[prefix_col, colname, 'hour']].groupby([prefix_col, colname, 'hour'], sort=False, as_index=False) 
#         click_on_col_at_hour = click_on_col_at_hour.size().reset_index()
#         click_on_col_at_hour.rename(columns={0:"%s_click_on_%s_at_hour" %(prefix_col, colname)}, inplace=True)
# 
#         # prefix_col 在 colname 上各个小时的点击数量/prefix_col 各个小时的点击数量
#         tmp = pd.merge(click_on_col_at_hour, click_at_hour, how='left', on=[prefix_col, 'hour'])
#         tmp['%s_click_ratio_on_%s_at_hour' % (prefix_col, colname)] = tmp["%s_click_on_%s_at_hour" % (prefix_col, colname)] / tmp[prefix_col + '_click_at_hour']
# 
#         df = pd.merge(df, tmp[[prefix_col, colname, 'hour', '%s_click_ratio_on_%s_at_hour' % (prefix_col, colname)]], how='left', on=[prefix_col, colname, 'hour'])
#         df.fillna(0, inplace=True)
# 
#         return df
#     
#     # prefix_col 各个小时的点击数量
#     click_at_hour = df[[prefix_col, 'hour']].groupby([prefix_col, 'hour'], sort=False, as_index=False) 
#     click_at_hour = click_at_hour.size().reset_index()
#     click_at_hour.rename(columns={0:"%s_click_at_hour" % prefix_col}, inplace=True) 
#     
#     
#     # prefix_col 一天的点击数量
#     click_whole_day = df[[prefix_col, 'date']].groupby([prefix_col, 'date'], sort=False, as_index=False)
#     click_whole_day = click_whole_day.size().reset_index()
#     click_whole_day.rename(columns={0:"click_whole_day"}, inplace=True)
#     del click_whole_day['date']
# 
#     #  prefix_col 在各个小时的点击数量/prefix_col 一天的点击数量
#     tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=[prefix_col])
#     tmp['%s_click_hour_ratio_at_day' % prefix_col] = tmp['%s_click_at_hour' % prefix_col] / tmp['click_whole_day']
#     df = pd.merge(df, tmp[[prefix_col, '%s_click_at_hour' % prefix_col, '%s_click_hour_ratio_at_day' % prefix_col]], how='left', on=prefix_col)
#     
#     # prefix_col 在各个小时的最大/最小点击数量
#     tmp = click_at_hour.groupby([prefix_col], as_index=False, sort=False)
#     click_min_max = tmp.max()
#     click_min_max.rename(columns={'%s_click_at_hour' % prefix_col:"%s_click_max_at_hour" % prefix_col}, inplace=True)
#     click_min_max['%s_click_min_at_hour' % prefix_col] = tmp.min()['%s_click_at_hour' % prefix_col]
#     del click_min_max['hour']
#     
#     df = pd.merge(df, click_min_max, how='left', on=[prefix_col])
#     df.fillna(0, inplace=True)
#     
#     # shop 的 item 上各个小时的点击数量/shop 各个小时的点击数量
#     df = shop_click_on_column_at_hour(df, 'item_id', click_at_hour)    
# 
#     return df
#             

def shop_click_ratio_on_level(df, level_name):
    # shop level 上的一天总的点击数量
    tmp = df[[level_name]].groupby(level_name, sort=False, as_index=False)
    tmp = tmp.size().reset_index()
    tmp.rename(columns={0:"%s_click_cnt" % level_name}, inplace=True)
    
    df = pd.merge(df, tmp, how='left', on=[level_name])
    df['shop_click_ratio_on_%s' % level_name] = df['shop_click_whole_day'] / df["%s_click_cnt" % level_name]
    return df

def shop_features(X_features, df): 
    def shop_click_on_column_at_day(X_features, df, colname, col_level):
        # shop 在 colname 上的点击数量
        click_on_col = df[['shop_id', colname]].groupby(['shop_id', colname], sort=False, as_index=False) 
        click_on_col = click_on_col.size().unstack().fillna(0)
        for each_level in col_level: # group 之后,  level 上可能没数据，在这里补全
            if (each_level not in click_on_col.columns):
                click_on_col[each_level] = 0

        click_on_col.rename(columns=lambda level: 'shop_click_on_%s_%d' % (colname, level), inplace=True)
        click_on_col = click_on_col.reset_index()
        
        X_features = pd.merge(X_features, click_on_col, how='left', on='shop_id')
        # shop 在 colname 上的点击数量 / shop 一整天的点击数量
        for each_level in col_level:
            X_features['shop_click_ratio_on_%s_%d' % (colname, each_level)] = X_features['shop_click_on_%s_%d' % (colname, each_level)]/X_features['shop_click_whole_day']
            
        X_features.fillna(0, inplace=True)

        return X_features
    
    def shop_click_on_column_at_hour(X_features, df, colname):
        # shop 在 colname 上各个小时的点击数量
        click_on_col_at_hour = df[['shop_id', colname, 'hour']].groupby(['shop_id', colname, 'hour'], sort=False, as_index=False) 
        click_on_col_at_hour = click_on_col_at_hour.size().unstack().fillna(0)
        for hour in range(24):
            if (hour not in click_on_col_at_hour.columns):
                click_on_col_at_hour[hour] = 0.0

        click_on_col_at_hour.rename(columns=lambda hour : "shop_click_on_%s_at_hour_%d" % (colname, hour), inplace=True)
        click_on_col_at_hour = click_on_col_at_hour.reset_index()
        X_features = pd.merge(X_features, click_on_col_at_hour, how='left', on=['shop_id', colname])
        X_features.fillna(0, inplace=True)
        
        # shop 在 colname 上各个小时的点击数量 / shop 在各个 hour 上总的点击数量
        for hour in range(24):
            df['shop_click_ratio_on_%s_at_hour_%d' % (colname, hour)] = df['shop_click_at_hour_%d' % hour] / df["shop_click_on_%s_at_hour_%d" % (colname, hour)]

        df.fillna(0, inplace=True)

        return df
    
    # shop 各个小时的点击数量
    click_at_hour = df[['shop_id', 'hour']].groupby(['shop_id', 'hour'], sort=False, as_index=False) 
    click_at_hour = click_at_hour.size().unstack()
    for hour in range(24): # group 之后, 某个 hour 上可能没数据，在这里补全
        if (hour not in click_at_hour.columns):
            click_at_hour[hour] = 0

    click_at_hour.rename(columns=lambda x: "shop_click_at_hour_%s" % str(x), inplace=True)
    click_at_hour = click_at_hour.reset_index()
    
    # shop 一天的点击数量
    click_whole_day = df[['shop_id', 'date']].groupby(['shop_id', 'date'], sort=False, as_index=False)
    click_whole_day = click_whole_day.size().reset_index()
    click_whole_day.rename(columns={0:"shop_click_whole_day"}, inplace=True)
    del click_whole_day['date']

    #  shop 在各个小时的点击数量 / shop 一天的点击数量
    tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=['shop_id'])
    for hour in range(24):
        tmp['shop_click_ratio_at_hour_%s_at_day' % (hour)] = tmp['shop_click_at_hour_%d' % (hour)] / tmp['shop_click_whole_day']
    tmp.fillna(0, inplace=True)
    X_features = pd.merge(X_features, tmp, how='left', on=['shop_id'])
    X_features.fillna(0, inplace=True)
    
    # shop 在 item 上各个小时的点击数量 / shop 在各个小时的点击数量 
    X_features = shop_click_on_column_at_hour(X_features, df, 'item_id')
    
    for col, level in ITEM_LEVELS_DICT.items():
        X_features = shop_click_on_column_at_day(X_features, df, col, level) # shop 在 level 上的点击数量 / shop 一整天的点击数量

    # shop 的 item 上的点击数量
    click_on_col = df[['shop_id', 'item_id']].groupby(['shop_id', 'item_id'], sort=False, as_index=False) 
    click_on_col = click_on_col.size().reset_index()
    click_on_col.rename(columns={0:"item_click_of_shop"}, inplace=True)
    X_features = pd.merge(X_features, click_on_col, how='left', on=['shop_id', 'item_id'])
    
    # shop 的 item 上的点击数量 /  shop 一整天的点击数量
    X_features['item_click_ratio_of_shop'] = X_features['item_click_of_shop'] / X_features['shop_click_whole_day']
    X_features.fillna(0, inplace=True)
    
    # shop 一整天的点击数量 / 该 shop level 总的点击数量 
    for level_name in SHOP_LEVELS_DICT.keys():
        X_features = shop_click_ratio_on_level(X_features, df, level_name)

    return X_features