'''
Created on Mar 15, 2018

@author: Heng.Zhang
'''

import pandas as pd 

import numpy as np
from global_variables import *


# 得到 col name 各个值中 is_trade = 1 的比例        
def get_trade_ratio(df, colname):
    gp = df[[colname, 'is_trade']].groupby([colname, 'is_trade'], sort=True, as_index=False)
    gp = gp.size().unstack().reset_index()
    gp.fillna(0, inplace=True)
    gp['trade ratio of ' + colname] = gp[1]/(gp[0] + gp[1])
    df = pd.merge(df, gp, how='left', on=[colname])
    return df, gp

def trade_ratio_of_colname(df, ratio_of_colname):
    for each in ratio_of_colname:
        df, _ = get_trade_ratio(df, each) 
    return df


def user_features(df): 
    def user_click_on_column_at_day(df, colname, col_level):
        # 用户在 colname 上的点击数量
        click_on_col = df[['user_id', colname]].groupby(['user_id', colname], sort=False, as_index=False) 
        click_on_col = click_on_col.size().unstack().fillna(0)
        for each_level in col_level: # group 之后,  level 上可能没数据，在这里补全
            if (each_level not in click_on_col.columns):
                click_on_col[each_level] = 0

        click_on_col.rename(columns=lambda level: 'user_click_on_%s_%d' % (colname, level), inplace=True)
        click_on_col = click_on_col.reset_index()
        
        df = pd.merge(df, click_on_col, how='left', on='user_id')
        # 用户在 colname 上的点击数量 / 用户一整天的点击数量
        for each_level in col_level:
            df['user_click_ratio_on_%s_%d' % (colname, each_level)] = df['user_click_on_%s_%d' % (colname, each_level)]/df['user_click_whole_day']
            
        df.fillna(0, inplace=True)

        return df

    def user_click_on_column_at_hour(df, colname):
        # 用户在 colname 上各个小时的点击数量
        click_on_col_at_hour = df[['user_id', colname, 'hour']].groupby(['user_id', colname, 'hour'], sort=False, as_index=False) 
        click_on_col_at_hour = click_on_col_at_hour.size().unstack().fillna(0)
        for hour in range(24):
            if (hour not in click_on_col_at_hour.columns):
                click_on_col_at_hour[hour] = 0.0

        click_on_col_at_hour.rename(columns=lambda hour : "user_click_on_%s_at_hour_%d" % (colname, hour), inplace=True)
        click_on_col_at_hour = click_on_col_at_hour.reset_index()
        df = pd.merge(df, click_on_col_at_hour, how='left', on=['user_id', colname])
        df.fillna(0, inplace=True)

        # 用户在 colname 上各个小时的点击数量 / 用户在各个 hour 上总的点击数量
        for hour in range(24):
            df['user_click_ratio_on_%s_at_hour_%d' % (colname, hour)] = df["user_click_on_%s_at_hour_%d" % (colname, hour)] / df['user_click_at_hour_%d' % hour]

        df.fillna(0, inplace=True)

        return df

    # 用户各个小时的点击数量
    click_at_hour = df[['user_id', 'hour']].groupby(['user_id', 'hour'], sort=False, as_index=False) 
    click_at_hour = click_at_hour.size().unstack()
    for hour in range(24): # group 之后, 某个 hour 上可能没数据，在这里补全
        if (hour not in click_at_hour.columns):
            click_at_hour[hour] = 0

    click_at_hour.rename(columns=lambda x: "user_click_at_hour_%d" % x, inplace=True)
    click_at_hour = click_at_hour.reset_index()
    
    # 用户一天的点击数量
    click_whole_day = df[['user_id', 'date']].groupby(['user_id', 'date'], sort=False, as_index=False)
    click_whole_day = click_whole_day.size().reset_index()
    click_whole_day.rename(columns={0:"user_click_whole_day"}, inplace=True)
    del click_whole_day['date']

    #  用户在各个小时的点击数量/用户一天的点击数量
    tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=['user_id'])
    for hour in range(24):
        tmp['user_click_ratio_at_hour_%d_at_day' % hour] = tmp['user_click_at_hour_%d' % hour] / tmp['user_click_whole_day']
    tmp.fillna(0, inplace=True)
    df = pd.merge(df, tmp, how='left', on=['user_id'])
    df.fillna(0, inplace=True)

    # 用户在item上各个小时的点击数量 / 用户在各个小时的点击数量 
    df = user_click_on_column_at_hour(df, 'item_id')
    
    # 用户在 shop 上各个小时的点击数量/ 用户在各个小时的点击数量
    df = user_click_on_column_at_hour(df, 'shop_id')
    
    for col, level in ITEM_LEVELS_DICT.items():
        df = user_click_on_column_at_day(df, col, level) # 用户在 level 上的点击数量 / 用户一整天的点击数量
    
    for col, level in SHOP_LEVELS_DICT.items():
        df = user_click_on_column_at_day(df, col, level) # 用户在 level 上的点击数量 / 用户一整天的点击数量

    return df
    
