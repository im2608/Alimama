'''
Created on Mar 15, 2018

@author: Heng.Zhang
'''

import pandas as pd 

import numpy as np


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


def user_click_count_at_hour(df): 
    def user_click_on_column_at_hour(df, colname, click_at_hour):
        # 用户在 colname 上各个小时的点击数量
        click_on_col_at_hour = df[['user_id', colname, 'hour']].groupby(['user_id', colname, 'hour'], sort=False, as_index=False) 
        click_on_col_at_hour = click_on_col_at_hour.size().reset_index()
        click_on_col_at_hour.rename(columns={0:"click_on_%s_at_hour" % colname}, inplace=True)

        # 用户在 colname 上各个小时的点击数量/用户各个小时的点击数量
        tmp = pd.merge(click_on_col_at_hour, click_at_hour, how='left', on=['user_id', 'hour'])
        tmp['click_ratio_on_%s_at_hour' % colname] = tmp["click_on_%s_at_hour" % colname] / tmp['click_at_hour']

        df = pd.merge(df, tmp[['user_id', colname, 'hour', 'click_ratio_on_%s_at_hour' % colname]], how='left', on=['user_id', colname, 'hour'])
        df.fillna(0, inplace=True)

        return df
    
    # 用户各个小时的点击数量
    click_at_hour = df[['user_id', 'hour']].groupby(['user_id', 'hour'], sort=False, as_index=False) 
    click_at_hour = click_at_hour.size().reset_index()
    click_at_hour.rename(columns={0:"click_at_hour"}, inplace=True) 
    
    # 用户一天的点击数量
    click_whole_day = df[['user_id', 'date']].groupby(['user_id', 'date'], sort=False, as_index=False)
    click_whole_day = click_whole_day.size().reset_index()
    click_whole_day.rename(columns={0:"click_whole_day"}, inplace=True)
    del click_whole_day['date']

    #  用户在各个小时的点击数量/用户一天的点击数量
    tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=['user_id'])
    tmp['click_hour_ratio_at_day'] = tmp['click_at_hour'] / tmp['click_whole_day']
    df = pd.merge(df, tmp[['user_id', 'click_at_hour', 'click_hour_ratio_at_day']], how='left', on='user_id')
    
    # 用户在各个小时的最大/最小点击数量
    tmp = click_at_hour.groupby(['user_id'], as_index=False, sort=False)
    click_min_max = tmp.max()
    click_min_max.rename(columns={'click_at_hour':"user_click_max_at_hour"}, inplace=True)
    click_min_max['user_click_min_at_hour'] = tmp.min()['click_at_hour']
    del click_min_max['hour']
    
    df = pd.merge(df, click_min_max, how='left', on=['user_id'])
    df.fillna(0, inplace=True)
    
    # 用户在item上各个小时的点击数量/用户各个小时的点击数量
    df = user_click_on_column_at_hour(df, 'item_id', click_at_hour)    
    
    # 用户在 shop 上各个小时的点击数量/用户各个小时的点击数量
    df = user_click_on_column_at_hour(df, 'shop_id', click_at_hour)

    return df
    
