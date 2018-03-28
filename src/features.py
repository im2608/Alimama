'''
Created on Mar 15, 2018

@author: Heng.Zhang
'''

import pandas as pd 

import numpy as np
from global_variables import getCurrentTime


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


def fillin_user_click(user_click_info, click_at_hour_df, prog):
    user_id = user_click_info['user_id']
    hour = user_click_info['hour']
    click_at_hour_df.loc[click_at_hour_df['user_id'] == user_id, 'user_click_at_hour_%d' % hour] = user_click_info['user_click_at_hour'] 
    click_at_hour_df.loc[click_at_hour_df['user_id'] == user_id, 'user_click_ratio_at_hour_%d_on_day' % hour] = user_click_info['user_click_hour_ratio_at_day']
   
    prog['idx'] += 1
    if (prog['idx'] % 1000 == 0):
        print(getCurrentTime(), "%d lines handeld" % prog['idx'])

    return

def user_click_count(df): 
    def user_click_on_column_at_hour(df, colname, click_at_hour):
        # 用户在 colname 上各个小时的点击数量
        click_on_col_at_hour = df[['user_id', colname, 'hour']].groupby(['user_id', colname, 'hour'], sort=False, as_index=False) 
        click_on_col_at_hour = click_on_col_at_hour.size().reset_index()
        click_on_col_at_hour.rename(columns={0:"user_click_on_%s_at_hour" % colname}, inplace=True)

        # 用户在 colname 上各个小时的点击数量/用户各个小时的点击数量
        tmp = pd.merge(click_on_col_at_hour, click_at_hour, how='left', on=['user_id', 'hour'])
        tmp['user_click_ratio_on_%s_at_hour' % colname] = tmp["user_click_on_%s_at_hour" % colname] / tmp['user_click_at_hour']

        df = pd.merge(df, tmp[['user_id', colname, 'hour', 'user_click_ratio_on_%s_at_hour' % colname]], how='left', on=['user_id', colname, 'hour'])
        df.fillna(0, inplace=True)

        return df
    
    # 用户各个小时的点击数量
    click_at_hour = df[['user_id', 'hour']].groupby(['user_id', 'hour'], sort=False, as_index=False) 
    click_at_hour = click_at_hour.size().reset_index()
    click_at_hour.rename(columns={0:"user_click_at_hour"}, inplace=True) 
    
    click_col = ["user_click_at_hour_%d" % h for h in range(24)] # 用户各个小时的点击数量onehot成列
    click_col.extend(['user_click_ratio_at_hour_%d_on_day' % h for h in range(24)]) #  用户在各个小时的点击数量/用户一天的点击数量 onehot成列
    click_at_hour_df = pd.DataFrame(columns=click_col)
    click_at_hour_df['user_id'] = click_at_hour['user_id'].unique()
    
    # 用户一天的点击数量
    click_whole_day = df[['user_id', 'date']].groupby(['user_id', 'date'], sort=False, as_index=False)
    click_whole_day = click_whole_day.size().reset_index()
    click_whole_day.rename(columns={0:"click_whole_day"}, inplace=True)
    del click_whole_day['date']
    
    prog = {'idx':0}

    #  用户在各个小时的点击数量/用户一天的点击数量
    tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=['user_id'])
    tmp['user_click_hour_ratio_at_day'] = tmp['user_click_at_hour'] / tmp['click_whole_day']
    tmp.apply(fillin_user_click, axis=1, args = (click_at_hour_df, prog, )) # 在行上apply
    
    click_at_hour_df.fillna(0, inplace=True)
    
    # 用户在各个小时的最大/最小点击数量
    tmp = click_at_hour.groupby(['user_id'], as_index=False, sort=False)
    click_at_hour_df['user_click_max_at_hour'] = tmp.max()['user_click_at_hour']
    click_at_hour_df['user_click_min_at_hour'] = tmp.min()['user_click_at_hour'] 
    
    df = pd.merge(df, click_at_hour_df, how='left', on=['user_id'])
    df.fillna(0, inplace=True)
    
    # 用户在item上各个小时的点击数量/用户各个小时的点击数量
    df = user_click_on_column_at_hour(df, 'item_id', click_at_hour)    
    
    # 用户在 shop 上各个小时的点击数量/用户各个小时的点击数量
    df = user_click_on_column_at_hour(df, 'shop_id', click_at_hour)

    return df
    
