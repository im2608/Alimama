'''
Created on Mar 15, 2018

@author: Heng.Zhang
'''

import pandas as pd 

import numpy as np
from global_variables import *


#  一个 item 只会属于一个  shop
def item_click_ratio_on_level(df, level_name):
    # item level 上的一天总的点击数量
    tmp = df[[level_name]].groupby(level_name, sort=False, as_index=False)
    tmp = tmp.size().reset_index()
    tmp.rename(columns={0:"%s_click_cnt" % level_name}, inplace=True)

    df = pd.merge(df, tmp, how='left', on=[level_name])
    df['item_click_ratio_on_%s' % level_name] = df['item_click_whole_day'] / df["%s_click_cnt" % level_name]
    return df


def item_features(df): 
    def item_click_on_column_at_day(df, colname, col_level):
        # item 在 colname 上的点击数量
        click_on_col = df[['item_id', colname]].groupby(['item_id', colname], sort=False, as_index=False) 
        click_on_col = click_on_col.size().unstack().fillna(0)
        for each_level in col_level: # group 之后,  level 上可能没数据，在这里补全
            if (each_level not in click_on_col.columns):
                click_on_col[each_level] = 0

        click_on_col.rename(columns=lambda level: 'item_click_on_%s_%d' % (colname, level), inplace=True)
        click_on_col = click_on_col.reset_index()
        
        df = pd.merge(df, click_on_col, how='left', on='item_id')
        # item 在 colname 上的点击数量 / item 一整天的点击数量
        for each_level in col_level:
            df['item_click_ratio_on_%s_%d' % (colname, each_level)] = df['item_click_on_%s_%d' % (colname, each_level)]/df['item_click_whole_day']
            
        df.fillna(0, inplace=True)

        return df
    
    def item_click_on_column_at_hour(df, colname):
        # item 在 colname 上各个小时的点击数量
        click_on_col_at_hour = df[['item_id', colname, 'hour']].groupby(['item_id', colname, 'hour'], sort=False, as_index=False) 
        click_on_col_at_hour = click_on_col_at_hour.size().unstack().fillna(0)
        for hour in range(24):
            if (hour not in click_on_col_at_hour.columns):
                click_on_col_at_hour[hour] = 0.0

        click_on_col_at_hour.rename(columns=lambda hour : "item_click_on_%s_at_hour_%d" % (colname, hour), inplace=True)
        click_on_col_at_hour = click_on_col_at_hour.reset_index()
        df = pd.merge(df, click_on_col_at_hour, how='left', on=['item_id', colname])
        df.fillna(0, inplace=True)
        
        # item 在 colname 上各个小时的点击数量 / item 在各个 hour 上总的点击数量
        for hour in range(24):
            df['item_click_ratio_on_%s_at_hour_%d' % (colname, hour)] = df['item_click_at_hour_%d' % hour] / df["item_click_on_%s_at_hour_%d" % (colname, hour)]

        df.fillna(0, inplace=True)

        return df
    
    # item 各个小时的点击数量
    click_at_hour = df[['item_id', 'hour']].groupby(['item_id', 'hour'], sort=False, as_index=False) 
    click_at_hour = click_at_hour.size().unstack()
    for hour in range(24): # group 之后, 某个 hour 上可能没数据，在这里补全
        if (hour not in click_at_hour.columns):
            click_at_hour[hour] = 0

    click_at_hour.rename(columns=lambda x: "item_click_at_hour_%d" % x, inplace=True)
    click_at_hour = click_at_hour.reset_index()
    
    # item 一天的点击数量
    click_whole_day = df[['item_id', 'date']].groupby(['item_id', 'date'], sort=False, as_index=False)
    click_whole_day = click_whole_day.size().reset_index()
    click_whole_day.rename(columns={0:"item_click_whole_day"}, inplace=True)
    del click_whole_day['date']

    #  item 在各个小时的点击数量/item 一天的点击数量
    tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=['item_id'])
    for hour in range(24):
        tmp['item_click_ratio_at_hour_%d_at_day' % hour] = tmp['item_click_at_hour_%d' % hour] / tmp['item_click_whole_day']
    tmp.fillna(0, inplace=True)
    df = pd.merge(df, tmp, how='left', on=['item_id'])
    df.fillna(0, inplace=True)
    
    # item 一整天的点击数量 / 该  item level 总的点击数量 
    for level_name in ITEM_LEVELS_DICT.keys():
        df = item_click_ratio_on_level(df, level_name) # item 在 level 上的点击数量 / item level 一整天的点击数量
    
    return df