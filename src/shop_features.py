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



def fillin_shop_click(shop_click_info, click_at_hour_df, prog):
    shop_id = shop_click_info['shop_id']
    hour = shop_click_info['hour']
    click_at_hour_df.loc[click_at_hour_df['shop_id'] == shop_id, 'shop_click_at_hour_%d' % hour] = shop_click_info['shop_click_at_hour'] 
    click_at_hour_df.loc[click_at_hour_df['shop_id'] == shop_id, 'shop_click_ratio_at_hour_%d_on_day' % hour] = shop_click_info['shop_click_hour_ratio_at_day']
   
    prog['idx'] += 1
    if (prog['idx'] % 1000 == 0):
        print(getCurrentTime(), "%d lines handeld" % prog['idx'])

    return

def shop_click_count(df): 
    def shop_click_on_column_at_hour(df, colname, click_at_hour):
        # shop 在 colname 上各个小时的点击数量
        click_on_col_at_hour = df[['shop_id', colname, 'hour']].groupby(['shop_id', colname, 'hour'], sort=False, as_index=False) 
        click_on_col_at_hour = click_on_col_at_hour.size().reset_index()
        click_on_col_at_hour.rename(columns={0:"shop_click_on_%s_at_hour" % colname}, inplace=True)

        # shop 在 colname 上各个小时的点击数量/shop 各个小时的点击数量
        tmp = pd.merge(click_on_col_at_hour, click_at_hour, how='left', on=['shop_id', 'hour'])
        tmp['shop_click_ratio_on_%s_at_hour' % colname] = tmp["shop_click_on_%s_at_hour" % colname] / tmp['shop_click_at_hour']

        df = pd.merge(df, tmp[['shop_id', colname, 'hour', 'shop_click_ratio_on_%s_at_hour' % colname]], how='left', on=['shop_id', colname, 'hour'])
        df.fillna(0, inplace=True)

        return df
    
    # shop 各个小时的点击数量
    click_at_hour = df[['shop_id', 'hour']].groupby(['shop_id', 'hour'], sort=False, as_index=False) 
    click_at_hour = click_at_hour.size().reset_index()
    click_at_hour.rename(columns={0:"shop_click_at_hour"}, inplace=True) 
    
    click_col = ["shop_click_at_hour_%d" % h for h in range(24)] # shop 各个小时的点击数量onehot成列
    click_col.extend(['shop_click_ratio_at_hour_%d_on_day' % h for h in range(24)]) #  shop 在各个小时的点击数量/shop 一天的点击数量 onehot成列
    click_at_hour_df = pd.DataFrame(columns=click_col)
    click_at_hour_df['shop_id'] = click_at_hour['shop_id'].unique()
    
    # shop 一天的点击数量
    click_whole_day = df[['shop_id', 'date']].groupby(['shop_id', 'date'], sort=False, as_index=False)
    click_whole_day = click_whole_day.size().reset_index()
    click_whole_day.rename(columns={0:"click_whole_day"}, inplace=True)
    del click_whole_day['date']
    
    prog = {'idx':0}

    #  shop 在各个小时的点击数量/shop 一天的点击数量
    tmp = pd.merge(click_at_hour, click_whole_day, how='left', on=['shop_id'])
    tmp['shop_click_hour_ratio_at_day'] = tmp['shop_click_at_hour'] / tmp['click_whole_day']
    tmp.apply(fillin_shop_click, axis=1, args = (click_at_hour_df, prog, )) # 在行上apply
    
    click_at_hour_df.fillna(0, inplace=True)
    
    # shop 在各个小时的最大/最小点击数量
    tmp = click_at_hour.groupby(['shop_id'], as_index=False, sort=False)
    click_at_hour_df['shop_click_max_at_hour'] = tmp.max()['shop_click_at_hour']
    click_at_hour_df['shop_click_min_at_hour'] = tmp.min()['shop_click_at_hour'] 
    
    df = pd.merge(df, click_at_hour_df, how='left', on=['shop_id'])
    df.fillna(0, inplace=True)
    
    # shop 中 item 上各个小时的点击数量/shop 各个小时的点击数量
    df = shop_click_on_column_at_hour(df, 'item_id', click_at_hour)    

    return df
    