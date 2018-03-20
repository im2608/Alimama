'''
Created on Mar 15, 2018

@author: Heng.Zhang
'''

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, '%.2f' % float(height))
        
def trade_ratio(df, colname):
    gp = df[[colname, 'is_trade']].groupby([colname, 'is_trade'], sort=True, as_index=False)
    gp = gp.size().unstack().reset_index()
    gp['trade ratio'] = gp[1]/(gp[0] + gp[1])
#     gp = gp.sort_values('trade ratio', axis=0, ascending=False)
    plt.xticks(range(gp.shape[0]), gp[colname], rotation=90)
    plt.xlabel('trade ratio on ' + colname)
    plt.bar(range(gp.shape[0]), gp['trade ratio'])
    plt.grid(True)

    return
        
# 用户在context page 购买数量占该  context page 总数量的比例
def context_page_trade_ratio(df):
    context_page = df[['context_page_id', 'is_trade']].groupby(['context_page_id', 'is_trade'], sort=True, as_index=False)
    context_page = context_page.size().unstack().reset_index()
    context_page['trade ratio'] = context_page[1]/(context_page[0] + context_page[1])
    context_page = context_page.sort_values('trade ratio', axis=0, ascending=False)
    plt.xticks(range(context_page.shape[0]), context_page['context_page_id'], rotation=90)
    plt.xlabel("trade ratio on each context page")
    rects = plt.bar(range(context_page.shape[0]), context_page['trade ratio'])
    
    plt.show()
    return

# 用户在 hour 上的购买数量占该 hour 总数量的比例
def hour_trade_ratio(df):
    hour_trade = df[['hour', 'is_trade']].groupby(['hour', 'is_trade'], sort=True, as_index=False)
    hour_trade = hour_trade.size().unstack().reset_index()
    hour_trade['hour trade ratio'] = hour_trade[1]/(hour_trade[0] + hour_trade[1])
    hour_trade = hour_trade.sort_values('hour trade ratio', axis=0, ascending=False)
    plt.xticks(range(hour_trade.shape[0]), hour_trade['hour'], rotation=90)
    rects = plt.bar(range(hour_trade.shape[0]), hour_trade['hour trade ratio'])

    plt.xlabel("trade ratio at each hour")
#     autolabel(rects)
    
    plt.show()    
    return


def ad_trade_ratio(df):
    plt.subplot(221)
    trade_ratio(df, "item_price_level")
     
    plt.subplot(222)
    trade_ratio(df, "item_sales_level")
     
    plt.subplot(223)
    trade_ratio(df, "item_collected_level")
     
    plt.subplot(224)
    trade_ratio(df, "item_pv_level")
    
    plt.show()
    return

def user_trade_ratio(df):
    plt.subplot(221)
    trade_ratio(df, "user_gender_id")
    
    plt.subplot(222)
    trade_ratio(df, "user_age_level")
    
    plt.subplot(223)
    trade_ratio(df, "user_occupation_id")
    
    plt.subplot(224)
    trade_ratio(df, "user_star_level")
    
    plt.show()

    return

def shop_trafe_ratio(df):
    plt.subplot(231)
    trade_ratio(df, "shop_review_num_level")
    
    plt.subplot(232)
    trade_ratio(df, "shop_review_positive_rate")
    
    plt.subplot(233)
    trade_ratio(df, "shop_star_level")
    
    plt.subplot(234)
    trade_ratio(df, "shop_score_service")
    
    plt.subplot(235)
    trade_ratio(df, "shop_score_delivery")
    
    plt.subplot(236)
    trade_ratio(df, "shop_score_description")
    
    plt.show()    

    return



