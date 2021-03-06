
import sys
import redis
import datetime
import time
runningPath = sys.path[0]
import pandas as pd


ISOTIMEFORMAT = "%Y-%m-%d %X"
def getCurrentTime():
 return time.strftime(ISOTIMEFORMAT, time.localtime())


ITEM_LEVELS_DICT = {
'item_price_level' :[i for i in range(18)], # price level 从 0--17,
'item_sales_level' : [i for i in range(1, 18)], # sales level 从1--17,
'item_collected_level' : [i for i in range(18)], # collect level 从 0--17,
'item_pv_level'  : [i for i in range(22)], # pv level 从 0--21,
   
   # label encode 之后的值
#   'item_price_level' :[0, 1,2,3,5,6,8],
#   'item_sales_level' : [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 13], 
#   'item_collected_level' : [ 2, 3, 5, 6, 7, 8, 9, 11], # collect level 从 0--17,
#   'item_pv_level'  : [0, 1, 2, 9], 
}

# 各个 level 销量与展示次数的比值， 比值越高，hash value 越大, 比值相近的 level value， hash value 也相近
LAB_ENCODE = {
    'item_price_level' :{
        0: 8, 1: 6, 2:5, 3:5, 4:5, 5:3, 6:2, 7:2,
        8:1, 9:1, 
        # prive level 10 -- 17 没有销量
        10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0,    
        },
    'item_sales_level' :{
        1:3, 2:2, 3:4, 4:2, 5:4, 6:4, 7:5,
        8:5, 9:5, 10:6, 11:7, 12:8, 13:9,
        14:10, 15:13, 16:13, 17:6
        },
    'item_collected_level' :{
        0:3, 1:3, 2:8, 3:2, 4:2, 5:2, 6:7, 7:5, 
        8:6, 9:7, 10:7, 11:7, 12:7, 13:7, 14:8,
        15:9, 16:8, 17:11
        },

    'item_pv_level' : {
        0:1, 1:0, 2:0, 3:9, 4:0, 5:0, 6:0, 
        7:1, 8:1, 9:1, 10:2, 11:2, 12:2, 13:2, 
        14:2, 15:2, 16:2, 17:2, 18:2, 19:2, 20:2, 21:2,   
        },
    
    'context_page_id' : {
        4001:1.9, 4002:2.2, 4003:1.85, 4004:1.9, 4005:1.8, 4006:1.7, 4007:1.9, 
        4008:1.85, 4009:1.6, 4010:1.7, 4011:1.4, 4012:1.6, 4013:1.6, 4014:1.4,
        4015:1.5, 4016:1.5, 4017:1.2, 4018:1.3, 4019:1.2, 4020:1.5, 
        },   
        
    'hour' : {
        0:2.3, 1:2.6, 2:2.2, 3:2.3, 4:2.3, 5:1.8, 6:1.9, 7:1.8, 8:1.6, 9:1.8, 10:1.9, 11:1.9, 12:1.8,
        13:1.7, 14:1.5, 15:1.3, 16:1.6, 17:1.6, 18:2.2, 19:1.8, 20:1.8, 21:1.9, 22:1.8, 23:2.2
        },

    'item_category_list' :{
        5799347067982556520 : 1.8,
        8277336076276184272 : 1.4,
        5755694407684602296 : 1.8,
        4879721024980945592 : 2.5, 
        509660095530134768 : 1.8, 
        2436715285093487584 : 2.5,
       7258015885215914736 : 4.0, 
       8710739180200009128 : 3.1, 
       2011981573061447208 : 1.8,
       22731265849056483 : 1.5, 
       3203673979138763595 : 3.6, 
       8868887661186419229 : 6.2,
       1968056100269760729 : 3.8,
       6233669177166538628 : 4.0
        }, 

    'user_gender_id' : {
        0:7, 1:8, 2:7
        }, 
    'user_age_level' : {
        1000:1.4, 1001:1.4, 1002:1.6, 1003:1.7, 1004:2.0, 1005:2.4, 1006:2.1, 1007:2.3
        }, 
    'user_occupation_id' : {
        2002:2.0, 2003:1.3, 2004:1.8, 2005:1.8
        },
    
    'user_star_level' : {
        3000:1.2, 3001:1.8, 3002:2.0, 3003:2.0, 3004:2.0, 3005:2.0, 3006:2.0, 3007:2.0, 3008:2.0,
        3008:2.2, 3009:2.8, 3010:2.8
        },
    
    'shop_review_num_level' : {
        0:0, 1:0, 2:23, 3:12, 4:11, 5:10, 6:20, 7:16, 8:17, 9:24,
        10:16,11:16, 12:16, 13:16, 14:16, 15:18, 16:22, 17:21, 18:17,
        19:23, 20:18, 21:10, 22:2, 23:11, 24:0, 25:0
        },
    'shop_star_level' : {
        4999:0, 5000:12, 5001:34, 5002:12, 5003:15, 5004:22, 5005:15,
        5006:18, 5007:24, 5008:16,5009:16, 5010:16, 5011:16, 5012:16,
        5013:22,5014:22, 5015:18, 5016:20, 5017:18, 5018:10, 5019:15, 
        5020:10
        },
    
    'shop_score_service' : {
        0:5, 79:0, 84:0, 85:0, 86:0, 87:7, 88:0, 89:0, 90:0, 
        91:2, 92:1, 93:2.6, 94:2.3, 95:3, 96:2.6, 97:2, 98:1.5,
        99:1, 100:1.5
        },
    
    'shop_review_positive_rate' : {
        0:0, 71:0, 83:0, 86:0, 87:0, 88:0, 89:0, 
        90:2, 91:0, 92:5, 93:1.4, 94:1.5, 95:2.2,
        96:3, 97:3.2, 98:2.5, 99:2, 100:1.8
        },
    
    'shop_score_delivery' : {
        0:5, 83:0, 84:0, 85:0, 86:0, 87:0, 88:0, 89:4, 90:0, 91:0, 
        92:2, 93:2, 94:2, 95:3, 96:2.8, 97:1.8, 98:1.5, 99:1, 100:1.2 
        },
    
    'shop_score_description' : {
        0:0, 79:0, 80:0, 81:0, 82:0, 83:0, 84:0, 85:0, 86:0, 87:4.9, 88:0, 89:0, 90:0, 
        91:1.5, 92:1.5, 93:2.8, 94:3.2, 95:3.2, 96:2.8, 97:1.8, 98:1.8, 99:1.8, 100:1.4 
        }
    }
SHOP_LEVELS_DICT = {
'shop_review_num_level' : [i for i in range(26)], # shop_review_num_level 从 0--25
'shop_star_level' : [4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020],
'shop_review_positive_rate' : [ 0., 71., 83., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 100.],
'shop_score_service' : [ 0., 79., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 100.],
'shop_score_delivery' : [ 0., 83., 84., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 100.],
'shop_score_description' : [ 0., 79., 80., 81., 83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 100.]
  
  # label encode 之后的值
#   'shop_review_num_level' : [ 0,  2, 10, 11, 12, 16, 17, 18, 20, 21, 22, 23, 24], 
#   'shop_star_level' : [  0, 10, 12, 15, 16, 18, 20, 22, 24, 34.],
#   'shop_review_positive_rate' :  [ 0,  1.4, 1.5,  1.8 , 2,  2.2,  2.5,  3,  3.2,  5. ],
#   'shop_score_service' : [0,  1,  1.5, 2,  2.3,  2.6,  3,  5,  7. ],
#   'shop_score_delivery' : [ 0,  1,  1.2,  1.5,  1.8,  2,  2.8,  3,  4,  5. ],
#   'shop_score_description' : [ 0,  1.4,  1.5,  1.8,  2.8,  3.2,  4.9],
  }

HOUR_LAB_ENCODE = [ 1.3,  1.5,  1.6,  1.7,  1.8,  1.9 , 2.2,  2.3 , 2.6]



def order_instance_id_as_test(prediction, dftest):
    ordered_predition = pd.DataFrame(dftest['instance_id'])
    ordered_predition['predicted_score'] = 0.0
    for i in range(dftest.shape[0]):
        ordered_predition.loc[i, 'predicted_score'] = prediction[prediction['instance_id'] == dftest.iloc[i]['instance_id']]['predicted_score'].values[0]
        if (i % 1000 == 0):
            print("%s %d lines ordered\r" % (getCurrentTime(), i), end='')

    return ordered_predition



labeled_features = set(['item_price_level', 'item_sales_level',
                    'item_pv_level', 'context_page_id', 
                    'hour', 'item_category_list',
                    'user_gender_id', 'user_age_level', 'user_occupation_id',
                    'user_star_level', 'shop_review_num_level', 'shop_star_level',
                    'shop_score_service', 'shop_review_positive_rate',
                    'shop_score_delivery', 'shop_score_description'])


