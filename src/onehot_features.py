'''
Created on Mar 29, 2018

@author: Heng.Zhang
'''
from global_variables import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def onehot_features(df):
    oht = OneHotEncoder()
    for level_dict in [ITEM_LEVELS_DICT, SHOP_LEVELS_DICT]:
        for level_name, item_level in level_dict.items():        
            oht.fit(np.array(item_level).reshape(-1, 1))
            tmp = pd.DataFrame(oht.transform(df[level_name].values.reshape(-1, 1)).toarray())
            tmp.rename(columns=lambda x : "%s_%d" % (level_name, x), inplace=True)
            df = pd.concat([df, tmp], axis=1)

    print(getCurrentTime(), 'after onehot, shape', df.shape)
    return df