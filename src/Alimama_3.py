'''
Created on Mar 8, 2018

@author: Heng.Zhang
'''

from global_variables import *
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from preprocess import *

from sklearn.linear_model import LogisticRegression

def main():
    print(getCurrentTime(), "running...")

    test_date = ['2018-09-24', '2018-09-25']
    X_test = []
    for each_date in test_date:
        each_df_of_date = pd.read_csv(r'%s\..\input\test_feature_%s.txt' % (runningPath, each_date))
        print(getCurrentTime(), "dftest %s after extracting feature shape %s" % (each_date, each_df_of_date.shape))
        X_test.append(each_df_of_date)
    X_test = pd.concat(X_test, axis=0, ignore_index=True)
    test_instance_id = X_test['instance_id']
    del X_test['instance_id']
    print(getCurrentTime(), "X_test shape",  X_test.shape)

    train_date = ['2018-09-17', '2018-09-18', '2018-09-19', '2018-09-20', '2018-09-21', '2018-09-22', '2018-09-23']
    train_date = ['2018-09-17']
    X_train = []
    for each_date in train_date:
        each_df_of_date  = pd.read_csv(r'%s\..\input\train_feature_%s.txt' % (runningPath, each_date))
        print(getCurrentTime(), "df %s after extracting feature shape %s" % (each_date, each_df_of_date.shape))
        X_train.append(each_df_of_date)

    X_train = pd.concat(X_train, axis=0, ignore_index=True)
    train_label = X_train['is_trade']
    del X_train['is_trade']
    del X_train['instance_id']

    print(getCurrentTime(), "X_train shape", X_train.shape)

    X_verify = pd.read_csv(r'%s\..\input\train_feature_2018-09-24.txt' % (runningPath))
    verify_label = X_verify['is_trade']    
    del X_verify['is_trade']
    del X_verify['instance_id']

    params = {'max_depth': 4, 'colsample_bytree': 0.8, 'subsample': 0.8, 'eta': 0.02, 'silent': 1,
          'objective': 'binary:logistic','eval_metric ':'logloss', 'min_child_weight': 2.5,#'max_delta_step':10,'gamma':0.1,'scale_pos_weight':230/1,
           'seed': 10}  #

#     print(getCurrentTime(), "traiing LR...")
#     logiReg = LogisticRegression(C=1000, n_jobs=-1)
#     logiReg.fit(X_train, train_label)
#     lr_predict_proba = logiReg.predict_proba(X_verify)[:, 1]
#     lr_logloss = -np.sum(verify_label * np.log(lr_predict_proba) + (1 - verify_label) * np.log(1 - lr_predict_proba))/ lr_predict_proba.shape[0]
#     print(getCurrentTime(), "lr_logloss  %.6f" %(lr_logloss))

    round_number = 500
    print(getCurrentTime(), "traiing LGBM...")
    clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=-1)

    clf.fit(X_train, train_label.values)
    lgbm_predict_proba = pd.DataFrame(clf.predict_proba(X_verify)[:, 1], columns=['predicted_score'])
    lgb_logloss = -np.sum(verify_label * np.log(lgbm_predict_proba['predicted_score']) + 
                          (1 - verify_label) * np.log(1 - lgbm_predict_proba['predicted_score']))/ lgbm_predict_proba.shape[0] 
    print(getCurrentTime(), "lgb logloss %.6f" %(lgb_logloss))
    
    
    dftest = pd.read_csv(r'%s\..\input\test.txt' % (runningPath))
    lgbm_predict_proba = pd.DataFrame(clf.predict_proba(X_test)[:, 1], columns=['predicted_score'])
    lgbm_predict_proba['instance_id'] = test_instance_id
    lgbm_predict_proba = lgbm_predict_proba[['instance_id', 'predicted_score']]
    lgbm_predict_proba = order_instance_id_as_test(lgbm_predict_proba, dftest)
    


    print(getCurrentTime(), "training XGBoost...")
    xgb_mod = xgb.train(params, xgb.DMatrix(X_train, label=train_label), round_number)
    xgb_predict_proba = pd.DataFrame(xgb_mod.predict(xgb.DMatrix(X_verify)), columns=['predicted_score'])
    

    xgb_logloss = -np.sum(verify_label * np.log(xgb_predict_proba['predicted_score']) + 
                      (1 - verify_label) * np.log(1 - xgb_predict_proba['predicted_score']))/ lgbm_predict_proba.shape[0]
    
    print(getCurrentTime(), "xgb_logloss  %.6f" %(xgb_logloss))
    
    

#     print(getCurrentTime(), "training MLP...")
#     mlp = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(32,64), random_state=1, activation='tanh', solver='sgd', 
#                         learning_rate='adaptive', batch_size=128)
#     mlp_gs_param = {
#         'activation':['tanh'], 
#         'solver':['sgd'], 
#         'learning_rate':['adaptive'], 
#         'batch_size':[512]
#     }#         
#     mlp_gs = GridSearchCV(mlp, mlp_gs_param, n_jobs=-1, cv=5, refit=True)
#     mlp_gs.fit(dok_train, train_label['is_trade'].values)
#     print("mlp_gs.best_params_", mlp_gs.best_params_)
#     print("mlp_gs.best_score_", mlp_gs.best_score_)
#     mlp.fit(dok_train, train_label['is_trade'].values)
#     print("mlp.n_layers_", mlp.n_layers_)
#     print("mlp.n_iter_", mlp.n_iter_)
#     print("mlp.loss_", mlp.loss_)
#     print("mlp.out_activation_", mlp.out_activation_)

    print(getCurrentTime(), "predicting...")
    
    ensemble_predict_proba = (xgb_predict_proba['predicted_score'] + lgbm_predict_proba['predicted_score'])/2
    ensemble_logloss = -np.sum(verify_label * np.log(ensemble_predict_proba) + 
                      (1 - verify_label) * np.log(1 - ensemble_predict_proba))/ ensemble_predict_proba.shape[0]

    print(getCurrentTime(), "ensemble_logloss  %.6f" %(ensemble_logloss))


#     predict_proba = pd.DataFrame(mlp_gs.predict_proba(dok_verify)[:, 1], columns=['is_trade'])
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_verify)[:, 1], columns=['is_trade'])
#     mpl_logloss = -np.sum(verify_label['is_trade'] * np.log(predict_proba['is_trade']) + 
#                           (1 - verify_label['is_trade']) * np.log(1 - predict_proba['is_trade']))/ verify_label.shape[0] 
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_verify)[:, 1], columns=['is_trade'])
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_verify)[:, 1], columns=['is_trade'])
#     mpl_logloss = -np.sum(verify_label['is_trade'] * np.log(predict_proba['is_trade']) + 
#                           (1 - verify_label['is_trade']) * np.log(1 - predict_proba['is_trade']))/ verify_label.shape[0] 
# 
#     print(getCurrentTime(), "MLP logloss %.6f" %(mpl_logloss))


    # 确定参数后，  将 train 和 vrify 合并重新训练模型 
    X_train = pd.concat([X_train, X_verify], axis=0, ignore_index=True)
    train_label = pd.concat([train_label, verify_label], axis=0, ignore_index=True)
    
    dftest = pd.read_csv(r'%s\..\input\test.txt' % (runningPath))
    
    xgb_mod = xgb.train(params, xgb.DMatrix(X_train, label=train_label), round_number)
    xgb_predict_proba = pd.DataFrame(xgb_mod.predict(xgb.DMatrix(X_test)), columns=['predicted_score'])
    xgb_predict_proba['instance_id'] = test_instance_id
    xgb_predict_proba = xgb_predict_proba[['instance_id', 'predicted_score']]
    xgb_predict_proba.to_csv(r"%s\..\output\xgb_prediction.csv" % runningPath, index=False,  sep=' ', encoding='utf-8')
     
#     mlp = MLPClassifier(**mlp_gs.best_params_)
#     mlp.fit(dok_train, train_label['is_trade'].values)
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_test)[:, 1], columns=['predicted_score'])
#     predict_proba['instance_id'] = dftest['instance_id']
#     predict_proba = predict_proba[['instance_id', 'predicted_score']]
#     predict_proba.to_csv(r"%s\..\output\mlp_prediction.csv" % runningPath, index=False,  sep=' ', float_format='{:f}'.format, encoding='utf-8')

    clf.fit(X_train, train_label.values)
    lgbm_predict_proba = pd.DataFrame(clf.predict_proba(X_test)[:, 1], columns=['predicted_score'])
    lgbm_predict_proba['instance_id'] = test_instance_id
    lgbm_predict_proba = lgbm_predict_proba[['instance_id', 'predicted_score']]
    lgbm_predict_proba.to_csv(r"%s\..\output\lgb_prediction.csv" % runningPath, index=False,  sep=' ', encoding='utf-8')
    
    ensemble_predict_proba = pd.DataFrame()
    ensemble_predict_proba['instance_id'] = xgb_predict_proba['instance_id'] 
    ensemble_predict_proba['predicted_score'] = (xgb_predict_proba['predicted_score'] + lgbm_predict_proba['predicted_score']) / 2
    ensemble_predict_proba = order_instance_id_as_test(ensemble_predict_proba, dftest)
    ensemble_predict_proba.to_csv(r"%s\..\output\ensemble_prediction.csv" % runningPath, index=False,  sep=' ', encoding='utf-8')
    
    return 

if (__name__ == '__main__'):
    main()

    