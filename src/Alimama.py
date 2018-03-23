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

def main():
    print(getCurrentTime(), "running...")
    dok_train = np.load(r"%s\..\input\sparse_train.npy" % runningPath)[()]# train 中包括 '2018-09-17'-- '2018-09-23' 的数据
    dok_verify = np.load(r"%s\..\input\sparse_verify.npy" % runningPath)[()] # verify 中只包含了 '2018-09-24' 的数据
    dok_test = np.load(r"%s\..\input\sparse_test.npy" % runningPath)[()]

    train_label = pd.read_csv(r'%s\..\input\train_label.txt' % runningPath, header=None)
    train_label.columns = ['label']
    verify_label = pd.read_csv(r'%s\..\input\verify_label.txt' % runningPath, header=None)
    verify_label.columns = ['label']
    
    dftest = pd.read_csv(r'%s\..\input\test.txt' % runningPath)
    
    print(getCurrentTime(), "loading finished...")
    
    forecast_date = '2018-09-24'
    
    params = {'max_depth': 4, 'colsample_bytree': 0.8, 'subsample': 0.8, 'eta': 0.02, 'silent': 1,
          'objective': 'binary:logistic','eval_metric ':'logloss', 'min_child_weight': 2.5,#'max_delta_step':10,'gamma':0.1,'scale_pos_weight':230/1,
           'seed': 10}  #

    round_number = 500

    pos_idx = train_label[train_label['label'] == 1].index
    neg_idx = train_label.sample(len(pos_idx) * 10)    
    
    training_samples_idx = pos_idx.append(neg_idx.index)
    
    print(getCurrentTime(), "training XGBoost...")
    xgb_mod = xgb.train(params, xgb.DMatrix(dok_train[training_samples_idx], label=train_label.iloc[training_samples_idx]), round_number)
 
    print(getCurrentTime(), "traiing LGBM...")
    clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=-1)
    
    clf.fit(dok_train[training_samples_idx], train_label.iloc[training_samples_idx]['label'].values)
    lgbm_predict_proba = pd.DataFrame(clf.predict_proba(dok_verify)[:, 1], columns=['label'])
    lgb_logloss = -np.sum(verify_label['label'] * np.log(lgbm_predict_proba['label']) + 
                          (1 - verify_label['label']) * np.log(1 - lgbm_predict_proba['label']))/ verify_label.shape[0] 
    print(getCurrentTime(), "lgb logloss %.6f" %(lgb_logloss))

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
#     mlp_gs.fit(dok_train, train_label['label'].values)
#     print("mlp_gs.best_params_", mlp_gs.best_params_)
#     print("mlp_gs.best_score_", mlp_gs.best_score_)
 
#     mlp.fit(dok_train, train_label['label'].values)
#     print("mlp.n_layers_", mlp.n_layers_)
#     print("mlp.n_iter_", mlp.n_iter_)
#     print("mlp.loss_", mlp.loss_)
#     print("mlp.out_activation_", mlp.out_activation_)

    print(getCurrentTime(), "predicting...")
    xgb_predict_proba = pd.DataFrame(xgb_mod.predict(xgb.DMatrix(dok_verify)), columns=['label'])
 
    xgb_logloss = -np.sum(verify_label['label'] * np.log(xgb_predict_proba['label']) + 
                      (1 - verify_label['label']) * np.log(1 - xgb_predict_proba['label']))/ verify_label.shape[0]
    
    print(getCurrentTime(), "xgb_logloss  %.6f" %(xgb_logloss))
    
    ensemble_predict_proba = (xgb_predict_proba['label'] + lgbm_predict_proba['label'])/2
    ensemble_logloss = -np.sum(verify_label['label'] * np.log(ensemble_predict_proba) + 
                      (1 - verify_label['label']) * np.log(1 - ensemble_predict_proba))/ verify_label.shape[0]

    print(getCurrentTime(), "ensemble_logloss  %.6f" %(ensemble_logloss))


#     predict_proba = pd.DataFrame(mlp_gs.predict_proba(dok_verify)[:, 1], columns=['label'])
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_verify)[:, 1], columns=['label'])
#     mpl_logloss = -np.sum(verify_label['label'] * np.log(predict_proba['label']) + 
#                           (1 - verify_label['label']) * np.log(1 - predict_proba['label']))/ verify_label.shape[0] 
#     

#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_verify)[:, 1], columns=['label'])
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_verify)[:, 1], columns=['label'])
#     mpl_logloss = -np.sum(verify_label['label'] * np.log(predict_proba['label']) + 
#                           (1 - verify_label['label']) * np.log(1 - predict_proba['label']))/ verify_label.shape[0] 
# 
#     print(getCurrentTime(), "MLP logloss %.6f" %(mpl_logloss))


    # 确定参数后，  将 train 和 vrify 合并重新训练模型, sparse_train_total 包含了 train 和 vrify 
    dok_train = np.load(r"%s\..\input\sparse_train_total.npy" % runningPath)[()]
    train_label = pd.read_csv(r'%s\..\input\train_label_total.txt' % runningPath, header=None)
    train_label.columns = ['label']
    print(dok_train.shape, train_label.shape)
    
    xgb_mod = xgb.train(params, xgb.DMatrix(dok_train, label=train_label), round_number)
    xgb_predict_proba = pd.DataFrame(xgb_mod.predict(xgb.DMatrix(dok_test)), columns=['predicted_score'])
    xgb_predict_proba['instance_id'] = dftest['instance_id']
    xgb_predict_proba = xgb_predict_proba[['instance_id', 'predicted_score']]
    xgb_predict_proba.to_csv(r"%s\..\output\xgb_prediction.csv" % runningPath, index=False,  sep=' ')
     
#     mlp = MLPClassifier(**mlp_gs.best_params_)
#     mlp.fit(dok_train, train_label['label'].values)
#     predict_proba = pd.DataFrame(mlp.predict_proba(dok_test)[:, 1], columns=['predicted_score'])
#     predict_proba['instance_id'] = dftest['instance_id']
#     predict_proba = predict_proba[['instance_id', 'predicted_score']]
#     predict_proba.to_csv(r"%s\..\output\mlp_prediction.csv" % runningPath, index=False,  sep=' ')

    clf.fit(dok_train, train_label['label'].values)
    lgbm_predict_proba = pd.DataFrame(clf.predict_proba(dok_test)[:, 1], columns=['predicted_score'])
    lgbm_predict_proba['instance_id'] = dftest['instance_id']
    lgbm_predict_proba = lgbm_predict_proba[['instance_id', 'predicted_score']]
    lgbm_predict_proba.to_csv(r"%s\..\output\lgb_prediction.csv" % runningPath, index=False,  sep=' ')
    
    ensemble_predict_proba = pd.DataFrame()
    ensemble_predict_proba['instance_id'] = xgb_predict_proba['instance_id'] 
    ensemble_predict_proba['predicted_score'] = (xgb_predict_proba['predicted_score'] + lgbm_predict_proba['predicted_score']) / 2

    ensemble_predict_proba.to_csv(r"%s\..\output\ensemble_prediction.csv" % runningPath, index=False,  sep=' ')
    return 

if (__name__ == '__main__'):
    main()

    