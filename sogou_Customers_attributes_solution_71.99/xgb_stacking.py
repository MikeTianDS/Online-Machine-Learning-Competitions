#第二层模型 stacking with XGB
#数据准备
import pandas as pd
import numpy as np
import xgboost as xgb
import cfg
import datetime

def xgb_acc_score(preds,dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds,axis=1)
    return [('acc',np.mean(y_true == y_pred))]

#tfidf+lr的预测结果
df_lr = pd.read_csv(cfg.data_path + 'tfidf_stack_10W.csv')
#dm+nn的预测结果
df_dm = pd.read_csv(cfg.data_path + 'dmd2v_stack_10W.csv')
#dbow+nn的预测结果
df_dbow = pd.read_csv(cfg.data_path + 'dbowd2v_stack_10W.csv')

df_lb = pd.read_csv(cfg.data_path + 'all_v2.csv',usecols=['Id','Education','age','gender'],nrows=100000)
ys = {}
for lb in ['Education','age','gender']:
    #每一种分类任务对应的标签存为字典
    ys[lb] = np.array(df_lb[lb])

'''最好的参数组合'''
#-------------------------education----------------------------------
TR = 80000
df_sub = pd.DataFrame()
df_sub['Id'] = df_lb.iloc[TR:]['Id']
seed = 10
lb = 'Education'
print(lb)

esr = 25
evals = 1
n_trees = 1000

#三个一层模型合并作为输入
df = pd.concat([df_lr,df_dbow,df_dm],axis=1)
print(df.columns)
num_class = len(pd.value_counts(ys[lb]))
X = df.iloc[:TR]
y = ys[lb][:TR]
X_te = df.iloc[TR:]
y_te = ys[lb][TR:]

ss = 0.9
mc = 2
md = 8
gm = 2
# n_trees = 10

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
#     "eval_metric": "merror",
    "num_class":num_class,
    'max_depth':md,
    'min_child_weight':mc,
    'subsample':ss,
    'colsample_bytree':0.8,
    'gamma':gm,
    "eta": 0.02,
    "lambda":0,
    'alpha':0,
    "silent": 1,
#     'seed':seed,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist,feval=xgb_acc_score,maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['Education'] = np.argmax(bst.predict(dvalid),axis=1) + 1


# #------------------------ age-----------------------------------
# lb = 'age'
# print(lb)
# num_class = len(pd.value_counts(ys[lb]))

# # df = pd.concat([df_stack_tfidf,df_stack_d2v],axis=1)
# num_class = len(pd.value_counts(ys[lb]))
# X = df.iloc[:TR]
# y = ys[lb][:TR]
# X_te = df.iloc[TR:]
# y_te = ys[lb][TR:]
# modellr = LogisticRegression(penalty = 'l1',C=100,n_jobs=5)
# modellr.fit(X,y)
# modelsvm = SVC()
# modelsvm.fit(X,y)



# a = modellr.predict(X_te)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(a,y_te))
# # b = modelsvm.predict(X_te)
# # from sklearn.metrics import accuracy_score
# # print(accuracy_score(b,y_te))
# sum(modellr.coef_)


#------------------------ age-----------------------------------
lb = 'age'
print(lb)
num_class = len(pd.value_counts(ys[lb]))

# df = pd.concat([df_stack_tfidf,df_stack_d2v],axis=1)
num_class = len(pd.value_counts(ys[lb]))
X = df.iloc[:TR]
y = ys[lb][:TR]
X_te = df.iloc[TR:]
y_te = ys[lb][TR:]

ss = 0.5
mc = 3
md = 7
gm = 2
# n_trees = 37

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
#     "eval_metric": "merror",
    "num_class":num_class,
    'max_depth':md,
    'min_child_weight':mc,
    'subsample':ss,
    'colsample_bytree':1,
    'gamma':gm,
    "eta": 0.005,
    "lambda":0,
    'alpha':0,
    "silent": 1,
#     'seed':seed,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist,feval=xgb_acc_score,maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['age'] = np.argmax(bst.predict(dvalid),axis=1)+1


#--------------------------gender-------------------------------------
lb = 'gender'
print(lb)
num_class = len(pd.value_counts(ys[lb]))

# df = pd.concat([df_lr,df_multid2v],axis=1)
num_class = len(pd.value_counts(ys[lb]))
X = df.iloc[:TR]
y = ys[lb][:TR]
X_te = df.iloc[TR:]
y_te = ys[lb][TR:]


ss = 0.5
mc = 0.8
md = 7
gm = 1
# n_trees = 25

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
#     "eval_metric": "merror",
    "num_class":num_class,
    'max_depth':md,
    'min_child_weight':mc,
    'subsample':ss,
    'colsample_bytree':1,
    'gamma':gm,
    "eta": 0.01,
    "lambda":0,
    'alpha':0,
    "silent": 1,
#     'seed':seed,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist,feval=xgb_acc_score,maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['gender'] = np.argmax(bst.predict(dvalid),axis=1)+1

df_sub = df_sub[['Id','age','gender','Education']]
df_sub.to_csv(cfg.data_path + 'tfidf_dm_dbow_2W.csv',index=None,header=None,sep=' ')