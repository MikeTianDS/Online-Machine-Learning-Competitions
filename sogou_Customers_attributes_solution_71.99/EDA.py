'''1.concat train data and test data 
   2.use lr to fill null label'''  #待验证：这样做真的好嘛？
import cfg
import pandas as pd
import numpy as np
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import pickle

#---load data-------
#pandas读不了
df_tr = []
for i,line in enumerate(open('C:/Data_Competition/Sogou_User_Profiles_Prediction/data/' + 'user_tag_query.10W.TRAIN',encoding='GB18030')):
    segs = line.split('\t')
    row = {}
    row['Id'] = segs[0]
    row['age'] = int(segs[1])
    row['gender'] = int(segs[2])
    row['Education'] = int(segs[3])
    row['query'] = '\t'.join(segs[4:])
    df_tr.append(row)
df_tr = pd.DataFrame(df_tr)

df_te=[]
for i,line in enumerate(open('C:/Data_Competition/Sogou_User_Profiles_Prediction/data/' + 'user_tag_query.10W.TEST',encoding='GB18030')):
    segs = line.split('\t')
    row = {}
    row['Id'] = segs[0]
    row['query'] = '\t'.join(segs[1:])
    df_te.append(row)
df_te = pd.DataFrame(df_te)

print(df_tr.shape)# 100000条 * 5列（id，age，gender，education，query）
print(df_te.shape)# 100000条 * 2列（id，query）

df_all = df_tr
# df_all = pd.concat([df_tr,df_te]).fillna(1)
# df_all.index = range(len(df_all))

for lb in ['Education','age','gender']:
    df_all[lb] = df_all[lb] - 1
    print(df_all.iloc[:100000][lb].value_counts())
    
    
#  4    37107 初中
#  3    28148 高中
#  2    18858 大学生
# -1     9280 未知
#  5     5693 小学
#  1      560 硕士
#  0      354 博士
# Name: Education, dtype: int64
#  0    38996 0-18
#  1    26744 19-23
#  2    18529 24-30
#  3    10654 31-40
#  4     2922  41-50
# -1     1666 未知
#  5      489 51-999岁
# Name: age, dtype: int64
#  0    56976 男
#  1    40869 女
# -1     2155 未知
# Name: gender, dtype: int64
    

'''
duplication ID check
'''
print(sum(df_all['Id'].duplicated( keep=False)))
print(sum(df_te['Id'].duplicated( keep=False)))
# No duplicated ID


# #基于先验知识的’异常值‘检测
# print(sum((df_all['Education'] == 1) & (df_all['age'] == 0)))# 0-18岁的硕士
# print(sum((df_tr['age'] == 0) & (df_tr['Education'] ==0))) #0-18岁的博士
# a = np.where((df_all['Education'] == 1) & (df_all['age'] == 0))
# b = [   37, 22436, 23775, 25795, 27190, 49324, 80317, 86666]
# df_tr = df_tr.drop(b)


