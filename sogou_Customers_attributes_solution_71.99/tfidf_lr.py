'''tfidf-lr stack for education/age/gender'''
#10W训练集分成8W和2W,8W用于第一层模型交叉验证，2W用于第二层模型交叉验证
import pandas as pd
import numpy as np
import jieba
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from datetime import datetime
import cfg

#helper function: 
def myAcc(y_true,y_pred):
    y_pred = np.argmax(y_pred,axis=1)
    return np.mean(y_true == y_pred)

#用已经填充好缺失值的数据
df_all = pd.read_csv(cfg.data_path + 'all_v2.csv',encoding='utf8',nrows=100000)
ys = {}
for label in ['Education','age','gender']:
    ys[label] = np.array(df_all[label])

#提取tfidf特征    
class Tokenizer():
    def __init__(self):
        self.n = 0
    def __call__(self,line):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1,2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i+gram])]
        if np.random.rand() < 0.00002:
            print(line)
            print('分割线'*20)
            print(tokens)
        self.n += 1
        if self.n%10000==0:
            print(self.n)
        return tokens
#min_df = 3
tfv = TfidfVectorizer(tokenizer=Tokenizer(),min_df=3,max_df=0.95,sublinear_tf=True)
X_sp = tfv.fit_transform(df_all['query'])
pickle.dump(X_sp,open(cfg.data_path + 'tfidf_10W.feat','wb'))
#造20行的空dataframe用来存储预测结果
df_stack = pd.DataFrame(index=range(len(df_all)))

#-----------------------stack for education/age/gender------------------
for lb in ['Education','age','gender']:
    print(lb)
    TR = 80000
    #num_class: y的取值类型
    num_class = len(pd.value_counts(ys[lb]))
    n = 5

    #前8w条
    X = X_sp[:TR]
    y = ys[lb][:TR]
    #后2W条
    X_te = X_sp[TR:]
    y_te = ys[lb][TR:]
    
    #存放train的结果
    stack = np.zeros((X.shape[0],num_class))
    #存放test的结果
    stack_te = np.zeros((X_te.shape[0],num_class))

    #利用之前切分好的前8W进行5折CV
    for i,(tr,va) in enumerate(KFold(len(y),n_folds=5)):
        print('%s stack:%d  /%d'%(str(datetime.now()),i+1,n))
        #l2: C=3
        clf = LogisticRegression(penalty='l2',C=3.5,n_jobs=4)
        clf.fit(X[tr],y[tr])
        y_pred_va = clf.predict_proba(X[va])
        y_pred_te = clf.predict_proba(X_te)
        print('validation set acc:',myAcc(y[va],y_pred_va))
        print('test set acc:',myAcc(y_te,y_pred_te))
        #validation set上的概率输出
        stack[va] += y_pred_va
        #test set 上的概率输出的累加
        stack_te += y_pred_te
    #取5折平均概率     
    stack_te /= n
    #合并8W和2W
    stack_all = np.vstack([stack,stack_te])
    #合并分别对三个类型进行的预测
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_{}_{}'.format(lb,i)] = stack_all[:,i]

        
#输出基于tfidf特征+LR的
df_stack.to_csv(cfg.data_path + 'tfidf_stack_10W.csv',index=None,encoding='utf8')
print(datetime.now(),'save tfidf stack done!')