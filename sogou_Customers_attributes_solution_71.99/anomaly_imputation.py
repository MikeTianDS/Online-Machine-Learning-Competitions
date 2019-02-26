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
        if np.random.rand() < 0.00001:
            print(line)
            print('！！分割线'*10)
            print(tokens)
        self.n += 1
        if self.n%10000==0:
            print(self.n,end=' ')
        return tokens    

tfv = TfidfVectorizer(tokenizer=Tokenizer(),min_df=3,max_df=0.95,sublinear_tf=True)
X_sp = tfv.fit_transform(df_all['query'])
# pickle.dump(X_sp,open(root + 'tfidf_10W.pkl','wb'))
print(len(tfv.vocabulary_))
X_all = X_sp



#--------------缺失值处理----------------------------------
#因为给的数据中每一列都有一些 -1 -- 代表 未知，但我们去做预测的时候显然不能输出未知作为预测结果，所以处理这些未知实在必行！
#用LR预测缺失值的取值，可能带来一定的偏差，但似乎没有更好的办法
# for lb,idx in [('Education',0),('age',2),('gender',3)]:
#     tr = np.where(df_all[lb]!=-1)[0]
#     va = np.where(df_all[lb]==-1)[0]

#education特征中缺失值数量：9280
lb = 'Education'
idx = 0
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
#用所有education不缺失的行的基于tfidf的特征 与 他们的education标签拟合LR，去预测缺失值的education标签
df_all.iloc[va,idx] = LogisticRegression(C=1).fit(X_all[tr],df_all.iloc[tr,idx]).predict(X_all[va])

#age特征中，缺失值： 1666
#做法同理
lb = 'age'
idx = 2
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
df_all.iloc[va,idx] = LogisticRegression(C=2).fit(X_all[tr],df_all.iloc[tr,idx]).predict(X_all[va])

#gender特征中，缺失值：2155
#做法同理
lb = 'gender'
idx = 3
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
df_all.iloc[va,idx] = LogisticRegression(C=2).fit(X_all[tr],df_all.iloc[tr,idx]).predict(X_all[va])

#输出处理好未知值之后的文件
df_all = pd.concat([df_all,df_te]).fillna(0)
df_all.to_csv(cfg.data_path + 'all_v2.csv',index=None)