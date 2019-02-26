#-- lsi model function
def lsi(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(len(texts))
	dictionary = corpora.Dictionary(texts)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get corpus..'
	corpusD = [dictionary.doc2bow(text) for text in texts]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'tfidf Model...'
	tfidf = TfidfModel(corpusD)
	corpus_tfidf = tfidf[corpusD]

	model = LsiModel(corpusD, num_topics=topicNum, chunksize=8000, extra_samples = 100)#, distributed=True)#, sample = 1e-5, iter = 10,seed = 1)

	lsiFeature = np.zeros((len(texts), topicNum))
	print 'translate...'
	i = 0

	for doc in corpusD:
		topic = model[doc]
		
		for t in topic:
			 lsiFeature[i, t[0]] = round(t[1],5)
		i = i + 1
		if i%1000 == 1:
			print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(i)

	return lsiFeature

#-- get lsi feature
def getLsiFeature(dim):
	train = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ",")
	
	train = train[['uid','qlist']]
	test = test[['uid','qlist']]
	data = pd.concat([train, test], axis = 0)
	lsiFeature = lsi(data['qlist'].values, dim)
	colName = getColName(dim, "qlsi")
	lsiFeature = pd.DataFrame(lsiFeature, columns = colName)	
	lsiFeature['uid'] = data['uid'].values.T
	print lsiFeature.shape
	name = '../feature/lsi/lsiFeature'+str(dim)+'.csv'
	lsiFeature.to_csv(name, index = False)	