#-- lda model function
def lda(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(len(texts))
	dictionary = corpora.Dictionary(texts)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get corpus..'
	corpusD = [dictionary.doc2bow(text) for text in texts]

	#id2word = dictionary.id2word
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'tfidf Model...'
	tfidf = TfidfModel(corpusD)
	corpus_tfidf = tfidf[corpusD]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'train lda Model...'
	ldaModel = gensim.models.ldamulticore.LdaMulticore(corpus_tfidf, workers = 8, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
	#ldaModel = gensim.models.ldamodel.LdaModel(corpus=corpusD, num_topics=topicNum, update_every=1, chunksize=8000, passes=10)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get lda feature...'
	ldaFeature = np.zeros((len(texts), topicNum))
	i = 0

	for doc in corpus_tfidf:
		topic = ldaModel.get_document_topics(doc, minimum_probability = 0.01)
		
		for t in topic:
			 ldaFeature[i, t[0]] = round(t[1],5)
		i = i + 1
		if i%1000 == 1:
			print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(i)

	return ldaFeature

#-- get lda feature function
def getLdaFeature(dim):
	#---user Lda info--
	train = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ",")

	train = train[['uid','qlist']]
	test = test[['uid','qlist']]
	data = pd.concat([train, test], axis = 0)
	ldaFeature = lda(data['qlist'].values, dim)
	colName = getColName(dim, "qlda")
	ldaFeature = pd.DataFrame(ldaFeature, columns = colName)	
	ldaFeature['uid'] = data['uid'].values.T
	print ldaFeature.shape
	name = '../feature/lda/ldaFeature'+str(dim)+'.csv'
	ldaFeature.to_csv(name, index = False)