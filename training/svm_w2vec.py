from extractFeatures import ExtractFeatures
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

'''
	Uses word2vec and svm to classify.
'''
class SVM_Word2Vec:


	def __init__(self):
		self.classifier = None
		self.extracter = ExtractFeatures('./data/reviewsData.txt')


	'''
	
	'''
	def train(self):

		h = .02  # step size in the mesh
		C = 1.0  # SVM regularization parameter
		
		tokenizedData = self.extracter.getTokenizedData()
		trainingData = tokenizedData['train']

		print ('')
		print ('Training SVM with Word2vec  Classifier')
		print ('Training data size = ', len(trainingData))
		print ('')

		train_wordList = []
		train_labels = []
		

		# Convert the word list into a sentence space separated.
		for item in trainingData:
			train_wordList.append(item[0])


		self.extracter.generateWord2Vec(train_wordList)
		
		# Convert the sentences to vec
		train_data_svm = []
		for item in trainingData:
			train_labels.append(item[1])
			space_sep_sent = " ".join( item[0] )
			train_data_svm.append(self.extracter.generateAvgWord2Vec(space_sep_sent))
		
		self.classifier = SVC(kernel='poly', degree=2, C=C)

		self.classifier.fit(train_data_svm, train_labels)
		
		print ('Training SVM with Word 2 vec  Classifier Completed')
		

	def validateClassifier(self):

		testData = self.extracter.finalData['test']
		print ('')
		print ('Validating SVM with Word 2 vec  Classifier')
		print ('Test data size = ', len(testData))
		print ('')
		test_labels = []
		test_data_svm = []
		for item in testData:

			test_labels.append(item[1])
			space_sep_sent = " ".join( item[0] )
			test_data_svm.append(self.extracter.generateAvgWord2Vec(space_sep_sent))
		
		predictedVal = self.classifier.predict(test_data_svm)
		

		print (predictedVal.shape)
		score = accuracy_score(test_labels,predictedVal)
		print ('Accuracy: ', score)

	def classify(self, statusMessage):
		features = self.extracter.generateAvgWord2Vec(statusMessage)
		predictedVal = self.classifier.predict([features])
		print (predictedVal)
		return predictedVal[0]

if __name__ == '__main__':
    trainObj = SVM_Word2Vec()
    trainObj.train()
    trainObj.validateClassifier()
    print (trainObj.classify('I didn''t care much for this one. Most of the special effects were good, but "Kong" moved too much like a human rather than a gorilla. The original King Kong was much more believable.'))
    print (trainObj.classify('Kong Skull Island: Bursting with special effects and soaring eye candy, one that benfits from Smaul L Jacksons preformance. Even tho it has one to many side characters and inaccurate pacing. But Kong Skulk Island is sure to please.'))

