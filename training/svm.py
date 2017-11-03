from extractFeatures import ExtractFeatures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm

'''
	Uses bag of words and valence value as features
	to train a random forest.
'''
class SVM:


	def __init__(self):
		self.valenceDict = {}
		self.classifier = None
		self.countVectorizer = None

	'''
	Takes the training data and counts the occurance of a word.
	Valence score is also added as a feature.
	This is feed into a Random Forest classifier.
	'''
	def train(self):
		
		extracter = ExtractFeatures('./data/reviewsData.txt')
		self.valenceDict = extracter.getValenceDict()
		
		tokenizedData = extracter.getTokenizedData()
		trainingData = tokenizedData['train']

		print ''
		print 'Training SVM Classifier'
		print 'Training data size = ', len(trainingData)
		print ''

		train_wordList = []
		train_labels = []
		valence_values_training = []

		# Convert the word list into a sentence space separated.
		for item in trainingData:
			train_wordList.append(" ".join( item[0] ))
			train_labels.append(item[1])

			# Calculate the valence for this sentence.
			valSum = 0
			for word in item[0]:
				valSum += self.valenceDict.get(word, 0)
			
			valence_values_training.append(valSum)

		# Count the occurance of words in the sentenes.
		self.countVectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
		countFeatures = self.countVectorizer.fit_transform(train_wordList)
		featuresArray = countFeatures.toarray()

		# Add valence score as a last feature.
		featuresArray_Valence = np.c_[featuresArray,valence_values_training]
		
		# Train the classifier
		self.classifier = svm.SVC()
		self.classifier.fit(featuresArray_Valence, train_labels)
		#self.classifier = RandomForestClassifier(n_estimators = 50) 
		#self.classifier = self.classifier.fit(featuresArray_Valence, train_labels)
		print 'Training SVM Classifier Completed'
		

	def validateClassifier(self):

		extracter = ExtractFeatures('./data/reviewsData.txt')
		tokenizedData = extracter.getTokenizedData()
		testData = tokenizedData['test']

		print ''
		print 'Validating SVM Classifier'
		print 'Test data size = ', len(testData)
		print ''

		test_wordList = []
		test_labels = []
		valence_values_test = []

		# Convert the word list into a sentence space separated.
		for item in testData:
			test_wordList.append(" ".join( item[0] ))
			test_labels.append(item[1])
			
			# Calculate the valence for this sentence.
			valSum = 0
			for word in item[0]:
				valSum += self.valenceDict.get(word, 0)
			
			valence_values_test.append(valSum)

		# Count the occurance of words in the sentenes using the vectorizer.
		countTestFeatures = self.countVectorizer.transform(test_wordList)
		testF = np.c_[countTestFeatures.toarray(), valence_values_test]
		predictedVal = self.classifier.predict(testF)
		print (testF)		

		score = accuracy_score(test_labels,predictedVal)
		print 'Accuracy: ', score

	def classify(self, statusMessage):

		# Calculate the valence for this sentence.
		valSum = 0
		for word in statusMessage.split():
			valSum += self.valenceDict.get(word, 0)

		countFeatures = self.countVectorizer.transform([statusMessage])
		featuresArray = countFeatures.toarray()
		featuresArray_Valence = np.c_[featuresArray,[valSum]]
		val = self.classifier.predict(featuresArray_Valence)
		return val[0]

if __name__ == '__main__':
    trainObj = SVM()
    trainObj.train()
    trainObj.validateClassifier()
    print trainObj.classify('What probably is a fine true story is given a Disney/Hallmark style treatment with a dose of US jingoism thrown in for bad measure. No thanks.')
    print trainObj.classify('It''s just popular because it has a good message. Very cheesy.')


