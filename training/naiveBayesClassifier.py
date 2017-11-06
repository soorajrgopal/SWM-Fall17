
from extractFeatures import ExtractFeatures
import nltk.classify.util
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import pickle

class SentiNaiveBayesClassifier:

	def __init__(self):
		self.classifier = None

	'''
		Train a naivebayes classifer with the training data.
	'''
	def train(self):
		
		# Get data from the extracter
		extracter = ExtractFeatures('./data/reviewsData.txt')
		tokenizedData = extracter.getTokenizedData()

		trainingData = tokenizedData['train']
		

		print ('')
		print ('Training Naive Bayes Classifier')
		print ('Training data size = ', len(trainingData))
		print ('')

		modifiedTrainingData = [(self.word_feats(item[0]), item[1]) for item in trainingData]
		self.classifier = NaiveBayesClassifier.train(modifiedTrainingData)
		print ('Training Naive Bayes Classifier Completed')

	def validateClassifier(self):

		extracter = ExtractFeatures('./data/reviewsData.txt')
		tokenizedData = extracter.getTokenizedData()
		testData = tokenizedData['test']

		print ('')
		print ('Validating Naive Bayes Classifier')
		print ('Test data size = ', len(testData))
		print ('')

		modifiedTestData = [(self.word_feats(item[0]), item[1]) for item in testData]
		print ('Accuracy: ', nltk.classify.util.accuracy(self.classifier, modifiedTestData))

	def word_feats(self, words):
		return dict([(word, True) for word in words])


	def classify(self, statusMessage):
		tokenizedwords = word_tokenize(statusMessage)
		return self.classifier.classify(self.word_feats(tokenizedwords))

if __name__ == '__main__':
    trainObj = SentiNaiveBayesClassifier()
    trainObj.train()
    trainObj.validateClassifier()
    f = open('./model/naiveBayesClassifierModel.pickle','wb')
    # Dumps the trained model as a pickle (serializes model object)
    pickle.dump(trainObj, f)
    f.close()
    # print (trainObj.classify('What probably is a fine true story is given a Disney/Hallmark style treatment with a dose of US jingoism thrown in for bad measure. No thanks.'))
    # print (trainObj.classify('It''s just popular because it has a good message. Very cheesy.'))
    # print (trainObj.classify('Raw features excellent performances, squirming violence, and plenty of bizarre energy and indie filmmaking to make a strong lasting impression'))
    # print (trainObj.classify('This is a bad movie.'))
    # print (trainObj.classify('This is not a bad movie.'))
