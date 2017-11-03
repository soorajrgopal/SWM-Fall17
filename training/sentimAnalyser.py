
from naiveBayesClassifier import SentiNaiveBayesClassifier
from randomForestClassifier import RandomForestClassifer
from rfngram import RandomForestnGramClassifer
from svm_w2vec import SVM_Word2Vec
from lstm import RNTNLSTMClassifier
from svm import SVM


"""
Class that generate the sentiment of the given sentence.

Initializing the class will train the classifier with the 
training data.

Then you can use the method getSEntiments(message) to get the sentiment 
value.

"""
class SentimentalAnalyzer:

	def __init__(self):
		

		self.classifier = RandomForestClassifer()
		# self.classifier = SentiNaiveBayesClassifier()
		# self.classifier = RandomForestClassifer()
		# self.classifier = SentiNaiveBayesClassifier()

		self.classifier.train()
		self.classifier.validateClassifier()
	
	# Return 1 if happy 0 if sad
	def getSentiments(self, message):
		return self.classifier.classify(message)

obj = SentimentalAnalyzer() 
messages = ['There was a time in life when I was walking alone a road and found no value in life, but now I feel so much better and happy.',
'There was a time in life when I was walking alone a road and found no value in life.',
'I am feeling so relived',
'I am feeling so awesome', 
'I am so worried',
'a hot girl outside',
'its very hot outside',
'This movie was actually neither that funny, nor super witty.']


for status in messages:
	print 'Predicted: ', 'HAPPY'.ljust(10) if obj.getSentiments(status) == '1' else 'SAD'.ljust(10), status.rjust(0)
