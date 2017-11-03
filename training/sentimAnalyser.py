
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

Then you can use the method getSEntiments(message) to get the predicted rating.

"""
class SentimentalAnalyzer:

	def __init__(self):
		

		self.classifier = RandomForestClassifer()
		# self.classifier = SentiNaiveBayesClassifier()
		# self.classifier = RandomForestClassifer()
		# self.classifier = SentiNaiveBayesClassifier()

		self.classifier.train()
		self.classifier.validateClassifier()
	
	def getSentiments(self, message):
		return self.classifier.classify(message)

obj = SentimentalAnalyzer() 
messages = ['An excellent and riveting story from start to finish.',
            'Cruiser just can''t act. Story line works, but not with him.',
            'Based on true events and action packed! Go see it!',
            'This film could have been better, it had a great story, a good cast, but the telling of the story was done awkwardly, they tried to make a villain a hero, and failed by making him too goofy, and straying from a true story that would have been a great film on its own',
            'Roller coaster ride. But can Cruise play anyone new?',
            'predictable crap, just wish he got more of his teeth knocked out']


for status in messages:
    print obj.getSentiments(status)

