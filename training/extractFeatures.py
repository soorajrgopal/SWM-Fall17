
import nltk
import re
from nltk.corpus import stopwords
from random import shuffle
from nltk.tokenize import word_tokenize

import numpy as np
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors



"""
Extracts features from the given file containing sentences and tags.

getTokenizedData returns a list of training data. Each item in the list contains
the list of words in that sentence and the tag.
[
	[['Item', 'Does', 'Not', 'Match', 'Picture', '.'], '0'],
	[['Great', 'brunch', 'spot', '.'], '1'],
	 ....

]
"""

class ExtractFeatures:

	FEATURE_SIZE_WORD2VEC = 300

	def __init__(self, filePath):
		self.filePath = filePath
		self.word2vecModel = None
		self.finalData = {}

	def getTokenizedData(self):
		
		# Read the file
		print('')
		print ("Extracting training data ")
		stops = set(stopwords.words("english"))
		result = []
		for line in open(self.filePath):
			
			sentence, tag = line.split('\t')

			# Remove all non letters from the sentence
			# We need to rethink this logic since smileys and
			# other emoticons will be usefull in sentimental 
			# analysis.
			sentence = re.sub("[^a-zA-Z]", " ", sentence) 

			# Convert the sentence into list of words.
			tokenizedlist = word_tokenize(sentence)

			# Remove all stop words
			tokenizedlist = [word for word in tokenizedlist if word not in stops]

			# Store the tokenized sentence and its tag to a list.
			item = [tokenizedlist, tag.rstrip()]
			result.append(item)

		print ("Extracted tokenized words from training data - " + self.filePath)
		shuffle(result)
		
		finalData = {}
		count = len(result)

		percentageOfTrainingData = 0.8

		trainDataIndex = int(count * percentageOfTrainingData)

		finalData['train'] = result[:trainDataIndex]
		finalData['test'] = result[trainDataIndex:]

		self.finalData = finalData
		return finalData

	'''
	Train the word2vec model with a list of list of words.
	'''
	def generateWord2Vec(self, tokensizedSentenceList):
		
		# self.word2vecModel = word2vec.Word2Vec(tokensizedSentenceList, \
		# 	workers = 2, \
		# 	size = ExtractFeatures.FEATURE_SIZE_WORD2VEC, \
		# 	min_count = 10, \
  #           window = 5, \
  #           sample = 0.001)
		# self.word2vecModel.init_sims(replace=True)
		self.word2vecModel = KeyedVectors.load_word2vec_format('./data/GoogleWord2Vec/GoogleNews-vectors-negative300.bin', binary=True)

	def word2vec(self, word):
		return self.word2vecModel[word]

	'''
	Pass a sentence, the function will return the average
	word2vec value for the words.
	'''
	def generateAvgWord2Vec(self, sentence):
		
		# Convert the sentence into list of words.
		tokenizedlist = word_tokenize(sentence)
		
		
		vectorValueSum = np.zeros((ExtractFeatures.FEATURE_SIZE_WORD2VEC,),dtype="float32")
		

		count = 0
		
		for word in tokenizedlist:
				
				count = count + 1.
				try:
					vectorValueSum = np.add(vectorValueSum,self.word2vec(word))
				except KeyError:
					
					count = count - 1.

		
		result = vectorValueSum
		if count > 0:
			
			result = np.divide(vectorValueSum,count)
		return result



	'''
	Uses valence data from http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
	The dataset contains words marked in range [-5,5] 
	'''
	def getValenceDict(self):
		valenceDict = dict(map(lambda (k,v): (k,int(v)), [ line.split('\t') for line in open('./data/AFINN/AFINN-111.txt') ]))
		return valenceDict
