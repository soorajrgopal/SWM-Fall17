import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from extractFeatures import ExtractFeatures
import numpy as np

class RNTNLSTMClassifier:

	SEQLEN = 5
	BATCHSIZE = 1
	NLAYERS = 3
	INTERNALSIZE = 512
	VOCABSIZE = 300

	learning_rate = 0.001  # fixed learning rate
	dropout_pkeep = 1.0

	def __init__(self):
		self.classifier = None
		self.word2vec = None

	'''
		Train a naivebayes classifer with the training data.
	'''
	def train(self):
		
		# Get data from the extracter
		extracter = ExtractFeatures('./data/reviewsData.txt')
		tokenizedData = extracter.getTokenizedData()

		trainingData = tokenizedData['train']
		self.word2vec = KeyedVectors.load_word2vec_format('./data/GoogleWord2Vec/GoogleNews-vectors-negative300.bin', binary=True)
		
		print ''
		print 'Training RNTN - LSTM Classifier'
		print 'Training data size = ', len(trainingData)
		print ''
		lr = tf.placeholder(tf.float32, name='lr')  # learning rate
		pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
		batchsize = tf.placeholder(tf.int32, name='batchsize')

		X = tf.placeholder(tf.float32, [None,None, self.VOCABSIZE], name='X')
		Y_ = tf.placeholder(tf.uint8, [None, 2], name='Y_')
		Hin = tf.placeholder(tf.float32, [None, self.INTERNALSIZE*self.NLAYERS], name='Hin')
		
		onecell = rnn.GRUCell(self.INTERNALSIZE)
		dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
		multicell = rnn.MultiRNNCell([dropcell]*self.NLAYERS, state_is_tuple=False)
		multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
		Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=Hin)
		H = tf.identity(H, name='H')


		Yflat = tf.reshape(Yr, [-1, self.INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
		Ylogits = layers.linear(Yflat, 2)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
		Yflat_ = tf.reshape(Y_, [-1, 2])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
		loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
		Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
		Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
		Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
		train_step = tf.train.AdamOptimizer(lr).minimize(loss)

		istate = np.zeros([self.BATCHSIZE, self.INTERNALSIZE*self.NLAYERS])  # initial zero input state
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		step = 0

		for idx in range(1):

			for train_item in trainingData:
				input_x = []

				for word in train_item[0]:
					try:
						input_x.append(self.word2vec[word])
					except KeyError:
						continue
				
				y_ = [[1 if train_item[1] == '0' else 0, 1 if train_item[1] == '1' else 0]]
				print 'One sentence' , y_	
				feed_dict = {X: [input_x], Y_: y_, Hin: istate, lr: self.learning_rate, pkeep: self.dropout_pkeep, batchsize: self.BATCHSIZE}
				_, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)
				istate = ostate
				

		print 'Training RNTN - LSTM Classifier Completed'

	def validateClassifier(self):

		
		testData=[]
		print ''
		print 'Validating RNTN - LSTM Classifier'
		print 'Test data size = ', len(testData)
		print ''

		
		print 'Accuracy: ', 0

	
	def classify(self, statusMessage):
		
		return 1