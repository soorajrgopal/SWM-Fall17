import pickle
from randomForestClassifier import RandomForestClassifer
from naiveBayesClassifier import SentiNaiveBayesClassifier
import sys

class TestClassifier:

	def __init__(self):
		self.modelMap = {'1':'./model/randomForestClassiferModel.pickle','2':'./model/naiveBayesClassifierModel.pickle'}
		self.testMovieList = {'path' : '../scraping/', 'fileName' : 'test_movie_list.txt'}
		self.testMovieReview = {'path' : './data/test/'}

	# Returns the deserialized trained model.
	def getPickle(self, opt):
		try:
			with open(self.modelMap[opt],'rb') as f_pickle:
				return pickle.load(f_pickle)
		except IOError:
			print ('The pickle file is missing from its path, please re-train the model to generate the file. \n')
			sys.exit(1)

	def run(self, opt):
		#Unpacks the trained model from the pickle file.
		trainObj = self.getPickle(opt)

		try:
			with open(self.testMovieList['path'] + self.testMovieList['fileName'], 'r') as f_test_movie:
				test_movies = f_test_movie.readlines()
			#Prepares the list of test movies.
			test_movies = [i.strip().replace('\n','')[3:] for i in test_movies]
		except IOError:
			print ('\n' + 'The file %s is missing from its path.\n' % (self.testMovieList['fileName']))
			sys.exit(1)

		for movie in test_movies:
			count = 0
			sum = 0
			try:
				with open(self.testMovieReview['path'] + movie + '.txt' , 'r') as f_movie_review:
					#Extracts the reviews for a test movie into a list.
					movie_reviews = f_movie_review.readlines()
			except IOError:
				print('The file containing the extracted reviews for %s is missing from its path, please generate file by running test_scrape.py' % (movie))
				continue

			for review in movie_reviews:
				#Computes the sum of the sentimental analysis rating for each review.
				sum = sum + int(trainObj.classify(review.strip().replace('\n','')))
				count += 1

			print("Sentiment Rating For %s Movie : %d" % (movie, sum/count))
			f_movie_review.close()

if __name__ == '__main__':
	classifierObject = TestClassifier()
	validOptions = ['1','2']
	opt = input('Available Classifiers : \n1.RandomForestClassifer \n2.NaiveBayesClassifier \nChoose your option : ')
	if opt in validOptions:
		classifierObject.run(opt)
	else:
		print('An invalid option was selected !')
		sys.exit(1)


