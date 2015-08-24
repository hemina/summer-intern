from sklearn import svm
import os
from sklearn import feature_selection
import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn import cross_validation
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

#trainset='bench_labeled_review.tsv',testset='testData.tsv'
#labeledTrainData.tsv bench_labeled_review.tsv
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data','bench_labeled_review.tsv'), header=0, \
                    delimiter="\t", quoting=3)
labels = train["sentiment"]
# print 'Download text data sets. If you already have NLTK datasets downloaded,just close the Python download window...'
# 	nltk.download() # Download text data sets, including stop words
  			
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length of the movie review list

#print len(train["review"])
print "Cleaning and parsing the training set movie reviews...\n"
for i in xrange( 0, len(train["review"])):
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
#train_reviews=clean_train_reviews
	
vectorizer = TfidfVectorizer(max_df=0.5)
FeatureVecs=vectorizer.fit_transform(clean_train_reviews)

FeatureVecs = FeatureVecs.toarray()
print len(FeatureVecs)

#FeatureVecs = FeatureVecs.astype('float32')

selector = SelectPercentile(chi2, percentile=15)
selector.fit(FeatureVecs,labels.values)
FeatureVecs = selector.transform(FeatureVecs)
"""  
param_grid = [
#{'tdidf__max_df':[0.3,0.4,0.5,0.6,0.7]},
{'selector__percentile':[5,10,15,20]},
]
 
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
""" 
clf = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0, degree=3,

  gamma=0.001, kernel='rbf', max_iter=-1, probability=False,

  random_state=None, shrinking=True, tol=0.001, verbose=False)

#clf = GaussianNB()
#pipeline=Pipeline([('selector',selector),('classifier',clf)])
#grid = GridSearchCV(pipeline, param_grid, cv=10)
#clf = tree.DecisionTreeClassifier()
#cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.3, random_state=0)
#grid.fit(FeatureVecs, labels)
 
scores = cross_validation.cross_val_score(clf, FeatureVecs, labels, cv=10)
#print(grid.best_estimator_)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#clf.fit(FeatureVecs, labels)
