# coding=utf-8
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import warnings
import time
import csv

from sklearn import datasets, preprocessing
from sklearn import tree
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,IsolationForest,RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2,f_classif,mutual_info_classif
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,cross_val_score,ShuffleSplit,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,Imputer,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from run_classifieurs import run_classifiers


# -------------------------------------------------------------------------------------------------------------------------#
#Definition des classifieurs dans un dictionnaire
clf_init = None
clfs =	{
    #Naive Bayes Classifier
    'NBS' : GaussianNB(),

    #Random Forest
    'RF':   RandomForestClassifier(n_estimators=20),

    #K plus proches voisins
    'KNN':  KNeighborsClassifier(n_neighbors=10,  weights='uniform', algorithm='auto', p=2, metric='minkowski'),

    #Arbres de décision CART
    'CART': tree.DecisionTreeClassifier(min_samples_split=50, random_state=99,criterion='gini'),

    #Adaboost avec arbre de décision
    'ADAB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,random_state=99,criterion='gini'),algorithm="SAMME",n_estimators=50),

    # MLP perceptron multi-couches,
    'MLP' : MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=3, learning_rate = 'adaptive'),

    #Gradient boosting classifier
    'GBC' : GradientBoostingClassifier( loss='deviance', learning_rate=0.1, n_estimators=10, subsample=0.3,min_samples_split=2,
                                        min_samples_leaf=1, max_depth=1, init=clf_init,random_state=1, max_features=None, verbose=0)
}
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
def countVectorize(corpus, targ):
    vect = CountVectorizer(stop_words='english')
    vectorizer = CountVectorizer(stop_words='english',max_df=1.0, min_df=15, max_features=500)

    vect.fit(corpus)
    vectorizer.fit(corpus) #cooccurences

    X1 = vect.transform(corpus)
    X = vectorizer.transform(corpus)

    analyze = vectorizer.build_analyzer()

    targ[targ == 'ham'] = 1
    targ[targ == 'spam'] = 0

    return X.toarray(), targ
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
def tfIdfize(X):
    transformer = TfidfTransformer(smooth_idf=False, norm='l2', use_idf=True, sublinear_tf=False)
    tfidf = transformer.fit_transform(X)

    return tfidf.toarray()
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
def truncateSVD(X):
    svd = TruncatedSVD(n_components=100, algorithm="randomized", n_iter=7,random_state=42, tol=0.)
    svdX = svd.fit_transform(X)

    return svdX
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
#Preparation du set de donnees textuelle pour faire de la classification
def textMining(df2):
    corpus = df2.values[:, 1] # le predicteur
    targ = df2.values[:, 0]   # la variable a predire  # TODO train et test

    X,Y = countVectorize(corpus,targ) #ajout pour chaque SMS des occurences des termes les plus frequents du dataset de SMS
    X = tfIdfize(X)                   #calcul d'importance des termes a l'aide de cooccurence
    X = truncateSVD(X)                #reduction de sparse matrix avec SVD (single value detection)
    return X,Y
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
#TEST
def testTextMining():
    df2 = pd.read_csv('data\SMSSpamCollection.data', sep='\t')
    X, Y = textMining(df2)
    run_classifiers(clfs, {'data': X.astype(np.float), 'target': Y.astype(np.float)})
# -------------------------------------------------------------------------------------------------------------------------#

#df2 = pd.read_csv('data\SMSSpamCollection.data', sep='\t')
#X, Y = textMining(df2)
testTextMining()
