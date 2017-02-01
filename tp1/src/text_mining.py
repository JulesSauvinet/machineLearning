# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from model.Classifiers import run_classifiers, clfs


# -------------------------------------------------------------------------------------------------------------------------#
def countVectorize(corpus, targ):
    vect = CountVectorizer(stop_words='english')
    vectorizer = CountVectorizer(stop_words='english',max_df=1.0, min_df=15, max_features=150)

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
    svd = TruncatedSVD(n_components=25, algorithm="randomized", n_iter=6,random_state=42, tol=0.)
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
    df2 = pd.read_csv('../data/SMSSpamCollection.data', sep='\t')
    X, Y = textMining(df2)
    run_classifiers(clfs, {'data': X.astype(np.float), 'target': Y.astype(np.float)})
# -------------------------------------------------------------------------------------------------------------------------#

#df2 = pd.read_csv('data\SMSSpamCollection.data', sep='\t')
#X, Y = textMining(df2)
#testTextMining()
