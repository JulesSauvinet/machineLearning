# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from model.Classifiers import run_classifiers, clfs, clfs2

#II. Apprentissage supervisé sur	des	données	textuelles :	Feature	engineering	et	Classification
# -------------------------------------------------------------------------------------------------------------------------#
def countVectorize(corpus, targ, max_df, minOccurence, maxfeatures, bigram, stop_words):

    if (stop_words == True):
        vectorizer = CountVectorizer(stop_words='english',max_df=1.0, min_df=minOccurence, max_features=maxfeatures)
    else :
        if (bigram == False):
            vectorizer = CountVectorizer(stop_words=None,max_df=max_df, min_df=minOccurence, max_features=maxfeatures)

        else:
            vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=None,max_df=max_df, min_df=minOccurence, max_features=maxfeatures, token_pattern=r'\b\w+\b')

    vectorizer.fit(corpus) #cooccurences
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
def truncateSVD(X, svdSize = 25):
    svd = TruncatedSVD(n_components=svdSize, algorithm="randomized", n_iter=6,random_state=42, tol=0.)
    svdX = svd.fit_transform(X)

    return svdX
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
#Preparation du set de donnees textuelle pour faire de la classification
def textMining(df2, max_df=1.0, minOccurence = 15, maxfeatures=100, svdSize = 25, bigram = False,  stop_words = True):
    corpus = df2.values[:, 1] # le predicteur
    targ = df2.values[:, 0]   # la variable a predire  # TODO train et test

    X,Y = countVectorize(corpus,targ, max_df,
                         minOccurence,  maxfeatures,
                         bigram, stop_words) #ajout pour chaque SMS des occurences des termes les plus frequents du dataset de SMS
    X = tfIdfize(X)                          #calcul d'importance des termes a l'aide de cooccurence
    X = truncateSVD(X, svdSize)              #reduction de sparse matrix avec SVD (single value detection)
    return X,Y
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
#Procedure de test
def testTextMining(df2, max_df=1.0, minOccurence = 15, maxfeatures=100, svdSize = 25, bigram = False,  stop_words = True):
    X, Y = textMining(df2, max_df, minOccurence, maxfeatures, svdSize, bigram,  stop_words)
    run_classifiers(clfs2, {'data': X.astype(np.float), 'target': Y.astype(np.float)})
# -------------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    df2 = pd.read_csv('../data/SMSSpamCollection.data', sep='\t')
    testTextMining(df2, 1.0, 15, 200, 25, False, True)
