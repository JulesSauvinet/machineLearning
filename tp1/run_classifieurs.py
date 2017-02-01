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


# -------------------------------------------------------------------------------------------------------------------------#
#Definition des classifieurs dans un dictionnaire
clf_init = None
clfs =	{
    'NBS' : GaussianNB(), #Naive Bayes Classifier
    'RF':	RandomForestClassifier(n_estimators=20), #Random Forest
    'KNN':	KNeighborsClassifier(n_neighbors=10,  weights='uniform', algorithm='auto', p=2, metric='minkowski'), #K plus proches voisins
    'CART':  tree.DecisionTreeClassifier(min_samples_split=50, random_state=99,criterion='gini'), #Arbres de décision CART
    'ADAB' : AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,random_state=99,criterion='gini'), #Adaboost avec arbre de décision
                         algorithm="SAMME",
                         n_estimators=50),
    'MLP' : MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=3, learning_rate = 'adaptive'), # MLP perceptron multi-couches,
    'GBC' : GradientBoostingClassifier( loss='deviance', learning_rate=0.1, n_estimators=10, subsample=0.3,
                                        min_samples_split=2, min_samples_leaf=1, max_depth=1, init=clf_init,
                                        random_state=1, max_features=None, verbose=0) #Gradient boosting classifier
}
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
# Fonction qui run les 7 classifieurs et qui affiche des mesures de qualité
# (Précision, Accuracy, AUC et temps d'exécution) pour comparer les différents classifieurs
def run_classifiers(clfs, credit):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for clf in clfs:

        timeStart = time.time()

        print "*",'-'*100,"*"

        print clf
        #Accuracy
        cv_acc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='accuracy', cv=kf)  # pour	le	calcul	de	l’accuracy
        avg_acc = cv_acc.mean()
        std_acc = cv_acc.std()
        print "Accuracy (mean) de ", clf, "    : " ,avg_acc, "( std de ", std_acc, ")"

        #ROC
        cv_roc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='roc_auc')#, cv=kf2)  # pour	le	calcul	de	l'AUC
        avg_roc = cv_roc.mean()
        std_roc = cv_roc.std()
        print "AUC (mean) de ", clf, "         : " ,avg_roc, "( std de ", std_roc, ")"

        #Précision +
        cv_prec = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='precision', cv=kf)  # pour	le	calcul	de	la précision des +
        avg_prec = cv_prec.mean()
        std_prec = cv_prec.std()
        print "Precision + (mean) de ", clf, " : " , avg_prec, "( std de ", std_prec, ")"

        timeEnd = time.time()
        timeExec = timeEnd - timeStart

        print "(in ", timeExec," secondes)"

        print "*",'-'*100,"*"
        print ""
# -------------------------------------------------------------------------------------------------------------------------#
