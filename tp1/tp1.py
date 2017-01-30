# coding=utf-8
import numpy as np
import time
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import csv

import pandas as pd
import	numpy	as	np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
import	warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('credit.data', sep='\t')

predictor = df.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
target = df.values[:,15]


def deleteRowsWithNan(predictor, target):
    predictor = predictor[:, [1, 2, 7, 10, 13, 14]]
    for j in range(len(predictor)):
        pred = predictor[j]
        toDelete = False
        for i in range(len(pred)):
            if pred[i] == '?':
                pred[i] = np.nan
                toDelete = True
        if toDelete :
            target = np.delete(target, (j), axis=0)

    predictor = predictor.astype(np.float)
    predictor = predictor[~np.isnan(predictor).any(axis=1)]

    return predictor, target

def transformTargetInBinary(target):
    target[target=='+']=1
    target[target=='-']=0
    target = target.astype(np.float)
    return target

def plotTargetDist(target):
    plt.hist(target, bins=2)  # plt.hist passes it's arguments to np.histogram
    plt.title("+ and -")
    plt.show()

#print "On a " + np.shape(predictor) + " lignes dans les donnees"
#print np.shape(target)
#print target

#Fonction pour le test du Naive Bayes
def NaiveBayesSimple(credit) :
    gnb = GaussianNB()
    y_pred = gnb.fit(credit['data'], credit['target']).predict(credit['data'])
    print("Number of mislabeled points out of a total %d points : %d"% (credit['data'].shape[0],(credit['target'] != y_pred).sum()))
    return y_pred

clf_init = None
clfs =	{
    'NBS' : GaussianNB(), #Naive Bayes Classifier
    'RF':	RandomForestClassifier(n_estimators=20), #Random Forest
    'KNN':	KNeighborsClassifier(n_neighbors=10,  weights='uniform', algorithm='auto', p=2, metric='minkowski'), #K plus proches voisins
    'CART':  tree.DecisionTreeClassifier(min_samples_split=300, random_state=99,criterion='gini'), #Arbres de décision CART
    'ADAB' : AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,random_state=99,criterion='gini'), #Adaboost avec arbre de décision
                         algorithm="SAMME",
                         n_estimators=50),
    'MLP' : MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=3, learning_rate = 'adaptive'), # MLP perceptron multi-couches,
    'GBC' : GradientBoostingClassifier( loss='deviance', learning_rate=0.1,
                             n_estimators=10, subsample=0.3,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             max_depth=1,
                             init=clf_init,
                             random_state=1,
                             max_features=None,
                             verbose=0) #Gradient boosting classifier
     #liste	a completer
}

def run_classifiers(clfs, credit):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for clf in clfs:

        timeStart = time.time()

        print "*",'-'*100,"*"
        print "Classifieur ", clf, " :"
        #Accuracy
        cv_acc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='accuracy', cv=kf)  # pour	le	calcul	de	l’accuracy
        avg_acc = cv_acc.mean()
        std_acc = cv_acc.std()
        print "Moyenne de l'accuracy du classifieur : ", avg_acc
        print "Ecart-type de l'accuracy du classifieur : ", std_acc

        #ROC
        cv_roc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='roc_auc', cv=kf)  # pour	le	calcul	de	l'AUC
        avg_roc = cv_roc.mean()
        std_roc = cv_roc.std()
        print "Moyenne de l'AUC du classifieur : ", avg_roc
        print "Ecart-type de l'AUC du classifieur : ", std_roc

        #Précision +
        cv_prec = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='precision',
                                 cv=kf)  # pour	le	calcul	de	la précision des +
        avg_prec = cv_prec.mean()
        std_prec = cv_prec.std()
        print "Moyenne de la précision + du classifieur : ", avg_prec
        print "Ecart-type de la précision + du classifieur : ", std_prec

        print ""
        timeEnd = time.time()
        timeExec = timeEnd - timeStart

        print "Temps d'exécution de l'algorithme (*3 pour obtenir les 3 scoring)  : ", timeExec," secondes"
        print ""

        print "*",'-'*100,"*"
        print ""


def test_classifiers(credit, withoutNorm = False, withStandardNorm = False, withMinMaxNorm = False,
                     withPCA = False, withPoly = False):
    #run_classifiers(clfs,credit)
    #On run les 7 classifieurs et on affiche les mesures de qualité (Précision, Accuracy, AUC et temps d'exécution) pour comparer
    #SANS NORMALISATION
    if (withoutNorm):
        print "Sans centrage des données au préalable"
        run_classifiers(clfs,credit)

    if (withStandardNorm):
        standardScaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False)
        scalePred = standardScaler.fit(predictor).transform(predictor)
        scaleCredit = {'data': scalePred, 'target': target}

        #On run les 7 classifieurs et on affiche les mesures de qualité (Précision, Accuracy, AUC et temps d'exécution) pour comparer
        #AVEC NORMALISATION
        print "Avec centrage standard des données au préalable"
        run_classifiers(clfs,scaleCredit)


    if (withMinMaxNorm or withPCA  or withPoly  == True):
        minMaxScaler = preprocessing.MinMaxScaler(copy=True)
        scalePred2 = minMaxScaler.fit(predictor).transform(predictor)
        scaleCredit2 = {'data': scalePred2, 'target': target}

        if (withMinMaxNorm):
            print "Avec centrage selon min et max des données au préalable"
            run_classifiers(clfs,scaleCredit2)


        if (withPCA or withPoly  == True):
            pca = PCA(n_components=0)
            for i in range(6):
                pca = PCA(n_components=i)
                pca.fit(scalePred2)
                if np.sum(pca.explained_variance_ratio_) > 0.7:
                    break

            pcaPred = pca.transform(scalePred2)
            pcaPred2 = np.concatenate((scalePred2, pcaPred), axis=1)

            pcaCredit = {'data': pcaPred2, 'target': target}

            if (withPCA):
                print "Avec pca au préalable"
                run_classifiers(clfs,pcaCredit)

            if (withPoly):
                poly = PolynomialFeatures(2)
                polyPred = poly.fit_transform(pcaPred2)
                polyPred = np.concatenate((polyPred, pcaPred2), axis=1)
                polyCredit = {'data': polyPred, 'target': target}
                print "Avec combinaisons polynomiales des données faites au préalable"
                run_classifiers(clfs,polyCredit)

#predictor, target = deleteRowsWithNan(predictor, target)
#target = transformTargetInBinary(target)
#credit = {'data': predictor, 'target': target}
#test_classifiers(credit, False,False,False,False,True)


pred_cat =	predictor[:,[0,3,4,5,6,8,9,11,12]]

for	col_id	in range(len(pred_cat[0])):
    unique_val,	val_idx	= np.unique(pred_cat[:,col_id],	return_inverse=True)

    valNan = np.argwhere(unique_val == '?')

    val_idx = val_idx.astype(np.float)
    if (len(valNan > 0)):
        valNan = valNan[0][0]
        val_idx[val_idx==valNan]=np.nan

    pred_cat[:,col_id]	= val_idx

pred_not_cat = predictor[:, [1, 2, 7, 10, 13, 14]]
for j in range(len(pred_not_cat)):
    pred = pred_not_cat[j]
    for i in range(len(pred)):
        if pred[i] == '?':
            pred[i] = np.nan

pred_cat = pred_cat.astype(np.float)
pred_not_cat = pred_not_cat.astype(np.float)

imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
pred_not_cat = imp_mean.fit_transform(pred_not_cat)

imp_most_frequent = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
pred_cat = imp_most_frequent.fit_transform(pred_cat)

enc = OneHotEncoder(categorical_features='all',
    handle_unknown='error', n_values='auto', sparse=True)


enc.fit(pred_cat[:, [0,1,2,3,4,5,6,7,8]])
pred_cat_bin = enc.transform(pred_cat[:, [0,1,2,3,4,5,6,7,8]]).toarray()

minMaxScaler = preprocessing.MinMaxScaler(copy=True)
scale_pred_not_cat = minMaxScaler.fit(pred_not_cat).transform(pred_not_cat)

poly = PolynomialFeatures(2)
polyPred = poly.fit_transform(scale_pred_not_cat)
scale_pred_not_cat_with_poly = np.concatenate((scale_pred_not_cat, polyPred), axis=1)

predictNorm = np.concatenate((scale_pred_not_cat_with_poly, pred_cat_bin), axis=1)

#print np.shape(predictNorm)
pca = PCA(n_components=0)
for i in range(46):
    pca = PCA(n_components=i)
    pca.fit(predictNorm)
    if np.sum(pca.explained_variance_ratio_) > 0.7:
        break

predictNormPCA = pca.transform(predictNorm)
predictNormWithPCA = np.concatenate((predictNorm, predictNormPCA), axis=1)

#print np.shape(predictNormWithPCA)

target = transformTargetInBinary(target)

creditNormalized  = {'data': predictNormWithPCA, 'target': target}

#run_classifiers(clfs,creditNormalized)

