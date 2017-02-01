# coding=utf-8
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
import csv

from sklearn import datasets, preprocessing
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2,f_classif,mutual_info_classif
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,Imputer,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import scipy

# -------------------------------------------------------------------------------------------------------------------------#
#Extraction des donnees et creations des sets de test et d'apprentissage
from lof import LOF, outliers

np.set_printoptions(threshold=np.nan)
warnings.filterwarnings('ignore')

df=pd.read_csv('credit.data', sep='\t')

# on scinde les donnees
#les predicteurs
predictor = df.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
#la variable a predire
target = df.values[:,15]

#print "On a " + str(np.shape(predictor)[0]) + " lignes dans les donnees"
#print "On a " + str(np.shape(target)[0]) + " lignes dans les predictions"
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
def deleteRowsWithNan(predictor, target):
	# on ne garde que les valeurs numeriques
    predictor = predictor[:, [1, 2, 7, 10, 13, 14]]
	
    # on modifie les '?' par des NaN, et on supprime la ligne dans target qui, dans les donnees, contient un NaN
    for j in range(len(predictor)):
        pred = predictor[j]
        toDelete = False
        for i in range(len(pred)):
            if pred[i] == '?':
                pred[i] = np.nan
                toDelete = True
        if toDelete :
            target = np.delete(target, (j), axis=0)

    #on met toutes les valeurs au format numerique, et on supprime les lignes de donnees contenant des NaN
    predictor = predictor.astype(np.float)
    predictor = predictor[~np.isnan(predictor).any(axis=1)]

    return predictor, target
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
def transformTargetInBinary(target):
    target[target=='+']=1
    target[target=='-']=0
    target = target.astype(np.float)
    return target
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
def plotTargetDist(target):
    plt.hist(target, bins=2)
    plt.title("+ and -")
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------#


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


# -------------------------------------------------------------------------------------------------------------------------#
# Fonction de test qui run les 7 classifieurs en normalisant ou non les donnees, en ajoutant les CP de l'ACP et
# en ajoutant des combinaisons polynomiales des donnees (optionnel)
def test_classifiers(credit, withoutNorm = False, withStandardNorm = False, withMinMaxNorm = False,withPCA = False, withPoly = False):

	# --------------------------- SANS NORMALISATION ---------------------------------#
    if (withoutNorm):
        print "Sans centrage des données au préalable"
        run_classifiers(clfs,credit)

    if (withStandardNorm):
        standardScaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False)
        scalePred = standardScaler.fit(credit['data']).transform(credit['data'])
        scaleCredit = {'data': scalePred, 'target': credit['target']}
        print "Avec centrage standard des données au préalable"
        run_classifiers(clfs,scaleCredit)

    if (withMinMaxNorm or withPCA or withPoly  == True):
        minMaxScaler = preprocessing.MinMaxScaler(copy=True)
        scalePred2 = minMaxScaler.fit(credit['data']).transform(credit['data'])
        scaleCredit2 = {'data': scalePred2, 'target': credit['target']}

	    #------------------------- AVEC NORMALISATION (MIN,MAX) -----------------------#
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
            pcaCredit = {'data': pcaPred2, 'target': credit['target']}

	        #--------------------- AVEC L'AJOUT DES CP DE L'ACP  -----------------------#
            if (withPCA):
                print "Avec pca au préalable"
                run_classifiers(clfs,pcaCredit)

            # --------------------- AVEC COMBINAISONS POLYNOMIALES -----------------------#
            if (withPoly):
                poly = PolynomialFeatures(2)
                polyPred = poly.fit_transform(pcaPred2)
                polyPred = np.concatenate((polyPred, pcaPred2), axis=1)
                polyCredit = {'data': polyPred, 'target': credit['target']}
                print "Avec combinaisons polynomiales des données faites au préalable"
                run_classifiers(clfs,polyCredit)
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
#Lancement du test des classifieurs
def launchTestClassifiers(predictor, target) :
    predictor, target = deleteRowsWithNan(predictor, target)
    target = transformTargetInBinary(target)
    credit = {'data': predictor, 'target': target}
    test_classifiers(credit, False,False,False,False,True)
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
#Pre-traitement des donnees
#On augmente le jeu de donnees en decomposant les variables categorielles,
#En ajoutant les composantes principales extraites d'une ACP
#En ajoutant des combinaisons polynomiales des variables numériques
def preProcessDatas(predictor,target):
    # on recupere les donnees categorielles en prenant soin de remplacer les '?'
    pred_cat =	predictor[:,[0,3,4,5,6,8,9,11,12]]

    for	col_id	in range(len(pred_cat[0])):
        unique_val,	val_idx	= np.unique(pred_cat[:,col_id],	return_inverse=True)

        valNan = np.argwhere(unique_val == '?')

        val_idx = val_idx.astype(np.float)
        if (len(valNan > 0)):
            valNan = valNan[0][0]
            val_idx[val_idx==valNan]=np.nan

        pred_cat[:,col_id]	= val_idx

    # on recupere les donnees non categorielles en prenant soin de remplacer les '?'
    pred_not_cat = predictor[:, [1, 2, 7, 10, 13, 14]]
    for j in range(len(pred_not_cat)):
        pred = pred_not_cat[j]
        for i in range(len(pred)):
            if pred[i] == '?':
                pred[i] = np.nan

    # on met les donnees categorielles et non-categorielles au format numerique
    pred_cat = pred_cat.astype(np.float)
    pred_not_cat = pred_not_cat.astype(np.float)

    # on remplace les donnees manquantes non categorielles '?' par la moyenne des valeurs existantes
    imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
    pred_not_cat = imp_mean.fit_transform(pred_not_cat)

    # on remplace les donnees manquantes categorielles '?' par la valeur la plus fréquente
    imp_most_frequent = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    pred_cat = imp_most_frequent.fit_transform(pred_cat)

    # on transforme chaque variable categorielle avec m modalités en m variables binaires ---> une seule sera active
    enc = OneHotEncoder(categorical_features='all',handle_unknown='error', n_values='auto', sparse=True)

    enc.fit(pred_cat[:, [0,1,2,3,4,5,6,7,8]])
    pred_cat_bin = enc.transform(pred_cat[:, [0,1,2,3,4,5,6,7,8]]).toarray()

    # on normalise les donnees numeriques
    minMaxScaler = preprocessing.MinMaxScaler(copy=True)
    scale_pred_not_cat = minMaxScaler.fit(pred_not_cat).transform(pred_not_cat)
    #scale_pred_not_cat = pred_not_cat

    # on cree un polynome de degres 3 et on concatene ensemble les donnees de poly, les donnees non-cat, les donnees de l'ACP et les donnees non-cat binaires encodees
    poly = PolynomialFeatures(3)
    polyPred = poly.fit_transform(scale_pred_not_cat)

    scale_pred_not_cat_with_poly = np.concatenate((scale_pred_not_cat, polyPred), axis=1)
    predictNorm = np.concatenate((scale_pred_not_cat_with_poly, pred_cat_bin), axis=1)

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

    return predictNormWithPCA, target
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#

#Pre-traitement des donnees pour augmenter le dataset
predictor, target = preProcessDatas(predictor, target)

#Pour l'apprentissage et pour les tests de validation
predictorTrain, predictorTest, targetTrain, targetTest = train_test_split(predictor, target, test_size = 0.30, random_state =42)


# -------------------------------------------------------------------------------------------------------------------------#
#SELECTION DE VARIABLE selon selectKBest ou Random Forest
#TODO FIND the skitest (le seuil ou il y a un creux)
def variableSelector(predictorTrain, targetTrain, method, threshold = 0.0000001) :

    varSelected = []

    #selectKBest
    if (method == 'selectKBest'):
        print np.shape(predictorTrain)
        selector = SelectKBest(mutual_info_classif, k = 'all').fit(predictorTrain,targetTrain)
        #selector = SelectKBest(mutual_info_classif).fit(predictNormWithPCA,target)
        #selector = SelectPercentile(mutual_info_classif, percentile=100).fit(predictorTrain,targetTrain)

        scores = selector.scores_

        print scores
        for i in range(len(scores)):
            if (scores[i] > threshold*1000):
                varSelected.append(i)
        print scores[0]
        plt.hist(scores, bins = 30)#, bins = range(1,len(scores)))
        plt.title("Score de la selection de variables avec SelectKBest")
        plt.ylabel('Nombre de variables')
        plt.xlabel('Score')
        plt.show()

    # Random forest
    elif (method == 'rf'):
        clf = RandomForestClassifier(n_estimators=20)  # Random Forest
        clf.fit(predictorTrain, targetTrain)

        indexes = range(len(predictorTrain[0]))
        #print "Features sorted by their score:"
        #print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), indexes),reverse=True)

        plt.hist(clf.feature_importances_, bins = 30)  # , bins = range(1,len(scores)))
        plt.title("Score de la selection de variables avec Random Forest")
        plt.ylabel('Nombre de variables')
        plt.xlabel('Score')
        plt.show()

        scores = defaultdict(list)
        # crossvalidate the scores on a number of different random splits of the data
        for train_idx, test_idx in ShuffleSplit(n_splits=10, random_state=0, test_size=.3).split(predictorTrain):
            X_train, X_test = predictorTrain[train_idx], predictorTrain[test_idx]
            Y_train, Y_test = targetTrain[train_idx], targetTrain[test_idx]

            r = clf.fit(X_train, Y_train)
            acc = r2_score(Y_test, clf.predict(X_test))

            for i in range(predictorTrain.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, clf.predict(X_t))

                scores[indexes[i]].append((acc - shuff_acc) / acc)


        for feat, score in scores.items():
            scorebis = np.mean(score)
            if scorebis > threshold:
                varSelected.append(feat)

        varSelected = np.unique(varSelected)

    return varSelected
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
def testRf(predictor,target):
    creditNormalized = {'data': predictor, 'target': target}
    clfsRF = {
        'RF': RandomForestClassifier(n_estimators=20),  # Random Forest
    }
    run_classifiers(clfsRF, creditNormalized)
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
#predictorTrain, targetTrain = variableSelector(predictorTrain,targetTrain, 'selectKBest')
# cf http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
def processAndSelVarAndRunClassif (predictorTrain, targetTrain, predictorTest, targetTest):

    print "Avant selection de variables"
    print np.shape(predictorTest)
    run_classifiers(clfs, {'data': predictorTest, 'target': targetTest})

    varSelected = variableSelector(predictorTrain, targetTrain, 'rf')
    predictorTest = predictorTest[:, varSelected]

    print "Après selection de variables"
    print np.shape(predictorTrain)

    run_classifiers(clfs,{'data': predictorTest, 'target': targetTest})
# -------------------------------------------------------------------------------------------------------------------------#

#testRf(predictorTrain,targetTrain)
#processAndSelVarAndRunClassif(predictorTrain, targetTrain, predictorTest, targetTest)
#TODO plus de classsifieurs

# -------------------------------------------------------------------------------------------------------------------------#


# *************************************************************************************************************************#


# -------------------------------------------------------------------------------------------------------------------------#
#II.Apprentissage supervise des donnees textuelles


# -------------------------------------------------------------------------------------------------------------------------#
def countVectorize(corpus, targ):
    vect = CountVectorizer(stop_words='english')

    vectorizer = CountVectorizer(stop_words='english',max_df=1.0, min_df=15, max_features=500)

    vect.fit(corpus)
    vectorizer.fit(corpus) #cooccurences

    X1 = vect.transform(corpus)
    X = vectorizer.transform(corpus)

    #print np.shape(X1)
    #print np.shape(X)

    analyze = vectorizer.build_analyzer()

    targ[targ == 'ham'] = 1
    targ[targ == 'spam'] = 0

    #print np.shape(X.toarray())

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
    svd = TruncatedSVD(n_components=100, algorithm="randomized", n_iter=7,
                 random_state=42, tol=0.)
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
    df2 = pd.read_csv('SMSSpamCollection.data', sep='\t')
    X, Y = textMining(df2)
    run_classifiers(clfs, {'data': X.astype(np.float), 'target': Y.astype(np.float)})
# -------------------------------------------------------------------------------------------------------------------------#


# *************************************************************************************************************************#


# -------------------------------------------------------------------------------------------------------------------------#
#III.Apprentissage non supervise : Detection d'anomalies


# -------------------------------------------------------------------------------------------------------------------------#
def showRawDatas(df):
    x = df.values[:, 0]
    y = df.values[:, 1]


    plt.plot(x,y, 'ro')
    plt.title("Detection d'anomalie")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#df3=pd.read_csv('mouse-synthetic-data.txt', sep=' ')
#showRawDatas(df3)
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
# Detection d'anomalie selon deux technique : isolationforest ou LOF
def detectAnomaly(X,method = 'isolationforest', plot=True):
    # Premiere technique de detection d'anomalie
    # Isolation forest
    if (method == 'isolationforest'):
        X_train, X_test = train_test_split(X, test_size = 0.30, random_state =42)

        # creation du modele
        clf = IsolationForest(n_estimators=100,max_samples='auto', random_state=0)

        #construction du modele a partir des donnes d'apprentissage
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        X_out_idx = np.where(y_pred_train!=1)
        X_outliers_train = X_train[X_out_idx]

        #mesure de verification
        #y_pred_outliers = clf.predict(X_outliers_train)
        #print y_pred_outliers

        #execution du modele construit sur le jeu de validation
        y_pred_test = clf.predict(X_test)
        X_out_idx = np.where(y_pred_test != 1)
        X_outliers = X_train[X_out_idx]

        if (plot == True):
            # affichage des resultats de detection sur le set d'apprentissage et de validation
            xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1.5, 50))
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.title("IsolationForest")
            plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

            b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
            b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
            c1 = plt.scatter(X_outliers_train[:, 0], X_outliers_train[:, 1], c='blue')
            c2 = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

            plt.axis('tight')
            plt.xlim((0, 1))
            plt.ylim((0, 1.5))
            plt.legend([b1, b2, c1,c2],
                       ["obs d'apprentissage","anomalies detectees sur les obs d'apprentissage"
                        "nouvelles obs", "anomalies detectees sur les nouvelles obs"],
                       loc="upper left")
            plt.show()
        else :
            print X_outliers

    # Deuxieme technique de detection d'anomalie
    # Local Outlier Factor
    # https://github.com/damjankuznar/pylof/blob/master/lof.py -> marche pas? trop lent ou trop de outlier?
    # http://scikit-learn.org/dev/auto_examples/neighbors/plot_lof.html
    if (method == 'lof'):
        instances = []
        for x in X:
            instances.append((x[0], x[1]))

        result = outliers(1, instances)

        X_outliers = []
        for outlier in result:
            #print outlier["lof"], outlier["instance"]
            x,y = outlier["instance"]
            coord = []
            coord.append(x)
            coord.append(y)
            X_outliers.append(coord)

        X_outliers = np.array(X_outliers)

        if (plot == True):
            plt.title("Local Outlier Factor (LOF)")

            a = plt.scatter(X[:, 0], X[:, 1], c='green')
            b = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

            plt.axis('tight')
            plt.xlim((0, 1))
            plt.ylim((0, 1.5))
            plt.legend([a, b],["observations normales","observations anormales"],loc="upper left")

            plt.show()
        else :
            print X_outliers

# Test
#1. Sur la base de données Mouse
#TODO normaliser les donnees
# #df3=pd.read_csv('mouse-synthetic-data.txt', sep=' ')
#X = df.values[:, [0,1]]
#detectAnomaly(X,'isolationforest')
#detectAnomaly(X,'lof')

# Test
# 2. Sur le	jeu	de données des SMS
df2 = pd.read_csv('SMSSpamCollection.data', sep='\t')

#Preparation des donnees
#representation	SVD des SMS et colonne Spam/Ham associee pour chaque SMS
X, Y = textMining(df2)
#print np.shape(X)

#concatenation de la representation SVD des textes SMS et la categorie Spam/Ham
Y = np.reshape(Y, (len(Y), 1))
Z = np.concatenate((X, Y), axis=1)

#extraction des SMS labelle Spam ou Ham
Spam = Z[Z[:, Z.shape[1]-1] == 0]
Ham =  Z[Z[:, Z.shape[1]-1] == 1]

#recuperer la moitie des donnees Ham aleatoirement
halfHam = Ham[np.random.randint(0,Ham.shape[0],Ham.shape[0]/2)]
#print np.shape(halfHam)

#recuperer 20 spams aleatoirement
sampleSpam = Spam[np.random.randint(0,Spam.shape[0],20)]
#print np.shape(sampleSpam)

#le jeu de donnee sur lequel on va faire de la detection d'anomalie
datas = np.concatenate((halfHam, sampleSpam), axis=0)

#les index des outliers
outs = np.argwhere(datas[:, datas.shape[1]-1] == 0)

#on supprime la colonne avec la valeur cible Spam/Ham
datas = scipy.delete(datas, datas.shape[1]-1, 1)

print np.shape(datas)
#on fit le modele
clf = IsolationForest(n_estimators=100, max_samples='auto', random_state=0, bootstrap=True,n_jobs=1, contamination = 0.01)
clf.fit(datas)
y_pred = clf.predict(datas)

#les outliers predits
X_out_idx = np.where(y_pred != 1)

#X_out_idx.append(2412)#, 2413, 2414, 2415])
outs = np.transpose(outs)

outs = outs[0]
X_out_idx = X_out_idx[0]

#print outs
#print X_out_idx

FP = len(np.intersect1d(outs, X_out_idx))
FN = len(X_out_idx)-FP

V = datas.shape[0]-len(X_out_idx)

VN = len(outs) - FP
VP = V - VN

print "Matrice de confusion"
print " ______________________________", "\n"  \
      "| P\R      Spam        Ham     |","\n"  \
      "| ---------------------------- |","\n"  \
      "| Spam", " "*4, FP, " "*8, FN, " "*3, "|","\n"  \
      "| ---------------------------- |","\n"  \
      "| Ham ", " "*4, VN, " "*7, VP, " "*2, "|","\n"  \
      "|_____________________________ |","\n"  \





# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#



