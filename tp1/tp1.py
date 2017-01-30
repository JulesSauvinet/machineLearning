# coding=utf-8

import numpy as np
import time
import csv

from sklearn import datasets, preprocessing
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import	warnings
warnings.filterwarnings('ignore')

ALPHA = 2

df=pd.read_csv('credit.data', sep='\t')

predictor = df.values[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
target = df.values[:,15]

predictor = predictor[:,[1,2,7,10,13,14]]

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

import matplotlib.pyplot as plt

target[target=='+']=1
target[target=='-']=0
plt.hist(target, bins=2)
#plt.hist passes it's arguments to np.histogram
#plt.title("+ and -")
#plt.show()

#print np.shape(predictor)
#print np.shape(target)

df.predictor = predictor
#print target

target=target.astype(np.float)

credit = {'data': predictor, 'target': target}

#Test du Naive Bayes
def NaiveBayesSimple(credit) :
    gnb = GaussianNB()
    y_pred = gnb.fit(credit['data'], credit['target']).predict(credit['data'])
    print("Number of mislabeled points out of a total %d points : %d"% (credit['data'].shape[0],(credit['target'] != y_pred).sum()))
    return y_pred

clf_init = None
clfs =	{
    'NBS'  : GaussianNB(), #Naive Bayes Classifier
    'RF'   : RandomForestClassifier(n_estimators=20), #Random Forest
    'KNN'  : KNeighborsClassifier(n_neighbors=10,  weights='uniform', algorithm='auto', p=2, metric='minkowski'), #K plus proches voisins
    'CART' : tree.DecisionTreeClassifier(min_samples_split=300, random_state=99,criterion='gini'), #Arbres de décision CART
    'MLP'  : MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=3, learning_rate = 'adaptive'), # MLP perceptron multi-couches,
    'ADAB' : AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,random_state=99,criterion='gini'), #Adaboost avec arbre de décision
                                 algorithm="SAMME",n_estimators=50),
    'GBC'  : GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10, subsample=0.3,
                                         min_samples_split=2,min_samples_leaf=1,max_depth=1,init=clf_init,
                                         random_state=1,max_features=None,verbose=0) #Gradient boosting classifier
}

def run_classifiers(clfs, credit):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for clf in clfs:

        timeStart = time.time()
        print "*",'-'*75,"*"
        print "Classifieur ", clf, " :"
        
        #Accuracy
        cv_acc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='accuracy', cv=kf)  # pour le calcul de l’accuracy
        avg_acc = cv_acc.mean()
        std_acc = cv_acc.std()
        print "Moyenne de l'accuracy du classifieur : ", avg_acc
        print "Ecart-type de l'accuracy du classifieur : ", std_acc

        #ROC
        cv_roc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='roc_auc', cv=kf)  # pour	le calcul de l'AUC
        avg_roc = cv_roc.mean()
        std_roc = cv_roc.std()
        print "Moyenne de l'AUC du classifieur : ", avg_roc
        print "Ecart-type de l'AUC du classifieur : ", std_roc

        #Précision +
        cv_prec = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='precision', cv=kf)  # pour le calcul de la précision des +
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


#On run les 7 classifieurs et on affiche les mesures de qualité (Précision, Accuracy, AUC et temps d'exécution) pour comparer


# ------------------------------------------------------- SANS NORMALISATION -------------------------------------------------------
print "Sans centrage des données au préalable"
run_classifiers(clfs,credit)
# ----------------------------------------------------------------------------------------------------------------------------------       


# ------------------------------------------------------- AVEC NORMALISATION -------------------------------------------------------
standardScaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False)
scalePred = standardScaler.fit(predictor).transform(predictor)
scaleCredit = {'data': scalePred, 'target': target}
print "Avec centrage standard des données au préalable"
run_classifiers(clfs,scaleCredit)
# ----------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------- AVEC NORMALISATION (MIN,MAX) -------------------------------------------------
minMaxScaler = preprocessing.MinMaxScaler(copy=True)
scalePred2 = minMaxScaler.fit(predictor).transform(predictor)
scaleCredit2 = {'data': scalePred2, 'target': target}
print "Avec centrage selon min et max des données au préalable"
run_classifiers(clfs,scaleCredit2)
# ----------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------- AVEC L'ACP CONCATENEE ----------------------------------------------------
pca = PCA(n_components=0)
for i in range(6):
    pca = PCA(n_components=i)
    pca.fit(scalePred2)
    if np.sum(pca.explained_variance_ratio_) > 0.7:
        break

pcaPred = pca.transform(scalePred2)
pcaPred2 = np.concatenate((scalePred2, pcaPred), axis=1)
pcaCredit = {'data': pcaPred2, 'target': target}

print "Avec pca au préalable"
run_classifiers(clfs,pcaCredit)
# ----------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------- AVEC COMBINAISONS POLYNOMIALES -----------------------------------------------
poly = PolynomialFeatures(2)
polyPred = poly.fit_transform(pcaPred2)
polyCredit = {'data': polyPred, 'target': target}
print "Avec combinaisons polynomiales des données faites au préalable"
run_classifiers(clfs,polyCredit)
# ----------------------------------------------------------------------------------------------------------------------------------
