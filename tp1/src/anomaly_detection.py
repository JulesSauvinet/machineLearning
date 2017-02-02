# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from prediction.lof import outliers
from text_mining import textMining


def showRawDatas(df):
    x = df.values[:, 0]
    y = df.values[:, 1]

    plt.plot(x,y, 'ro')
    plt.title("Detection d'anomalie")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#df3=pd.read_csv('../data/mouse-synthetic-data.txt', sep=' ')
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
df2 = pd.read_csv('../data/SMSSpamCollection.data', sep='\t')

#Preparation des donnees
#representation	SVD des SMS et colonne Spam/Ham associee pour chaque SMS
X, Y = textMining(df2, 10, 500, 50)
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
clf = IsolationForest(n_estimators=500, max_samples='auto', random_state=0, bootstrap=True,n_jobs=1, contamination = 0.02)#07)
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
      "| Spam", " "*4, FP, " "*8, FN, " "*4, "|","\n"  \
      "| ---------------------------- |","\n"  \
      "| Ham ", " "*4, VN, " "*7, VP, " "*2, "|","\n"  \
      "|_____________________________ |","\n"  \

# -------------------------------------------------------------------------------------------------------------------------#

