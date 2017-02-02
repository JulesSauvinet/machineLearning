# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from prediction.lof import outliers
from text_mining import textMining

# -------------------------------------------------------------------------------------------------------------------------#
def showRawDatas(df):
    x = df.values[:, 0]
    y = df.values[:, 1]

    plt.plot(x,y, 'ro')
    plt.title("Detection d'anomalie")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
# Detection d'anomalie selon deux technique : isolationforest ou LOF
def detectAnomaly(X,method = 'isolationforest', plot=True):
    # Premiere technique de detection d'anomalie
    # Isolation forest
    if (method == 'isolationforest'):
        clf = IsolationForest(contamination = 10./500.)

        #construction du modele a partir des donnes d'apprentissage
        clf.fit(X)

        # on recupere le resultat
        y_pred = clf.predict(X)
        X_out_idx = np.where(y_pred != 1)
        X_outliers = X[X_out_idx]

        if (plot == True):
            # affichage des resultats de detection sur le set d'apprentissage et de validation
            xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1.5, 50))
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.title("IsolationForest")
            plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

            b = plt.scatter(X[:, 0], X[:, 1], c='white')
            c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

            plt.axis('tight')
            plt.xlim((0, 1))
            plt.ylim((0, 1.5))
            plt.legend([b,c],
                       ["obs d'apprentissage",
                        
                        "anomalies detectees"],
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


# ---- Test sur la base de données Mouse ----

df3=pd.read_csv('../data/mouse-synthetic-data.txt', sep=' ')

showRawDatas(df3)

X = df3.values[:, [0,1]]

print 'Detection des anomalies par IsolationForest'
detectAnomaly(X,'isolationforest')

print 'Detection des anomalies par LOF'
detectAnomaly(X,'lof')

