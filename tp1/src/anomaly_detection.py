# coding=utf-8
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from text_mining import textMining


# -------------------------------------------------------------------------------------------------------------------------#
def runDetection(outliers, inliers, X, outs):
    outliers_fraction = 10. / X.shape[0]
    rng = np.random.RandomState(69)
    clusters_separation = [0]#, 1, 2]

    # les differents outils de detection d'anomalies
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction,
                                         kernel="rbf", gamma=0.1),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(n_estimators=1000,
                                            max_samples='auto',
                                            bootstrap=False,
                                            contamination=outliers_fraction,
                                            random_state=rng)}

    # Compare given classifiers under given settings
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.3, 100), np.linspace(-0.2, 1.9, 100))

    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(69)

        # Fit the model
        plt.figure(figsize=(10.8, 3.6))


        for i, (clf_name, clf) in enumerate(classifiers.items()):
            # fit the data and tag outliers
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            threshold = stats.scoreatpercentile(scores_pred,
                                                100 * outliers_fraction)
            y_pred = clf.predict(X)

            X_out_idx = np.where(y_pred == -1)[0]

            print clf_name
            print "True outliers     :",  outs
            print "Outliers detected :", X_out_idx

            # Calcul de la matrice de confusion a la main
            FP = len(np.intersect1d(outs, X_out_idx))
            FN = len(X_out_idx) - FP

            V = X.shape[0] - len(X_out_idx)
            VN = len(outs) - FP
            VP = V - VN

            n_errors = (VN + FN)

            print "Matrice de confusion"
            print " _________________________________", "\n"  \
                  "| P\R      Outliers    Inliers     |","\n"  \
                  "| -------------------------------- |","\n"  \
                  "| Outliers ", " "*4, FP, " "*8, FN, " "*4, "|","\n"  \
                  "| -------------------------------- |","\n"  \
                  "| Inliers  ", " "*4, VN, " "*7, VP, " "*3, "|","\n"  \
                  "|_________________________________ |","\n"  \

            # plot the levels lines and the points
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            subplot = plt.subplot(1, 3, i + 1)

            subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                             cmap=plt.cm.Blues_r)

            a = subplot.contour(xx, yy, Z, levels=[threshold],
                                linewidths=2, colors='red')

            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                             colors='orange')

            b = subplot.scatter(inliers[:, 0], inliers[:, 1], c='white')
            c = subplot.scatter(outliers[:, 0], outliers[:, 1], c='black')

            subplot.axis('tight')

            subplot.legend(
                [a.collections[0], b, c],
                ['learned decision function', 'true inliers', 'true outliers'],
                prop=matplotlib.font_manager.FontProperties(size=11),
                loc='upper left')

            subplot.set_title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
            subplot.set_xlim((-0.2, 1.3))
            subplot.set_ylim((-0.2, 1.9))

        plt.subplots_adjust(0.04, 0.1, 0.96, 0.92, 0.1, 0.26)

    plt.show()
# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
def runDetectionSMS(outliers, inliers, X, outs):
    outliers_fraction = 20. / X.shape[0]
    
    rng = np.random.RandomState(69)
    clusters_separation = [0]#, 1, 2]

    # les differents outils de detection d'anomalies
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95*outliers_fraction,kernel="rbf", gamma=0.1),
        #"Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(n_estimators=1000,max_samples='auto',bootstrap=False,
                                            contamination=outliers_fraction,random_state=rng)
    }

    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(69)

        for i, (clf_name, clf) in enumerate(classifiers.items()):
            # fit the data and tag outliers
            clf.fit(X)
            #scores_pred = clf.decision_function(X)
            #threshold = stats.scoreatpercentile(scores_pred,300*outliers_fraction)
            y_pred = clf.predict(X)

            X_out_idx = np.where(y_pred == -1)[0]

            outs2 = []
            for out in outs :
                for outS in out :
                    outs2.append(outS)
                
            print clf_name
            print "True outliers     :", outs2,"\n"
            print "Outliers detected :", X_out_idx,"\n"

            # Calcul de la matrice de confusion a la main
            FP = len(np.intersect1d(outs, X_out_idx))
            FN = len(X_out_idx) - FP

            V = X.shape[0] - len(X_out_idx)
            VN = len(outs) - FP
            VP = V - VN

            n_errors = (VN + FN)

            print "Matrice de confusion"
            print " _________________________________", "\n"  \
                  "| P\R      Outliers    Inliers     |","\n"  \
                  "| -------------------------------- |","\n"  \
                  "| Outliers ", " "*4, FP, " "*8, FN, " "*4, "|","\n"  \
                  "| -------------------------------- |","\n"  \
                  "| Inliers  ", " "*4, VN, " "*7, VP, " "*3, "|","\n"  \
                  "|_________________________________ |","\n"  \
            

# -------------------------------------------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------------------------------------------#
# Plot les donnees (coordonnees a deux dimensions)
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
# 2. Sur le	jeu	de données des SMS
def preProcessDatas(df):
    Z = df.values[:, [0, 1]]

    # extraction des SMS labelle Spam ou Ham
    Spam = Z[Z[:, 0] == 'spam']
    Ham = Z[Z[:, 0] == 'ham']

    # recuperer la moitie des donnees Ham aleatoirement
    halfHam = Ham[np.random.randint(0, Ham.shape[0], Ham.shape[0] / 2)]

    # recuperer 20 spams aleatoirement
    sampleSpam = Spam[np.random.randint(0, Spam.shape[0], 20)]

    # on concatene les donnees ham et spam
    datas = np.concatenate((halfHam, sampleSpam), axis=0)
    np.random.shuffle(datas)

    # on cree une table de notre resultat
    df = pd.DataFrame(datas)

    # on execute le code textMining
    X, Y = textMining(df, 0.5, 1, 2500, 750, False, True)
    Y = np.reshape(Y, (len(Y), 1))

    # le jeu de donnee sur lequel on va faire de la detection d'anomalie
    datas = np.concatenate((X, Y), axis=1)

    # les index des outliers
    outs = np.argwhere(datas[:, datas.shape[1] - 1] == 0)

    # on supprime la colonne avec la valeur cible Spam/Ham
    datas = scipy.delete(datas, datas.shape[1] - 1, 1)

    # on separe pour finir les donnees outliers des vraies donnees
    outliers = np.where(datas[:, datas.shape[1] - 1] == 0)
    inliers = np.where(datas[:, datas.shape[1] - 1] == 1)

    return outs, datas, outliers, inliers
# -------------------------------------------------------------------------------------------------------------------------#

def isolationForest(outs, datas, outliers, inliers) :
    clf = IsolationForest(n_estimators=500, max_samples='auto',
                          random_state=0, bootstrap=False, n_jobs=1,
                          contamination=(20. / datas.shape[0]) * 20)
    clf.fit(datas)
    y_pred = clf.predict(datas)
    scores = clf.decision_function(datas)

    X_out_idx = np.where(y_pred == -1)

    outs = np.transpose(outs)
    outs = outs[0]

    X_out_idx = X_out_idx[0]

    # Calcul de la matrice de confusion a la main
    FP = len(np.intersect1d(outs, X_out_idx))
    FN = len(X_out_idx) - FP

    V = datas.shape[0] - len(X_out_idx)
    VN = len(outs) - FP
    VP = V - VN

    print " Matrice de confusion"
    print " ______________________________", "\n"  \
          "| P\R      Spam        Ham     |","\n"  \
          "| ---------------------------- |","\n"  \
          "| Spam", " "*4, FP, " "*7, FN, " "*3, "|","\n"  \
          "| ---------------------------- |","\n"  \
          "| Ham ", " "*4, VN, " "*8, VP, " "*2, "|","\n"  \
          "|_____________________________ |","\n"  \
          
# -------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    # ************************************************************#
    # Test
    # 1. Sur la base de données Mouse
    print ""
    print "Detection d'anomalie sur les donnees de Mickey"
    print ""
    df=pd.read_csv('../data/mouse-synthetic-data.txt', sep=' ', header=None)
    X = df.values[:, [0,1]]

    outliers = X[0:10]
    inliers = X[11:len(X)]

    true_outs_idx = np.array([0,1,2,3,4,5,6,7,8,9])

    #runDetection(outliers, inliers, X, true_outs_idx)
    # ************************************************************#


    # ************************************************************#
    # Test
    # 2. Sur le	jeu de données des SMS
    print ""
    print "*"*75
    print ""
    print "Detection d'anomalie sur les donnees de SMS"
    print ""
    df = pd.read_csv('../data/SMSSpamCollection.data', sep='\t', header=None)
    true_outs_idx, datas, outliers, inliers = preProcessDatas(df)

    #print ' Detection avec Isolation Forest'
    #isolationForest(outs, datas, outliers, inliers)

    runDetectionSMS(outliers, inliers, datas, true_outs_idx)


    # ************************************************************#

