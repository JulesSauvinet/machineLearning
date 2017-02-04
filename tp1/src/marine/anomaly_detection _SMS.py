# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import pandas as pd
import scipy

from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
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
def detectAnomaly(X,plot=True):
    n_samples = 200
    outliers_fraction = 0.25
    clusters_separation = [0, 1, 2]

    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,kernel="rbf", gamma=0.1),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(max_samples=n_samples,contamination=outliers_fraction,random_state=rng)
    }

    # Compare given classifiers under given settings
    xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1.5, 50))
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = -1

    # X is the our data and it contains 20 outliers to 500 data rows
    
    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(42)

        # Fit the model
        plt.figure(figsize=(10.8, 3.6))
        
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            
            # fit the data and tag outliers
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
            y_pred = clf.predict(X)
            n_errors = (y_pred != ground_truth).sum()
            
            # plot the levels lines and the points
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            subplot = plt.subplot(1, 3, i + 1)
            subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
            
            a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
            
            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
            
            b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
            c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')
            
            subplot.axis('tight')
            subplot.legend( [a.collections[0], b, c],
                            ['learned decision function', 'true inliers', 'true outliers'],
                            prop=matplotlib.font_manager.FontProperties(size=11),
                            loc='lower right')
            subplot.set_title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
            subplot.set_xlim((-7, 7))
            subplot.set_ylim((-7, 7))
        plt.subplots_adjust(0.04, 0.1, 0.96, 0.92, 0.1, 0.26)

    plt.show()


# Test sur la base de données Mouse

df3=pd.read_csv('../data/mouse-synthetic-data.txt', sep=' ')
showRawDatas(df3)
X = df3.values[:, [0,1]]
print 'Detection des anomalies'
detectAnomaly(X)
