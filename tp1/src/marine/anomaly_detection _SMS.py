# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from prediction.lof import outliers
from text_mining import textMining

# Test Sur le jeu de données des SMS
df2 = pd.read_csv('../data/SMSSpamCollection.data', sep='\t')

#Preparation des donnees
#representation	SVD des SMS et colonne Spam/Ham associee pour chaque SMS
X, Y = textMining(df2, 1.0, 10, 500, 50, False, True)
#print np.shape(X)

#concatenation de la representation SVD des textes SMS et la categorie Spam/Ham
Y = np.reshape(Y, (len(Y), 1))
Z = np.concatenate((X, Y), axis=1)

#extraction des SMS labelle Spam ou Ham
Spam = Z[Z[:, Z.shape[1]-1] == 0]
Ham =  Z[Z[:, Z.shape[1]-1] == 1]

#le jeu de donnee sur lequel on va faire de la detection d'anomalie
datas = np.concatenate((Ham,Spam), axis=0)

#les index des outliers
outs = np.argwhere(datas[:, datas.shape[1]-1] == 0)

#on supprime la colonne avec la valeur cible Spam/Ham
datas = scipy.delete(datas, datas.shape[1]-1, 1)

print "Il y a dans les donnees :"
print "-",np.shape(datas)[0], "Spam"
print "-",np.shape(datas)[1]," "*1,"Ham\n"
#on fit le modele
clf = IsolationForest(n_estimators=100, max_samples='auto', random_state=0, bootstrap=True,n_jobs=1, contamination = 0.007)
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
      "| P\R      Ham        Spam     |","\n"  \
      "| ---------------------------- |","\n"  \
      "| Ham ", " "*3, VP, " "*7, FP, " "*4, "|","\n"  \
      "| ---------------------------- |","\n"  \
      "| Spam", " "*4, FN, " "*7, VN, " "*3, "|","\n"  \
      "|_____________________________ |","\n"  \

# -------------------------------------------------------------------------------------------------------------------------#

