# coding=utf-8
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, tree

# -------------------------------------------------------------------------------------------------------------------------#
#Definition des classifieurs dans un dictionnaire
clf_init = None
clfs =	{
    #Naive Bayes Classifier
    'NBS' : GaussianNB(),

    #Random Forest
    'RF':   RandomForestClassifier(n_estimators=20),

    #K plus proches voisins
    'KNN':  KNeighborsClassifier(n_neighbors=10,  weights='uniform', algorithm='auto', p=2, metric='minkowski'),

    #Arbres de décision CART
    'CART': tree.DecisionTreeClassifier(min_samples_split=50, random_state=99,criterion='gini'),

    #Adaboost avec arbre de décision
    'ADAB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,random_state=99,criterion='gini'),algorithm="SAMME",n_estimators=50),

    # MLP perceptron multi-couches,
    'MLP' : MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=3, learning_rate = 'adaptive'),

    #Gradient boosting classifier
    'GBC' : GradientBoostingClassifier( loss='deviance', learning_rate=0.1, n_estimators=10, subsample=0.3,min_samples_split=2,
                                        min_samples_leaf=1, max_depth=1, init=clf_init,random_state=1, max_features=None, verbose=0)
}

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
        cv_acc = cross_val_score(clfs[clf], credit['data'], credit['target'], scoring='accuracy', cv=kf)  # pour    le	calcul	de	l’accuracy
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
