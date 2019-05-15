#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:12:18 2018

@author: safia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:14:21 2018

@author: safia
"""

from xgboost import XGBClassifier


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import learning_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


dataset=pd.read_csv("/home/safia/drebinDataset/DatasetsFOrFlowAnalysis/labeledDatasetMixed2MobileTradWITHLABEL.csv") #remember to remove the first column
data=dataset.as_matrix(columns=None)
np.random.shuffle(data)
X=data[:,:-1] # input
y=data[:,-1:] # output
n_classes=y.shape[1]
y=y.ravel()
class_names=["malicious", "benign"]

numeric_data_best = SelectKBest(f_classif, k=27).fit_transform(X, y)

model_rfc = RandomForestClassifier(max_features='auto', criterion='entropy',n_estimators=1000)
model_knc = KNeighborsClassifier(n_neighbors = 18) 
model_lr = LogisticRegression(penalty='l1', tol=0.01) 
model_gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=900)
#model_svc = svm.SVC() 
model_xgb = XGBClassifier(max_depth=27, n_estimators=1300, learning_rate=0.08,n_jobs=4, gamma=0.0001)
model_svc = SVC(kernel='rbf', random_state=0)
model_nv= GaussianNB()
model_ada = AdaBoostClassifier( DecisionTreeClassifier(max_depth=19),algorithm="SAMME.R", n_estimators=1000)
model_xtc= ExtraTreesClassifier(max_features='auto', criterion='entropy',n_estimators=1000)
model_vc = VotingClassifier(estimators=[('xgboost', model_xgb), ('rf', model_rfc), ('gnb', model_gb)], voting='soft')
model_bagC = BaggingClassifier(RandomForestClassifier(n_estimators=100,random_state=40), n_estimators=100)

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = model_selection.train_test_split(numeric_data_best, y, test_size=0.25) 
results = {}
kfold = 5

'''
results['Bagging'] = model_selection.cross_val_score(model_bagC, numeric_data_best, y, cv = kfold,scoring='precision').mean()
results['ExtraTree'] = model_selection.cross_val_score(model_xtc, numeric_data_best, y, cv = kfold,scoring='precision').mean()
results['AdaBoost'] = model_selection.cross_val_score(model_ada, numeric_data_best, y, cv = kfold,scoring='precision').mean()
results['XGB'] = model_selection.cross_val_score(model_xgb, numeric_data_best, y, cv = kfold,scoring='precision').mean()
#results['StackingClassifier'] = model_selection.cross_val_score(model_stack, numeric_data_best, y, cv = kfold).mean()
results['Voting']=model_selection.cross_val_score(model_vc, numeric_data_best, y, cv = kfold,scoring='precision').mean()
results['RandomForest'] = model_selection.cross_val_score(model_rfc, numeric_data_best, y, cv=kfold).mean()
'''
results['RandomForestClassifier_best_params'] = model_selection.cross_val_score(model_rfc, numeric_data_best, y, cv=kfold).mean()
results['KNeighborsClassifier_best_params'] = model_selection.cross_val_score(model_knc, numeric_data_best, y, cv=kfold).mean()
results['LogisticRegression_best_params'] = model_selection.cross_val_score(model_lr, numeric_data_best, y, cv = kfold).mean()
results['GradientBoosting_best_params'] = model_selection.cross_val_score(model_gb, numeric_data_best, y, cv = kfold).mean()
results['SVC_best_params'] = model_selection.cross_val_score(model_svc, numeric_data_best, y, cv = kfold).mean()
results['XGB_best_params'] = model_selection.cross_val_score(model_xgb, numeric_data_best, y, cv = kfold).mean()
results['AdaBoost_best_params'] = model_selection.cross_val_score(model_ada, numeric_data_best, y, cv = kfold).mean()
results['ExtraTreeClassifier_best_params'] = model_selection.cross_val_score(model_xtc, numeric_data_best, y, cv = kfold).mean()
results['BaggingClassifier_best_params'] = model_selection.cross_val_score(model_bagC, numeric_data_best, y, cv = kfold).mean()
results['VotingClassifier_best_params']=model_selection.cross_val_score(model_vc, numeric_data_best, y, cv = kfold).mean()


'''
results['RandomForestClassifier_all_params'] = model_selection.cross_val_score(model_rfc, X, y, cv=kfold).mean()
results['KNeighborsClassifier_all_params'] = model_selection.cross_val_score(model_knc, X, y, cv=kfold).mean()
results['LogisticRegression_all_params'] = model_selection.cross_val_score(model_lr, X, y, cv = kfold).mean()
results['GradientBoosting_all_params'] = model_selection.cross_val_score(model_gb, X, y, cv = kfold).mean()
results['SVC_all_params'] = model_selection.cross_val_score(model_svc, X, y, cv = kfold).mean()
results['XGB_all_params'] = model_selection.cross_val_score(model_xgb, X, y, cv = kfold).mean()
results['AdaBoost_all_params'] = model_selection.cross_val_score(model_ada, X, y, cv = kfold).mean()
results['ExtraTreeClassifier_all_params'] = model_selection.cross_val_score(model_xtc, X, y, cv = kfold).mean()
results['BaggingClassifier_all_params'] = model_selection.cross_val_score(model_bagC, X, y, cv = kfold).mean()
results['VotingClassifier_all_params']=model_selection.cross_val_score(model_vc, X, y, cv = kfold).mean()
'''
print(results)
#plt.axvline(x=0.22058956)

#plt.add_subplot(1,1,1)
plt.bar(range(len(results)), results.values(), align='center')
#plt.yticks(np.arange(0.0, 1.0, 0.05))
#plt.yticks([0.7, 0.8, 0.9, 1.0])
#barlist[0].set_color('r')
#barlist[1].set_color('g')
#barlist[2].set_color('b')
plt.xticks(range(len(results)), list(results.keys()), rotation='vertical')
plt.show()

roc_train_all, roc_test_all, roc_train_all_class, roc_test_all_class = model_selection.train_test_split(X, y, test_size=0.25) 
roc_train_best, roc_test_best, roc_train_best_class, roc_test_best_class = model_selection.train_test_split(numeric_data_best, y, test_size=0.25) 

models = [
    {
        'label' : 'GradientBoosting_best_params',
        'model': model_gb,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,                
    },
    {
        'label' : 'RandomForestClassifier_best_params',
        'model': model_rfc,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },
    {
        'label' : 'XGB_best_params',
        'model': model_gb,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },    
    {
        'label' : 'SVC_best_params',
        'model': model_svc,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },        
    {
        'label' : 'KNeighborsClassifier_all_params',
        'model': model_knc,
        'roc_train': roc_train_all,
        'roc_test': roc_test_all,
        'roc_train_class': roc_train_all_class,        
        'roc_test_class': roc_test_all_class,        
    },
    {
        'label' : 'LogisticRegression_all_params',
        'model': model_knc,
        'roc_train': roc_train_all,
        'roc_test': roc_test_all,
        'roc_train_class': roc_train_all_class,        
        'roc_test_class': roc_test_all_class,        
    },
    {
        'label' : 'AdaBoost_best_params',
        'model': model_ada,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },
    {
        'label' : 'ExtraTreeClassifier_best_params',
        'model': model_xtc,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },
    {
        'label' : 'BaggingClassifier_best_params',
        'model': model_bagC,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },
    {
        'label' : 'VotingClassifier_best_params',
        'model': model_vc,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    }            
]

plt.clf()
plt.figure(figsize=(10,8))

#AUC curve
for m in models:
    m['model'].probability = True
    probas = m['model'].fit(m['roc_train'], m['roc_train_class']).predict_proba(m['roc_test'])
    fpr, tpr, thresholds = roc_curve(m['roc_test_class'], probas[:, 1])
    outp=[fpr,tpr];
    roc_auc  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()


#box plot
models_box=[]
models_box.append(('RF', RandomForestClassifier(max_features='auto', criterion='entropy',n_estimators=1000)))
models_box.append(('KNN', KNeighborsClassifier(n_neighbors = 18)))
models_box.append(('LR', LogisticRegression(penalty='l1', tol=0.01) ))
models_box.append(('GB', GradientBoostingClassifier(learning_rate=0.1, n_estimators=900)))
models_box.append(('SVC', SVC(kernel='rbf', random_state=0)))
models_box.append(('XGB', XGBClassifier(max_depth=27, n_estimators=1300, learning_rate=0.08,n_jobs=4, gamma=0.0001)))
models_box.append(('AB', AdaBoostClassifier( DecisionTreeClassifier(max_depth=19),algorithm="SAMME.R", n_estimators=1000)))
models_box.append(('ETC', ExtraTreesClassifier(max_features='auto', criterion='entropy',n_estimators=1000)))
models_box.append(('BC', BaggingClassifier(RandomForestClassifier(n_estimators=100,random_state=40), n_estimators=100)))
models_box.append(('VC', VotingClassifier(estimators=[('xgboost', model_xgb), ('rf', model_rfc), ('gnb', model_gb)], voting='soft')))
seed=7
results_box = []
names = []
scoring = 'accuracy'
for name, model in models_box:
    	kfold = model_selection.KFold(n_splits=10, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, numeric_data_best, y, cv=kfold, scoring=scoring)
    	results_box.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(10,8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_box)
ax.set_xticklabels(names)
plt.show()



#confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#y_pred_xgb=model_knc.fit(roc_train_best, roc_train_best_class.ravel()).predict(roc_test_best)
y_pred_xgb=model_nv.fit(roc_train_best, roc_train_best_class.ravel()).predict(roc_test_best)

cnf_matrix = confusion_matrix(roc_test_best_class, y_pred_xgb)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
print("confusion matrix",cnf_matrix)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for Naive Bayes')

plt.show()

#Learning curve XGBoost
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves for XGBoost"
plot_learning_curve(model_xgb, title, numeric_data_best, y, ylim=(0.0, 1.0), cv=kfold, n_jobs=4)
plt.show()

#caliberation curve

def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(roc_train_best, roc_train_best_class)
        y_pred = clf.predict(roc_test_best)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(roc_test_best)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(roc_test_best)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(roc_test_best_class, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(roc_test_best_class, y_pred))
        print("\tRecall: %1.3f" % recall_score(roc_test_best_class, y_pred))
        print("\tF1: %1.3f\n" % f1_score(roc_test_best_class, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(roc_test_best_class, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

# Plot calibration curve for Linear SVC
plot_calibration_curve(RandomForestClassifier(max_features='auto', criterion='entropy',n_estimators=1000), "RF", 2)

plt.show()