import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import VotingClassifier

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.ensemble import VotingClassifier

from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from itertools import product
from sklearn.feature_selection import RFE
from time import sleep
from sklearn.naive_bayes import MultinomialNB
'''
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
'''
# reading the dataset
#dataset=pd.read_csv("/home/safia/drebinDataset/DatasetsFOrFlowAnalysis/labeledDatasetMixed2MobileTradWITHLABEL.csv") #remember to remove the first column
dataset=pd.read_csv("labeledDatasetMixed2MobileTradWITHLABEL.csv") #remember to remove the first column
data=dataset.as_matrix(columns=None)
np.random.shuffle(data)
class_names=["malicious", "benign"]
print("classes", class_names)

x=data[:,:-1] # input
y=data[:,-1:] # output
n_classes=y.shape[1]

#scaling the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)
print(rescaledX)



#feature selection - chi2
test = SelectKBest(score_func=chi2, k=27)
fit = test.fit(rescaledX, y)
X_new = fit.transform(rescaledX)
print("shape of x", x.shape)

'''
model = ExtraTreesClassifier()
rfe = RFE(model, 15)
fit = rfe.fit(X_new, y.ravel())
X_new=fit.transform(X_new)
'''
# feature selection- feature importance
clf1 = ExtraTreesClassifier()
clf1 = clf1.fit(rescaledX, y.ravel())
clf1.feature_importances_  
model = SelectFromModel(clf1, prefit=True)
#X_new = model.transform(rescaledX)

#sleep(5)

#print(data)


#x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=0)
#print(x_train, x_test, y_train, y_test)

#scaler = StandardScaler().fit(x)
#rescaledX = scaler.transform(x)
#binarizer = Binarizer(threshold=0.0).fit(x)
#rescaledX = binarizer.transform(x)
#scaler = Normalizer().fit(x)
#normalizedX = scaler.transform(x)
#test = SelectKBest(score_func=mutual_info_classif, k=15)



x_train, x_test, y_train, y_test=train_test_split(X_new,y,test_size=0.2, random_state=1)

#testFile=open("testFile","w")
#with testFile:
 # writer=csv.writer(testFile)
 # writer.writerows(x_test+y_test)

# MLP alssifier
clfMLP = MLPClassifier(solver='lbfgs', activation='tanh',alpha=1e-2, hidden_layer_sizes=(14,), random_state=1)
clfMLP=clfMLP.fit(x_train, y_train.ravel())
scoreMLP=clfMLP.score(x_test, y_test)
y_pred_ada=clfMLP.predict(x_test)
print(y_test[0:5],y_pred_ada[0:5])
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)
print("scoreMLP", scoreMLP)


from sklearn.metrics import precision_score

#x_train,x_val,y_train, y_val=train_test_split(x_train,y_train, test_size=0.2, random_state=1)
clf2 = ExtraTreesClassifier(max_features='auto', criterion='entropy',n_estimators=1000)
clf2 = clf2.fit(x_train,y_train.ravel())
y_pred_et=clf2.predict(x_test)
scoreExtraTC = clf2.score(x_test, y_test)
f1ET=f1_score(y_test, y_pred_ada) 
print("scoreRFExtreme: ",scoreExtraTC)
print("f1: ",f1ET)
p1=precision_score(y_test, y_pred_et) 
print("precision: ",p1)




print(y_test[0:5],y_pred_ada[0:5])
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
#scoreAda = (cnf_matrix[0][0] + cnf_matrix[1][0])/(cnf_matrix[0][0]+cnf_matrix[0][1] + cnf_matrix[1][0]+cnf_matrix[1][1])
scoreAda = accuracy_score(x_test, y_test.ravel())
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)
print("AccuracyAda", scoreAda)

#RandomForest classifier
clfR = RandomForestClassifier(max_features='auto', criterion='entropy',n_estimators=1000)
clfR = clfR.fit(x_train, y_train.ravel())
scoreRF=clfR.score(x_test, y_test)
y_pred_ada=clfR.predict(x_test)
print(y_test[0:5],y_pred_ada[0:5])
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("AccuracyRF", scoreRF)
print("specificity (1-FPR)", specificity)


#adaptive boost
bdt = AdaBoostClassifier( DecisionTreeClassifier(max_depth=19),
                         algorithm="SAMME.R",
                         n_estimators=1000)
bdt=bdt.fit(x_train, y_train.ravel())

y_pred_adap=bdt.predict(x_test)
f1ETada=f1_score(y_test, y_pred_adap) 
p1=precision_score(y_test, y_pred_adap) 
print("precision: ",p1)
scoreAda=bdt.score(x_test, y_test.ravel())
#xgboost - 94.26%
modelXGB = XGBClassifier(max_depth=27, n_estimators=1300, learning_rate=0.08,n_jobs=4, gamma=0.0001)
modelXGB=modelXGB.fit(x_train, y_train.ravel())


y_pred_xg = modelXGB.predict(x_test)
f1ETxg=f1_score(y_test, y_pred_xg) 
print("f1: ",f1ETxg)
p3=precision_score(y_test, y_pred_xg) 
print("precision: ",p3)
scoreXGB=modelXGB.score(x_test, y_test.ravel())
print("scoreXGB", scoreXGB)
predictions = [round(value) for value in y_pred_xg]
# evaluate predictions
accuracy = accuracy_score(y_test.ravel(), predictions)
print("AccuracymodelXGB: %.2f%%" % (accuracy * 100.0))

# voting algorithm
clff1 = KNeighborsClassifier(n_neighbors=23)
rt_lm = LogisticRegression()
eclf = VotingClassifier(estimators=[('lr', clff1), ('rf', clfR), ('gnb', rt_lm)], voting='soft')

for clf, label in zip([clff1, clfR, rt_lm, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X_new, y.ravel(), cv=10, scoring='precision')
    print("precision: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
#KNeighborsClassifier
bagging = BaggingClassifier(RandomForestClassifier(n_estimators=1000,random_state=40), n_estimators=100)
clfKNC=bagging.fit(x_train, y_train.ravel())
scoreKNC=clfKNC.score(x_test, y_test)

y_pred_bag=clfKNC.predict(x_test)
p2=precision_score(y_test, y_pred_bag) 
print("precision: ",p2)
f1ET=f1_score(y_test, y_pred_bag) 

print("f1: ",f1ET)
print("baggingKNC", scoreKNC)
print("x new shape", X_new.shape )
#scoreFeatureSelection = clf2.score(x_test, y_test)

#extremely randomizedTree
clfRFExtreme = RandomForestClassifier(n_estimators=600, max_depth=19,min_samples_split=5, random_state=0)
scores = cross_val_score(clfRFExtreme, X_new,y)
scoreRFExtreme=scores.mean() 

#GradientBoostingClassifier
params = {'n_estimators': 1300, 'max_depth': 27, 'subsample': 0.8,
          'learning_rate': 0.01, 'min_samples_leaf': 30, 'random_state': 60}
clfGB = ensemble.GradientBoostingClassifier(max_features='auto',loss='deviance',**params)
clfGB.fit(x_train, y_train.ravel())
acc = clfGB.score(x_test, y_test.ravel())
print("Accuracy: {:.4f}".format(acc))
y_pred_ada=clfGB.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("AccuracyGradientClassifier:", acc)
print("specificity (1-FPR)", specificity)



#linearRegression

lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train.ravel())
predictions = lm.predict(x_test)
scoreLR=model.score(x_test, y_test)
print("linear regression:", scoreLR)
print(predictions[0:5])


print("ADA boost: ",scoreAda)
print("RF: ",scoreRF)
#print("AdaBoost: ",scoreAB)
print("KNC: ",scoreKNC)
print("scoreRFExtreme: ",scoreRFExtreme)
print("MLP: ", scoreMLP)

#y_pred_ada=bdt.fit(x_train, y_train.ravel()).predict(x_test)
y_pred_ada=bdt.predict(x_test)

print(y_test[0:5],y_pred_ada[0:5])
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)

plt.scatter(y_test, y_pred_ada)
plt.xlabel("actual Values")
plt.ylabel("Predictions")

#SVC - not ensemble
svcc=SVC(gamma=2, C=0.5)
svcc=svcc.fit(x_train,y_train)
scoresvc=svcc.score(x_test, y_test)
print("SVCAccuracy: ", scoresvc)
y_pred_ada=svcc.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)

#logistic regression

rt_lm = LogisticRegression()
rt_lm=rt_lm.fit(x_train,y_train)
scoreLR=rt_lm.score(x_test, y_test)
print("LGAccuracy: ", scoreLR)
y_pred_ada=rt_lm.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)

# gaussian Naive-Bayes
gnb = GaussianNB()
gnb=gnb.fit(x_train,y_train)
scoreNB=gnb.score(x_test, y_test)
print("NBAccuracy: ", scoreNB)
y_pred_ada=gnb.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)

#multinomial Naive-bayes
mnb = MultinomialNB()
mnb=mnb.fit(x_train,y_train.ravel())
scoreMNB=mnb.score(x_test, y_test.ravel())
print("MNBAccuracy: ", scoreMNB)
y_pred_ada=mnb.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
print(cnf_matrix)
TPR=cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])
specificity=cnf_matrix[1][1]/(cnf_matrix[0][1]+cnf_matrix[1][1])
print("sensitivity (TPR)", TPR)
print("specificity (1-FPR)", specificity)

#stacking classifier
clff1 = KNeighborsClassifier(n_neighbors=23)
clff2 = RandomForestClassifier(n_estimators=200,random_state=40)
clff3 = GaussianNB()
lr = LogisticRegression()
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=19),
                             algorithm="SAMME.R",
                             n_estimators=100)
sclf = StackingClassifier(classifiers=[ clff1,clff2,bdt],use_probas=True, average_probas=False, 
                              meta_classifier=lr)
for clf, label in zip([clff1,clff2, sclf], 
                          ['KNN', 
                           'RF', 
                           'StackingClassifier']):
    
        scores = model_selection.cross_val_score(clf, x, y.ravel(), 
                                                  cv=10, scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f) [%s]" 
              % (scores.mean(), scores.std(), label))




'''
    
#voting algorithm
cclf1 = DecisionTreeClassifier(max_depth=4)
cclf2 = KNeighborsClassifier(n_neighbors=7)
cclf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', cclf1), ('knn', cclf2),
                                    ('svc', cclf3)],
                        voting='soft', weights=[2, 1, 2])

cclf1.fit(X_new, y.ravel())
cclf2.fit(X_new, y.ravel())
cclf3.fit(X_new, y.ravel())
eclf.fit(X_new, y.ravel())

# Plotting decision regions
x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(27, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [cclf1, cclf2, cclf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X_new[:, 0], X_new[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
'''    
'''
gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10,8))

for clf, lab, grd in zip([clff1, clff2, clff3, sclf], 
                         ['KNN', 
                          'Random Forest', 
                          'Naive Bayes',
                          'StackingClassifier'],
                          itertools.product([0, 1], repeat=2)):

    clf.fit(X_new, y.ravel().astype(np.integer))
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_new, y=y.ravel().astype(np.integer), clf=clf)
    plt.title(lab)
'''
#plt.show()
#fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_ada, pos_label=2)
#print(metrics.auc(fpr, tpr))
#--------------------------------------for AUC
'''
fpr=dict()
tpr=dict()
roc_auc=dict()
for l in range(n_classes):
  fpr[l], tpr[l]=roc_curve(y_test[:,l], y_pred_ada[:,l])
  roc_auc=auc(fpr[l], tpr[l])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

'''


'''
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)
score = clf.score(x_test, y_test)
print("Decision Tree: ",score)


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                         algorithm="SAMME",
                         n_estimators=200)
bdt=bdt.fit(x_train, y_train)
y_pred_ada=bdt.fit(x_train, y_train).predict(x_test)
scoreAda=bdt.score(x_test, y_test)
cnf_matrix = confusion_matrix(y_test, y_pred_ada)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
print("confusion matrix",cnf_matrix)
print("AdaBoosting: ", scoreAda)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

#plt.show()




rfc=RandomForestClassifier(max_depth=9, n_estimators=10, max_features=12)
rfc=rfc.fit(x_train, y_train)
scorerfc=rfc.score(x_test, y_test)
print("Random Forest", scorerfc)
print("Decision Tree: ",score)
print("AdaBoosting: ", scoreAda)
print("SVC: ", scoresvc)
#dot_data=tree.export_graphviz(clf, out_file=None)
#graph=graphviz.Source(dot_data)
#graph.render("sample")

#print(x)
#print(y)
'''