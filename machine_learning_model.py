# Check the versions of libraries

# Python version
import sys

# scipy
import scipy

# numpy
import numpy

# matplotlib
import matplotlib

# pandas
import pandas
import pandas as pd

# scikit-learn
import sklearn

import numpy as np


import pickle



# Load libraries

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy.stats import spearmanr
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import RepeatedKFold

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from imblearn.metrics import specificity_score

#import shap



final_score_labels_workbook = 'Enter your path\\Final_scores_labels_classifier_1_VUS.xlsx'



df_final_score_labels = pd.read_excel(final_score_labels_workbook)

print(df_final_score_labels.shape)



# Split-out validation dataset

array = df_final_score_labels.values

X = array[:,1:18]
y = array[:,18]
indexes = array[:,0]

print(X)
print(X.shape)
print(y)



final_score_labels_Clinvar_workbook = 'Enter your path\\Final_scores_labels_Clinvar_classifier_1_VUS.xlsx'


df_final_score_labels_Clinvar = pd.read_excel(final_score_labels_Clinvar_workbook)

print(df_final_score_labels_Clinvar.shape)



# Split-out validation dataset

array = df_final_score_labels_Clinvar.values

X_test = array[:,1:18]
Y_test = array[:,18]





# Spot Check Algorithms

models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LR', LogisticRegression(penalty='l2', C=1.0, solver='newton-cg', multi_class='ovr', class_weight='balanced')))

#models.append(('LR1', LogisticRegression(penalty='l2', C=1.0, solver='liblinear', multi_class='auto', class_weight='balanced')))
#models.append(('LRCV', LogisticRegressionCV(Cs=10, fit_intercept=True, cv= k_fold, dual=False, penalty='l2', solver='newton-cg', multi_class='multinomial', class_weight='balanced')))
#models.append(('LRCV', LogisticRegressionCV(Cs=10, fit_intercept=True, cv= k_fold, dual=False, penalty='l2', solver='liblinear', multi_class='auto', class_weight='balanced')))

#models.append(('LDA', LinearDiscriminantAnalysis(solver='svd', shrinkage=None)))

#models.append(('QDA', QuadraticDiscriminantAnalysis()))

#models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='uniform')))

#models.append(('CART', DecisionTreeClassifier()))

models.append(('BernoulliNB', BernoulliNB()))

##models.append(('CART', DecisionTreeClassifier(criterion='entropy', max_features='log2')))
#models.append(('NB', GaussianNB()))
#models.append(('SVM1', SVC(gamma='auto')))
#models.append(('Linear SVM', LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)))
#models.append(('RBF SVM', SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=None)))

#models.append(('SVM', SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)))

models.append(('RF', RandomForestClassifier(n_estimators=200, max_depth=None, max_features=11, random_state=0)))

models.append(('MLP', MLPClassifier(hidden_layer_sizes=(11), activation='relu', solver='lbfgs', alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, max_iter=400, random_state=1)))



X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)



# evaluate each model in turn

results = []
names = []
for name, model in models:
	
        kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=1)
        #kfold = RepeatedKFold(n_splits=3, n_repeats=20, random_state=1)
        cv_results = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=kfold, n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))



# Compare Algorithms

#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()



# Make predictions on validation dataset

names = ["RF", "MLP", "BernoulliNB", "LR", "LDA", "KNN", "CART", "SVM"]



models = [
RandomForestClassifier(n_estimators=200, max_depth=None, max_features=11, random_state=0),
MLPClassifier(hidden_layer_sizes=(11), activation='relu', solver='lbfgs', alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, max_iter=400, random_state=1),
BernoulliNB(), LogisticRegression(penalty='l2', C=1.0, solver='newton-cg', multi_class='ovr', class_weight='balanced'), LinearDiscriminantAnalysis(solver='svd', shrinkage=None), KNeighborsClassifier(n_neighbors=5, weights='uniform'), DecisionTreeClassifier(), SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)]






# iterate over classifiers
for name, model in zip(names, models):

    model.fit(X_train, Y_train)

    predictions = model.predict(X_validation)


# Evaluate predictions

    print('%s:' % (name))
    print("Accuracy score based on training set=%s" % (model.score(X_train, Y_train)))
    print("Accuracy score based on test set=%s" % (model.score(X_validation, Y_validation)))
    print("Accuracy score based on ClinVar dataset=%s" % (model.score(X_test, Y_test)))


#Get the confusion matrix
    cf_matrix = confusion_matrix(Y_validation, predictions)
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    print("specificity_score_micro=%s" % (specificity_score(Y_validation, predictions, labels=['Other variants', 'VUS'], average='micro')))
    print("specificity_score_macro=%s" % (specificity_score(Y_validation, predictions, labels=['Other variants', 'VUS'], average='macro')))
    print("specificity_score_weighted=%s" % (specificity_score(Y_validation, predictions, labels=['Other variants', 'VUS'], average='weighted')))
    print("specificity_score_each_class=%s" % (specificity_score(Y_validation, predictions, labels=['Other variants', 'VUS'], average=None)))



# save the model to disk
    #filename = 'finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print('%s:' % (name))
    #print("accuracy score_Clinvar=%s" % (result)


    predictions_Clinvar = model.predict(X_test)


#Get the confusion matrix
    cf_matrix_Clinvar = confusion_matrix(Y_test, predictions_Clinvar)
    print("Clinvar=%s" % (confusion_matrix(Y_test, predictions_Clinvar)))
    print("Clinvar=%s" % (classification_report(Y_test, predictions_Clinvar)))
    print("specificity_score_micro_Clinvar=%s" % (specificity_score(Y_test, predictions_Clinvar, labels=['Other variants', 'VUS'], average='micro')))
    print("specificity_score_macro=%s" % (specificity_score(Y_test, predictions_Clinvar, labels=['Other variants', 'VUS'], average='macro')))
    print("specificity_score_weighted=%s" % (specificity_score(Y_test, predictions_Clinvar, labels=['Other variants', 'VUS'], average='weighted')))
    print("specificity_score_each_class=%s" % (specificity_score(Y_test, predictions_Clinvar, labels=['Other variants', 'VUS'], average=None)))



# new instances where we do not know the answer

# show the inputs and predicted outputs
    #for i in range(len(index)):
    for i in range(len(X_validation)):
        print("Variant=%s, Predicted=%s" % (X_validation[i], predictions[i]))
        #print("Variant=%s, Predicted=%s" % (index[i], predictions[i]))

    #with open("Predictions_%s % (name).txt", "a") as f:
         #print(index, "Variant=%s, Predicted=%s" % (X_validation[i], predictions[i]), file=f)


# misclassified samples

    misclassifiedIndexes = []
    for index, label, predict in zip(indexes, Y_validation, predictions):
     if label != predict: 
      misclassifiedIndexes.append(index)
    print("misclassified samples=%s" % (misclassifiedIndexes))



###Confusion Matrix###

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', xticklabels=('Other variants', 'VUS'), yticklabels=('Other variants', 'VUS'), ax=ax)

    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    plt.title("Confusion Matrix dataset, " '%s' % (name))
    plt.show()



###Confusion Matrix-Clinvar###

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix_Clinvar.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix_Clinvar.flatten()/np.sum(cf_matrix_Clinvar)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cf_matrix_Clinvar, annot=labels, fmt='', cmap='Blues', xticklabels=('Other variants', 'VUS'), yticklabels=('Other variants', 'VUS'), ax=ax)

    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    plt.title("Confusion Matrix_ClinVar dataset, " '%s' % (name))
    plt.show()





###SHAP values###

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

    ##shap_values = explainer.shap_values(X_validation)
    ##explainer = shap.KernelExplainer(model.predict, X_train[1:50, :])
    #explainer = shap.KernelExplainer(model.predict_proba, X_train)
    #shap_values = explainer.shap_values(X_train)
    ##shap_obj = explainer(X_train)

    #feature_names = list(X.columns.values)
# summarize the effects of all the features
    ##shap.plots.beeswarm(shap_obj)
    ##plt.show()
    ##shap.summary_plot(shap_values, X_train[1:50, :], feature_names=('PVS1', 'BA1', 'BS1', 'PM2', 'PP3', 'BP4', 'BP7', 'PP2', 'BP1', 'PM1', 'PP5', 'BP6', 'PM4', 'BP3', 'PS4', 'PS1', 'PM5'), class_names=('Other variants', 'VUS'))
    #shap.summary_plot(shap_values, X_train, feature_names= feature_names, class_names=('Other variants', 'VUS'))
    #plt.xlim([0.00, 0.7])
    #plt.xticks(np.arange(0.00, 0.75, step=0.05))
    #plt.title("Feature importance analysis")
    ##plt.savefig('Shapvalues_%s.png' % (name))



    ##vals = shap_values.values
    ##feature_names = list(X.columns.values)
    #vals = shap_values
    #vals_abs = np.abs(vals)
    #val_mean = np.mean(vals_abs, axis=0)
    #val_final = np.mean(val_mean, axis=1)
    #feature_importance = pd.DataFrame(list(zip(feature_names, val_final)), columns=['features', 'importance'])
    ##feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)

    #file_name = 'SHAP Values %s.xlsx' % (name)
  
# saving the excel
    #feature_importance.to_excel(file_name)







###ROC CURVES####

# Binarize the output
y = label_binarize(y, classes=['Other variants', 'VUS'])
print(y)
n_classes = y.shape[1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


Y_test = label_binarize(Y_test, classes=['Other variants', 'VUS'])
n_classes = Y_test.shape[1]



# Learn to predict each class against the other

classifiers = [
    RandomForestClassifier(n_estimators=200, max_depth=None, max_features=11, random_state=0),
MLPClassifier(hidden_layer_sizes=(11), activation='relu', solver='lbfgs', alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, max_iter=400, random_state=1)]

#BernoulliNB()

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','roc_auc'])




# iterate over classifiers
for name, classifier in zip(names, classifiers):

    #cls = classifier.fit(X_train, Y_train)
    #y_score = cls.predict_proba(X_validation)[::,1]

    #classifier_ovr = OneVsRestClassifier(classifier)
    y_score = classifier.fit(X_train, Y_train.ravel()).predict_proba(X_validation)

    y_score_clinvar = classifier.fit(X_train, Y_train.ravel()).predict_proba(X_test)

# keep probabilities for the positive outcome only
    y_score = y_score[:, 1]
    y_score_clinvar = y_score_clinvar[:, 1]

    ##y_score = np.array(y_score_prob)
    ##y_score = np.transpose([pred[:, 1] for pred in y_score])
    ##y_score = classifier_ovr.fit(X_train, Y_train).decision_function(X_validation)
    ##y_score = y_score[:, 1]


    #print(roc_auc_score(Y_validation, y_score, average=None)) 
    print("roc_auc score=%s" % (roc_auc_score(Y_validation, y_score, average='micro')))

    print("roc_auc score_clinvar=%s" % (roc_auc_score(Y_test, y_score_clinvar, average='micro')))


# Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #for i in range(n_classes):
        #fpr[i], tpr[i], _ = roc_curve(Y_validation[:, i], y_score[:, i])
        #roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area#
    #fpr["micro"], tpr["micro"], _ = roc_curve(Y_validation, y_score)
    #roc_auc["micro"] = roc_auc_score(Y_validation, y_score)


    #Y_test = Y_test.map({'Other variants': 1, 'VUS': 0}).astype(int)
    
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_validation.ravel(), y_score.ravel())
    #fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score_clinvar.ravel())
    
    #fpr["micro"], tpr["micro"], _ = roc_curve(Y_test, y_score, pos_label=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



    result_table = result_table.append({'classifiers':classifier.__class__.__name__,
                                        'fpr':fpr["micro"], 
                                        'tpr':tpr["micro"], 
                                        'roc_auc':roc_auc["micro"]}, ignore_index=True)



# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)



# Plot ROC curve

    ##plt.figure(0).clf()
    ##plt.figure()

    ##disp = plot_roc_curve()



fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'], label="{}, AUC={:0.3f})".format(i, result_table.loc[i]['roc_auc']))


    #colors = cycle(['blue', 'red', 'green'])
    #for i, color in zip(range(n_classes), colors):
    ##for i in range(n_classes):
        #plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label="ROC curve of class {0}, %s (area = {1:0.2f})".format(i, roc_auc[i]) % (name),)
        ##plt.plot(fpr[i], tpr[i], label="ROC curve of class {'B/LB'} (area = {1:0.2f})".format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel('False Positive Rate')

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel('True Positive Rate')
plt.title('ROC curves, Test set', fontweight='bold')
plt.legend(loc="lower right")
plt.show()









###Classifier_2(B/LB_P/LP)###



final_score_labels_workbook_2 = 'Enter your path\\Final_scores_labels_classifier_2_B_P.xlsx'



df_final_score_labels_2 = pd.read_excel(final_score_labels_workbook_2)

print(df_final_score_labels_2.shape)



# Split-out validation dataset

array = df_final_score_labels_2.values

Z = array[:,1:18]
w = array[:,18]
indexes = array[:,0]


#k_fold= RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=1)
#k_fold= RepeatedKFold(n_splits=3, n_repeats=20, random_state=1)


print(Z)
print(Z.shape)
print(w)



final_score_labels_Clinvar_workbook_2 = 'Enter your path\\Final_scores_labels_Clinvar_classifier_2_B_P.xlsx'


df_final_score_labels_Clinvar_2 = pd.read_excel(final_score_labels_Clinvar_workbook_2)

print(df_final_score_labels_Clinvar_2.shape)



# Split-out validation dataset

array = df_final_score_labels_Clinvar_2.values

Z_test = array[:,1:18]
W_test = array[:,18]




# Spot Check Algorithms

models = []
##models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LR', LogisticRegression(penalty='l2', C=1.0, solver='newton-cg', multi_class='ovr', class_weight='balanced')))

#models.append(('LR1', LogisticRegression(penalty='l2', C=1.0, solver='liblinear', multi_class='auto', class_weight='balanced')))
##models.append(('LRCV', LogisticRegressionCV(Cs=10, fit_intercept=True, cv= k_fold, dual=False, penalty='l2', solver='newton-cg', multi_class='multinomial', class_weight='balanced')))
#models.append(('LRCV', LogisticRegressionCV(Cs=10, fit_intercept=True, cv= k_fold, dual=False, penalty='l2', solver='liblinear', multi_class='auto', class_weight='balanced')))
#models.append(('LDA', LinearDiscriminantAnalysis(solver='svd', shrinkage=None)))
#models.append(('QDA', QuadraticDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='uniform')))
#models.append(('CART', DecisionTreeClassifier()))

models.append(('BernoulliNB', BernoulliNB()))

##models.append(('CART', DecisionTreeClassifier(criterion='entropy', max_features='log2')))
#models.append(('NB', GaussianNB()))
#models.append(('SVM1', SVC(gamma='auto')))
#models.append(('Linear SVM', LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)))
#models.append(('RBF SVM', SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=None)))

#models.append(('SVM', SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)))

models.append(('RF', RandomForestClassifier(n_estimators=200, max_depth=None, max_features=11, random_state=0)))

#models.append(('MLP', MLPClassifier(hidden_layer_sizes=(11), activation='relu', solver='lbfgs', alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, max_iter=400, random_state=1)))


Z_train, Z_validation, W_train, W_validation = train_test_split(Z, w, test_size=0.20, random_state=1)




# evaluate each model in turn

results = []
names = []
for name, model in models:
	##kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
        kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=20, random_state=1)
        #kfold = RepeatedKFold(n_splits=3, n_repeats=20, random_state=1)
        cv_results = cross_val_score(model, Z_train, W_train, scoring='accuracy', cv=kfold, n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))



# Compare Algorithms

#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()



# Make predictions on validation dataset

names = ["LR", "LDA", "KNN", "CART", "BernoulliNB", "SVM", "RF", "MLP"]



models = [
LogisticRegression(penalty='l2', C=1.0, solver='newton-cg', multi_class='ovr', class_weight='balanced'), LinearDiscriminantAnalysis(solver='svd', shrinkage=None), KNeighborsClassifier(n_neighbors=5, weights='uniform'), DecisionTreeClassifier(), BernoulliNB(), SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None), RandomForestClassifier(n_estimators=200, max_depth=None, max_features=11, random_state=0), MLPClassifier(hidden_layer_sizes=(11), activation='relu', solver='lbfgs', alpha=0.0001, learning_rate='constant', learning_rate_init=0.001, max_iter=400, random_state=1)]






# iterate over classifiers
for name, model in zip(names, models):

    model.fit(Z_train, W_train)

    #Z_validation = scaler.transform(Z_validation)
    predictions = model.predict(Z_validation)

    predictions_Clinvar = model.predict(Z_test)

# Evaluate predictions

    print('%s:' % (name))
    print(model.score(Z_train, W_train))
    print(model.score(Z_validation, W_validation))

    print(model.score(Z_test, W_test))



#Get the confusion matrix
    cf_matrix_2 = confusion_matrix(W_validation, predictions)
    print(confusion_matrix(W_validation, predictions))
    print(classification_report(W_validation, predictions))

    print(specificity_score(W_validation, predictions, labels=['B/LB', 'P/LP'], average='micro'))
    print(specificity_score(W_validation, predictions, labels=['B/LB', 'P/LP'], average='macro'))
    print(specificity_score(W_validation, predictions, labels=['B/LB', 'P/LP'], average='weighted'))
    print(specificity_score(W_validation, predictions, labels=['B/LB', 'P/LP'], average=None))



#Get the confusion matrix_Clinvar
    cf_matrix_Clinvar_2 = confusion_matrix(W_test, predictions_Clinvar)
    print(confusion_matrix(W_test, predictions_Clinvar))
    print(classification_report(W_test, predictions_Clinvar))

    print(specificity_score(W_test, predictions_Clinvar, labels=['B/LB', 'P/LP'], average='micro'))
    print(specificity_score(W_test, predictions_Clinvar, labels=['B/LB', 'P/LP'], average='macro'))
    print(specificity_score(W_test, predictions_Clinvar, labels=['B/LB', 'P/LP'], average='weighted'))
    print(specificity_score(W_test, predictions_Clinvar, labels=['B/LB', 'P/LP'], average=None))



# new instances where we do not know the answer

# show the inputs and predicted outputs
    
    for i in range(len(Z_validation)):
        print("Variant=%s, Predicted=%s" % (Z_validation[i], predictions[i]))



# misclassified samples

    misclassifiedIndexes = []
    for index, label, predict in zip(indexes, W_validation, predictions):
     if label != predict: 
      misclassifiedIndexes.append(index)
    print("misclassified samples=%s" % (misclassifiedIndexes))



###Confusion Matrix###

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix_2.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix_2.flatten()/np.sum(cf_matrix_2)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cf_matrix_2, annot=labels, fmt='', cmap='Blues', xticklabels=('B/LB', 'P/LP'), yticklabels=('B/LB', 'P/LP'), ax=ax)

    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    plt.title("Confusion Matrix dataset, " '%s' % (name))
    plt.show()



###Confusion Matrix_Clinvar###

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix_Clinvar_2.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix_Clinvar_2.flatten()/np.sum(cf_matrix_Clinvar_2)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cf_matrix_Clinvar_2, annot=labels, fmt='', cmap='Blues', xticklabels=('B/LB', 'P/LP'), yticklabels=('B/LB', 'P/LP'), ax=ax)

    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    plt.title("Confusion Matrix_ClinVar dataset, " '%s' % (name))
    plt.show()




###ROC CURVES####

# Binarize the output
w = label_binarize(w, classes=['B/LB', 'P/LP'])
print(w)
n_classes = w.shape[1]

Z_train, Z_validation, W_train, W_validation = train_test_split(Z, w, test_size=0.20, random_state=1)


W_test = label_binarize(W_test, classes=['B/LB', 'P/LP'])
#print(w)
n_classes = W_test.shape[1]



# Learn to predict each class against the other

classifiers = [
LogisticRegression(penalty='l2', C=1.0, solver='newton-cg', multi_class='ovr', class_weight='balanced'),
BernoulliNB()]



# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','roc_auc'])




# iterate over classifiers
for name, classifier in zip(names, classifiers):

    #cls = classifier.fit(X_train, Y_train)
    #y_score = cls.predict_proba(X_validation)[::,1]

    #classifier_ovr = OneVsRestClassifier(classifier)

    w_score = classifier.fit(Z_train, W_train.ravel()).predict_proba(Z_validation)
    w_score_clinvar = classifier.fit(Z_train, W_train.ravel()).predict_proba(Z_test)

# keep probabilities for the positive outcome only
    w_score = w_score[:, 1]
    w_score_clinvar = w_score_clinvar[:, 1]

    ##y_score = np.array(y_score_prob)
    ##y_score = np.transpose([pred[:, 1] for pred in y_score])
    ##y_score = classifier_ovr.fit(X_train, Y_train).decision_function(X_validation)
    ##y_score = y_score[:, 1]


    #print(roc_auc_score(W_validation, w_score, average=None)) 
    print(roc_auc_score(W_validation, w_score, average='micro'))
    print(roc_auc_score(W_test, w_score_clinvar, average='micro'))
    #print(macro_roc_auc_ovr = roc_auc_score(W_validation, w_score, multi_class="ovr", average="macro"))
    #print(weighted_roc_auc_ovr = roc_auc_score(W_validation, w_score, multi_class="ovr", average="weighted"))




# Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #for i in range(n_classes):
        #fpr[i], tpr[i], _ = roc_curve(Y_validation[:, i], y_score[:, i])
        #roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area#
    #fpr["micro"], tpr["micro"], _ = roc_curve(Y_validation, y_score)
    #roc_auc["micro"] = roc_auc_score(Y_validation, y_score)

    #W_test = W_test.map({'B/LB': 1, 'P/LP': 0}).astype(int)

    fpr["micro"], tpr["micro"], _ = roc_curve(W_validation.ravel(), w_score.ravel())

    #fpr["micro"], tpr["micro"], _ = roc_curve(W_test, w_score, pos_label=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



    result_table = result_table.append({'classifiers':classifier.__class__.__name__,
                                        'fpr':fpr["micro"], 
                                        'tpr':tpr["micro"], 
                                        'roc_auc':roc_auc["micro"]}, ignore_index=True)



# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)



# Plot ROC curve

    ##plt.figure(0).clf()
    ##plt.figure()

    ##disp = plot_roc_curve()



fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'], label="{}, AUC={:0.3f})".format(i, result_table.loc[i]['roc_auc']))


    #colors = cycle(['blue', 'red', 'green'])
    #for i, color in zip(range(n_classes), colors):
    ##for i in range(n_classes):
        #plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label="ROC curve of class {0}, %s (area = {1:0.2f})".format(i, roc_auc[i]) % (name),)
        ##plt.plot(fpr[i], tpr[i], label="ROC curve of class {'B/LB'} (area = {1:0.2f})".format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel('False Positive Rate')

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel('True Positive Rate')
plt.title('ROC curves, Test set', fontweight='bold')
plt.legend(loc="lower right")
plt.show()








