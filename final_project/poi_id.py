#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features

##list of features
full_list = ['salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'from_messages',
 'other',
 'from_this_person_to_poi',
 'director_fees',
 'deferred_income',
 'long_term_incentive',
 'from_poi_to_this_person']


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### Task 2: Remove outliers

###looping through dataset to find outliers
###especially entries with features with over
###15 NaN features
nanCount = {}
for each in data_dict:
    for eachkey in data_dict[each].keys():
        if data_dict[each][eachkey] == "NaN":
            if each in nanCount:
                nanCount[each] += 1
            else:
                nanCount[each] = 1
for each in nanCount:
    if nanCount[each] > 15 or each == "TOTAL":
        data_dict.pop(each, 0)

my_dataset = data_dict
my_features_list = features_list + full_list

###### Two new features createdCreating two new features.
message_lists = ['from_poi_to_this_person', 'from_this_person_to_poi',
'to_messages', 'from_messages']
for each in data_dict:
    data_dict[each]["messageratio"] = 0
    for field in message_lists:
        if data_dict[each][field] == "NaN":
            data_dict[each]["messageratio"] = "NaN"
            continue
        else:
            total_messages = float(data_dict[each]['to_messages'] + data_dict[each]['from_messages'])
            poi_messages = float(data_dict[each]['from_this_person_to_poi'] + data_dict[each]['from_poi_to_this_person'])
            data_dict[each]["messageratio"] = float(poi_messages/total_messages)

my_features_list += ["messageratio"]

financial_list = ["bonus", "salary"]
for each in data_dict:
    data_dict[each]["bonussalary"] = 0
    if data_dict[each][financial_list[0]] == "NaN" or data_dict[each][financial_list[1]]:
        data_dict[each]["bonussalary"] = "NaN"
        continue
    else:
        data_dict[each]["bonussalary"] = data_dict[each]["bonus"] + data_dict[each]["salary"]

#my_features_list += ["bonussalary"]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile

kper = SelectKBest(k=3)
pca = PCA(n_components=15)


###############################################
##Using Gridsearch for param tuning with Decision tree classifier
# from sklearn import grid_search
# from sklearn.tree import DecisionTreeClassifier
# param = {"criterion": ("gini", "entropy"),
#          "min_samples_split": [1, 100]}
#
# classifier = GridSearchCV(DecisionTreeClassifier(), param)
# clf = Pipeline([('scaling', MinMaxScaler()), ('dim_red', pca), ('f_select', kper), ('classifier', classifier)])
#####################################################


#svc = SVC(C=50, kernel='rbf', gamma=2.0)
#adaboost = AdaBoostClassifier(n_estimators=70)

###Passing all scaler, pca, feature selection and clasifier through pipeline.
clf = Pipeline([('Mscaling', MinMaxScaler()), ('dim_red', pca), ('f_select', kper), ('classifier', GaussianNB())])

#pipe = Pipeline([('scaling', MinMaxScaler()), ('dim_red', pca), ('classifier', adaboost)])
# Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#################################################################################################################
###################################################################################################

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

##fitting classifier and printint metric scores using train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# clf.fit(x_train, y_train)
# pred = clf.predict(x_test)
# print "Accuracy: ", accuracy_score(pred, y_test)
# print "Precision: ",  precision_score(pred, y_test)
# print "Recall: ",  recall_score(pred, y_test)

#print classifier.best_params_

#####Checing whether fitting to
#####either training set or all of features
####returns the same pca explained var ratio. It does.
'''
pca_features = pca.fit(features)
pca_train = pca.fit(x_train)

print pca.explained_variance_ratio_
print "---/n"
for x in range(len(pca.components_[0])):
    print "Feature", x+1, pca.components_[0][x], my_features_list[x+1]

print max(pca.components_[0])
##Uncomment to test

if np.any([pca_features.explained_variance_ratio_, pca_train.explained_variance_ratio_]):
    print "true"
else:
    print "false"
'''
#######################################################################################################
################################################################################################################

test_classifier(clf, my_dataset, my_features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)