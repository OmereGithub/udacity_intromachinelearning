
### Import requisite Python Libraries
import sys
import pickle
import pandas as pd
from __future__ import division
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


### Import requisite created functions 
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

    
    

### Remove Outliers identified during data exploration
Outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for data_point in Outliers:
    data_dict.pop(data_point)


    
### Create Features and update data_dict

#create function that will compute fractions
def computeFraction(poi_messages, all_messages):
    '''
    given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    '''
    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0
    else:
        fraction = poi_messages/all_messages
    return fraction

for name in data_dict:
    
    data_point = data_dict[name]

    data_point["frac_from_poi"] = computeFraction(data_point["from_poi_to_this_person"], data_point["to_messages"])
    data_point["frac_to_poi"] = computeFraction(data_point["from_this_person_to_poi"], data_point["from_messages"])
    data_point["frac_shared_with_poi"] = computeFraction(data_point["shared_receipt_with_poi"], data_point["to_messages"])


    
### update feature list to be used for this study

ftrs2exclude = ['restricted_stock_deferred', 'director_fees', 'total_payments', 'total_stock_value', 'email_address' ]
                                                                        #features to exclude based on human intuition

features_list = ['poi'] #list must begin with 'poi'
all_features = data_dict["METTS MARK"].keys()

for feature in all_features:
    if feature != 'poi' and feature not in ftrs2exclude:
        features_list.append(feature)
        
        
        
### Extract features and labels from dataset for testing

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Select the best features 

k = 4 #we want only the 4 best features 
selector = SelectKBest(f_classif, k = k)
features = selector.fit_transform(features, labels)
scores = selector.scores_ 



### Employ Cross-Validation, and Parameter Tuning on selected Algorithm 

n_folds = 4.
skf = StratifiedKFold(labels, n_folds = n_folds, shuffle = True, random_state = 42)

parameters = {'n_neighbors':[3, 5, 7, 9], 
             'weights': ('uniform', 'distance'),
             'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}
neigh = KNeighborsClassifier()

precisions = []
recalls = []
f1s = []
accuracies = []

for train_idx, test_idx in skf: 
    
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
        
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
            
    clf = GridSearchCV(neigh, parameters, scoring = 'f1')
    
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    
    precisions.append(precision_score(labels_test, pred))
    recalls.append(recall_score(labels_test, pred))
    f1s.append(f1_score(labels_test, pred))
    accuracies.append(accuracy_score(labels_test, pred))

performance = {'accuracy': round(sum(accuracies)/n_folds, 4), 'precision': round(sum(precisions)/n_folds, 4),
              'recall': round(sum(recalls)/n_folds, 4), 'f1': round(sum(f1s)/n_folds, 4) }

print 'performance:', performance


### Dump classifier, dataset, and features_list for testing in tester.py

#extract the top features to be put into tester.py
features_list.remove('poi') #'poi' is a label and has no score

ftr_score = {}
for idx, value in enumerate(features_list):
    ftr_score[value] = scores[idx]

sorted_score = sorted(ftr_score.items(), key=lambda x: x[1], reverse=True)

feature_list = ['poi'] #list must begin with the label, 'poi'

ctr = 0
for item in sorted_score:
    feature_list.append(item[0])
    ctr += 1
    if ctr == k:
        break

my_dataset = data_dict
clf = clf
features_list = features_list

dump_classifier_and_data(clf, my_dataset, features_list)