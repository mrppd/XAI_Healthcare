# -*- coding: utf-8 -*-
"""
Created on Mon May  8 00:56:07 2023

@author: prona
"""

import os
from os.path import isfile, join
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from datetime import date 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics

from imblearn.over_sampling import RandomOverSampler
from collections import Counter



#def merge_and_apply_RF(ma=30, ftype='max', dropna=False):
ma=31
ftype='max'
dropna=True
random_state = 7
ii = 0
p_list = []
priority = ['female']

df_all_new_ = pd.read_csv('hourly_split_v5.csv')  

selected_cols = ['temp_max', 'temp_min', 'temp_avg',
    'temp_median', 'pulse_max', 'pulse_min',
    'pulse_avg', 'pulse_median', 'leukozyten_max',
    'leukozyten_min', 'leukozyten_avg', 'leukozyten_median', 
    'respiration_max', 'respiration_min', 'respiration_avg',
    'respiration_median', 'pressure_systole_max',
    'pressure_systole_min', 'pressure_systole_avg',
    'pressure_systole_median', 'pressure_diastol_max',
    'pressure_diastol_min', 'pressure_diastol_avg',
    'pressure_diastol_median',
    'gender', 'Entscheidung', 'age']

df_all_new_sel = df_all_new_[selected_cols]


df_all_new_sel_enc = pd.get_dummies(df_all_new_sel.gender)
df_all_new_sel = df_all_new_sel.drop(['gender'], axis=1)
df_all_new_sel = pd.concat([df_all_new_sel, df_all_new_sel_enc], axis=1)    

if ii>0:
    selected_cols = priority.copy()
    selected_cols.append('Entscheidung')
    df_all_new_sel = df_all_new_sel[selected_cols]

print('Original data: ',df_all_new_sel.drop(['Entscheidung'], axis=1).shape)
print('Origianl dataset shape %s' % Counter( df_all_new_sel.Entscheidung))

# Apply the random over-sampling
ros = RandomOverSampler(random_state=random_state)
X_resampled, y_resampled = ros.fit_resample(df_all_new_sel.drop(['Entscheidung'], axis=1), 
                                                    df_all_new_sel.Entscheidung)

print('Original data: ', X_resampled.shape)
print('Resampled dataset shape %s' % Counter(y_resampled))

X_train, X_test, y_train, y_test = train_test_split(X_resampled, 
                                                    y_resampled, test_size=0.2, random_state=random_state)

# X_train_id = X_train[['study_no', 'start_date_time', 'end_date_time']]
# X_train = X_train.drop(['study_no', 'start_date_time', 'end_date_time'], axis=1)
# X_test_id = X_test[['study_no', 'start_date_time', 'end_date_time']]
# X_test = X_test.drop(['study_no', 'start_date_time', 'end_date_time'], axis=1)




X_train_tr = np.array(X_train)
X_test_tr = np.array(X_test)
# sc = StandardScaler()
# X_train_tr = sc.fit_transform(X_train)
# X_test_tr = sc.transform(X_test)

def gen_output(y_pred, y_test, X_test_id):
    df_y_pred = pd.DataFrame(y_pred, columns = ['Prediction'])
    df_y_test = pd.DataFrame(y_test, columns = ['Entscheidung'])
    df_X_test = (df_y_test.Entscheidung.to_frame().join(X_test_id)).reset_index().drop(['index'], axis=1)
    df_comb = (df_y_pred.Prediction.to_frame().join(df_X_test))
    df_comb = df_comb.sort_values(by=['study_no', 'start_date_time'],ascending=[True, True])   
    return df_comb['study_no', 'Entscheidung', 'Prediction', 'start_date_time', 'end_date_time']



#Decision Tree
#model = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=2)
model = DecisionTreeClassifier(random_state=7)
model.fit(X_train_tr, y_train)
y_pred = model.predict(X_test_tr)
acc = accuracy_score(y_test, y_pred)   
print('accuracy: ', acc)
confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
sensitivity = tp/(tp+fn)    #TPR
recall = sensitivity
print('sensitivity: ',sensitivity)
specificity = tn/(tn+fp)    #TNR
print('specificity:', specificity)
precision = tp/(tp+fp)    #PPV
print('precision:', precision)
Fscore = 2*((precision*recall)/(precision+recall))
print('Fscore:', Fscore)



# #Random Forest
# model = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=7, n_jobs=2)
# model.fit(X_train_tr, y_train)
# y_pred = model.predict(X_test_tr)
# acc = accuracy_score(y_test, y_pred)   
# print('accuracy: ', acc)
# confusion_matrix(y_test, y_pred)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print(tn, fp, fn, tp)
# sensitivity = tp/(tp+fn)    #TPR
# recall = sensitivity
# print('sensitivity: ',sensitivity)
# specificity = tn/(tn+fp)    #TNR
# print('specificity:', specificity)
# precision = tp/(tp+fp)    #PPV
# print('precision:', precision)
# Fscore = 2*((precision*recall)/(precision+recall))
# print('Fscore:', Fscore)




#feature inportance using DT
sort = model.feature_importances_.argsort()
plt.barh(X_test.columns[sort], model.feature_importances_[sort])
plt.xlabel("Feature Importance")
feature_importances_dt = {fea: imp for imp, fea in zip(model.feature_importances_[sort], X_test.columns[sort])}







# Import the SHAP library
from scipy.special import softmax
import shap

# load JS visualization code to notebook
shap.initjs()

# Create the explainer
explainer = shap.TreeExplainer(model)

"""
Compute shap_values for all of X_test rather instead of 
a single row, to have more data for plot.
"""
shap_values = explainer.shap_values(X_test)
     
print("Variable Importance Plot - Global Interpretation")
figure = plt.figure()
shap.summary_plot(shap_values, X_test)

# Summary Plot Deep-Dive on Label 1
shap.summary_plot(shap_values[1], X_test)


# Dependence Plot on Age feature
#shap.dependence_plot('age', shap_values[0], X_test, interaction_index="male")





def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.shape[1]):
        importances.append(np.mean(np.abs(shap_values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")
    
    return feature_importances




feature_importances_sh = print_feature_importances_shap_values(shap_values[0], X_test.columns)



def merge_dicts_to_dataframe(dict1, dict2):
    merged_dict = {}
    for key in dict1.keys():
        if key in dict2:
            merged_dict[key] = [dict1[key], dict2[key]]

    df = pd.DataFrame.from_dict(merged_dict, orient='index', columns=['Feature importance DT', 'Feature importance shaply'])
    return df


# Merge dictionaries into a dataframe
merged_df_FI_ROS = merge_dicts_to_dataframe(feature_importances_dt, feature_importances_sh)
merged_df_FI_ROS.to_csv('XAI/merged_df_FI_ROS_it'+str(ii)+'.csv', index=True)
#print(merged_df_FI_ROS)




merged_df_FI_ROS_s = merged_df_FI_ROS.sort_values('Feature importance shaply')
priority = list(merged_df_FI_ROS_s.index)
#print(priority)

with open('XAI/sampling_results_iter.txt', 'a') as f:
    print('ii', ii, file=f)
    print('Original data: ',df_all_new_sel.drop(['Entscheidung'], axis=1).shape, file=f)
    print('Origianl dataset shape %s' % Counter( df_all_new_sel.Entscheidung), file=f)
    print('accuracy: ', acc, file=f)
    print(tn, fp, fn, tp, file=f)
    print('sensitivity: ',sensitivity, file=f)
    print('specificity:', specificity, file=f)
    print('precision:', precision, file=f)
    print('Fscore:', Fscore, file=f)
    print('top_feature:', priority[-1], file=f)
    print('\n', file=f)

p_list.append(priority.pop())
ii = ii + 1

import pickle
with open("XAI/p_list.pkl", "wb") as fp:   #Pickling
    pickle.dump(p_list, fp)
    
with open("XAI/p_list.pkl", "rb") as fp:   # Unpickling
    pist = pickle.load(fp)



















