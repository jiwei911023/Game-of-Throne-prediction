# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:51:50 2019

@author: 54329
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:59:30 2019

@author: 54329
"""
###############################################################################
###########################Machine Learning Assignment2########################
###############################################################################

# Importing new libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
from sklearn.metrics import roc_curve, auc


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split 
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
#Import Data
got = pd.read_excel('GOT_character_predictions.xlsx')

###############################################################################
# Dataset Summary 
###############################################################################

# General Summary of the Dataset
got.info()

got.head(n = 5)

got.describe().round(2)

got.corr().round(3)

###############################################################################
# Check and imputing missing value
###############################################################################
print(got
      .isnull()
      .sum())
## Missing values in following columns:
                                       # age
                                       # title
                                       # culture
                                       # dateOfBirth
                                       # house
                                       # isAliveSpouse
                                       # spouse
                                       # isAliveMother
                                       # isAliveFather
                                       # isAliveHeir
                                       # mother
                                       # father
                                       #heir

#####################################################
##############Feature Engineering#####################
#####################################################

#Flag missing value
for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)
# Deal with Date of birth and there are  2 wrong data we must correct them
got.loc[110,'dateOfBirth']=298
got.loc[1350,'dateOfBirth']=272

got['dateOfBirth'].mean()
got['dateOfBirth'].median()

## Check which characters have a negative age and it's value.
print(got['name'][got['age']< 0])
print(got['age'][got['age']< 0])

## We can google and find their exact name , Rhaego's age is 0, Doreah is 25
## Replace negative age by the corret name

got.loc[110, 'age'] = 0.0
got.loc[1350, 'age'] = 25.0

got['age'].mean()
got['age'].median()

#####if there are few nans , we can drop them directly but in this dataset
#we have so many nans so wo should impute them.
# use plot to decide whether to use mean or median to fill the nans.

plt.subplot(1, 2, 1)
got_drop_age=got['age'].dropna(axis=0)
sns.distplot(got_drop_age)
plt.xlabel('age')


plt.subplot(1, 2, 2)
got_drop_birth=got['dateOfBirth'].dropna(axis=0)
sns.distplot(got_drop_birth)
plt.xlabel('dateofbirth')

plt.show()



# Fill the missing value
got["age"].fillna(got["age"].mean(), inplace=True)
got["dateOfBirth"].fillna(got["dateOfBirth"].mean(), inplace=True)
got["culture"].fillna("unknowncul", inplace=True)
got["house"].fillna("unknownhou", inplace=True)
got["title"].fillna("unknowntit", inplace=True)
got["spouse"].fillna("unknownspouse", inplace=True)
got["father"].fillna("unknownfather", inplace=True)
got["mother"].fillna("unknownmother", inplace=True)
got["heir"].fillna("unknownheir", inplace=True)

# For some numeric missing value we dont know them so fill them with -1
got.fillna(value=-1, inplace=True)
got.isnull().sum()
got2=got.copy()

##############################################################################
########set up the group and make the combination to simplify the data########
##############################################################################

################# culture group##############
#filter the duplicated value
set(got['culture'])
got['culture'].nunique()
cul = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Andal': ['andal', 'andals'],
    'Lysene': ['lysene', 'lyseni'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Vale': ['vale', 'valemen', 'vale mountain clans'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Mereen': ['meereen', 'meereenese'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    'Qartheen': ['qartheen', 'qarth'],
    'Ironborn': ['ironborn', 'ironmen'],
    'Astapor': ['astapor', 'astapori'],
    'Lhazareen': ['lhazareen', 'lhazarene'],
    'RiverLands': ['riverlands', 'rivermen']
    }


def combine_culture(value):
    value = value.lower()
    i = [j for (j, i) in cul.items() if value in i]
    return i[0] if len(i) > 0 else value.title()
got.loc[:, "culture"] = [combine_culture(x) for x in got["culture"]]

#Then use the final culture counts to make the group

culture_list = got['culture'].value_counts()
cul_lo = 40
unknowncul=1269
got['new_culture'] = " "

for count in range(0, got.shape[0]):
    if (culture_list.loc[got.loc[count, 'culture']] > cul_lo):
        got.loc[count, 'new_culture'] = 1
    elif (culture_list.loc[got.loc[count, 'culture']]==unknowncul) :
        got.loc[count, 'new_culture'] = 0
    else:
        got.loc[count, 'new_culture'] = 0
        

##############Set up House group ##############

house_list = got['house'].value_counts()
house_lo = 40
unknownhou=427
got['new_house'] = " "
for count in range(0, got.shape[0]):
    if (house_list.loc[got.loc[count, 'house']] > house_lo):
        got.loc[count, 'new_house'] = 1
    elif (house_list.loc[got.loc[count, 'house']]==unknownhou):
        got.loc[count, 'new_house'] = 1
    else:
        got.loc[count, 'new_house'] = 0
        
############title group#######################
got['title'].value_counts()

title_lo = 30
unknowntit=1008
title_list = got['title'].value_counts()
got['new_title'] = " "
for count in range(0, got.shape[0]):
    if (title_list.loc[got.loc[count, 'title']] > title_lo):
        got.loc[count, 'new_title'] = 1
    elif (title_list.loc[got.loc[count, 'title']] == unknowntit):
        got.loc[count, 'new_title'] = 1
    else:
        got.loc[count, 'new_title'] = 0

#################outlier#####################
#popularity
sns.distplot(got['popularity'])

sns.lmplot(x="S.No", y="popularity", data=got,
                fit_reg=False,scatter=True)

pop_hi=0.8
got['out_popularity'] = 0


for val in enumerate(got.loc[ : , 'popularity']):
    
    if val[1] >= pop_hi:
        got.loc[val[0], 'out_popularity'] = 1
        
#age
sns.distplot(got['age'])

sns.lmplot(x="S.No", y="age", data=got,
                fit_reg=False,scatter=True)

age_hi=100
age_lo=0
got['out_age'] = 0


for val in enumerate(got.loc[ : , 'age']):
    
    if val[1] >= age_hi:
        got.loc[val[0], 'out_age'] = 1
        
for val in enumerate(got.loc[ : , 'age']):
    
    if val[1] <= age_lo:
        got.loc[val[0], 'out_age'] = -1
        
        
###############create different houses/culture/title alive rate########
#culture survive rate
alive_df=got[got['isAlive']==1]
a1 = pd.Series(got.culture.value_counts(), name='overall_culture_count')
b1 = pd.Series(alive_df.culture.value_counts(),name = 'alive_culture_count')
c1 = pd.concat([a1,b1], axis=1, join_axes=[a1.index])
c1['alive_rate_culture'] = (c1['alive_culture_count']/
                            c1['overall_culture_count']).round(2)
got['alive_rate_culture'] = got['culture'].map(c1['alive_rate_culture'])

#house survive rate
alive_df=got[got['isAlive']==1]
a2 = pd.Series(got.house.value_counts(), name='overall_house_count')
b2 = pd.Series(alive_df.house.value_counts(),name = 'alive_house_count')
c2 = pd.concat([a2,b2], axis=1, join_axes=[a2.index])
c2['alive_rate_house'] = (c2['alive_house_count']/
                          c2['overall_house_count']).round(2)
got['alive_rate_house'] = got['house'].map(c2['alive_rate_house'])


#title survive rate

alive_df=got[got['isAlive']==1]
a3 = pd.Series(got.title.value_counts(), name='overall_title_count')
b3 = pd.Series(alive_df.title.value_counts(),name = 'alive_title_count')
c3 = pd.concat([a3,b3], axis=1, join_axes=[a3.index])
c3['alive_rate_title'] = (c3['alive_title_count']/
                          c3['overall_title_count']).round(2)
got['alive_rate_title'] = got['title'].map(c3['alive_rate_title'])

###########impute missing value again#########
got_drop=got.loc[:,['alive_rate_title',
                    'alive_rate_house',
                    'alive_rate_culture']].dropna()

sns.distplot(got_drop['alive_rate_title'])
title_median=got['alive_rate_title'].median()
title_mode=got['alive_rate_title'].mode()##use median impute

sns.distplot(got_drop['alive_rate_house'])
house_median=got['alive_rate_house'].median()#use median impute
house_mean=got['alive_rate_house'].mean()

sns.distplot(got_drop['alive_rate_culture'])
culture_median=got['alive_rate_culture'].median()
culture_mode=got['alive_rate_culture'].mode()##use median impute

#fill missing value again
got["alive_rate_title"].fillna(title_median, inplace=True)
got["alive_rate_house"].fillna(house_median, inplace=True)
got["alive_rate_culture"].fillna(culture_median, inplace=True)


###finally check the nans

got.isnull().sum()


#########################
## EDA##
#########################

#Cross plot

def crosstbplot(x, y):  
   crosstab = pd.crosstab(x, y)
   crosstab.plot(kind = 'bar', figsize=(18, 18))
   plt.xlabel(x)
   plt.ylabel('Count')
   plt.show()

# Plot popularity
plt.figure(figsize=(8, 6))

got_male = pd.crosstab(got2['popularity'], got2['isAlive'])

got_male.plot(kind = 'bar')

plt.xlabel('If a person is male')
plt.ylabel('Count')

plt.show()


## Corr visualization
corr=got.corr()
corr_isalive=corr.loc['isAlive'].sort_values(ascending = False)

sns.heatmap(got.corr(),annot=True,cmap='Set2',linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(30,25)
plt.show()

###Use voilin plot to show the distribution and  density .
f,ax=plt.subplots(2,2,figsize=(17,15))
sns.violinplot("isNoble", "age", hue="isAlive", 
               data=got ,split=True, ax=ax[0, 0])
ax[0, 0].set_title('isNoble and age vs Mortality')
ax[0, 0].set_yticks(range(2))

sns.violinplot("isNoble", "male", hue="isAlive", 
               data=got ,split=True, ax=ax[0, 1])
ax[0, 1].set_title('isNoble and Male vs Mortality')
ax[0, 1].set_yticks(range(2))

sns.violinplot("isNoble", "isMarried", hue="isAlive", 
               data=got ,split=True, ax=ax[1, 0])
ax[1, 0].set_title('isNoble and isMarried vs Mortality')
ax[1, 0].set_yticks(range(2))


sns.violinplot("isNoble", "book1_A_Game_Of_Thrones", hue="isAlive", 
               data=got ,split=True, ax=ax[1, 1])
ax[1, 1].set_title('isNoble and book1AGameOfThrones vs Mortality')
ax[1, 1].set_yticks(range(2))

plt.show()

####violin
f,ax=plt.subplots(2,2,figsize=(17,15))
sns.violinplot("isNoble", "book2_A_Clash_Of_Kings", hue="isAlive", 
               data=got ,split=True, ax=ax[0, 0])
ax[0, 0].set_title('isNoble and book2AClashOfKings vs Mortality')
ax[0, 0].set_yticks(range(2))

sns.violinplot("isNoble", "book3_A_Storm_Of_Swords", hue="isAlive", 
               data=got ,split=True, ax=ax[0, 1])
ax[0, 1].set_title('isNoble and book3AStormOfSwords vs Mortality')
ax[0, 1].set_yticks(range(2))

sns.violinplot("isNoble", "book4_A_Feast_For_Crows", hue="isAlive", 
               data=got ,split=True, ax=ax[1, 0])
ax[1, 0].set_title('isNoble and book4AFeastForCrows vs Mortality')
ax[1, 0].set_yticks(range(2))


sns.violinplot("isNoble", "book5_A_Dance_with_Dragons", hue="isAlive", 
               data=got ,split=True, ax=ax[1, 1])
ax[1, 1].set_title('isNoble and book5ADancewithDragons vs Mortality')
ax[1, 1].set_yticks(range(2))

plt.show()



####################################################################
#####delete duplicated and useless cols and then build the model####
####################################################################
got = got.drop(['house',
                'culture',
                'title',
                'name',
                'S.No',
                'spouse',
                'mother',
                'father',
                'heir'],
                 axis=1)

got.isnull().any()

#######################
###Bulid the model#####
#######################


#1.Logistic model

##smf model

got.nunique()

logistic_full = smf.logit(formula = """isAlive ~ new_house+
                                                 new_title+
                                                 new_culture+
                                                 male+
                                                 dateOfBirth+
                                                 book1_A_Game_Of_Thrones+
                                                 book2_A_Clash_Of_Kings+
                                                 book3_A_Storm_Of_Swords+
                                                 book4_A_Feast_For_Crows+
                                                 book5_A_Dance_with_Dragons+
                                                 isMarried+
                                                 isNoble+
                                                 age+
                                                 m_title+
                                                 m_culture+
                                                 m_dateOfBirth+
                                                 m_mother+
                                                 m_father+
                                                 m_heir+
                                                 m_house+
                                                 m_spouse+
                                                 numDeadRelations+
                                                 popularity+
                                                 m_isAliveMother+
                                                 m_isAliveFather+
                                                 m_isAliveHeir+
                                                 m_isAliveSpouse+
                                                 m_age+
                                                 out_age+
                                                 out_popularity+
                                                 alive_rate_culture+
                                                 alive_rate_house+
                                                 alive_rate_title""",
                                                 data=got)
results_logistic_full = logistic_full.fit()

results_logistic_full.summary()

## Implementing the  full model
got_data = got.drop('isAlive',axis=1)
got_target = got.loc[:, 'isAlive']
X_train, X_test, y_train, y_test = train_test_split(
                                               got_data,
                                               got_target,
                                               test_size = 0.10,
                                               random_state = 508,
                                               stratify = got_target)

# Instantiate
lg_full = LogisticRegression()

# Fit
lg_full_fit = lg_full.fit(X_train, y_train)

# Predict
lg_full_pred = lg_full_fit.predict(X_test)

# Score
y_score_test = lg_full_fit.score(X_test, y_test)
y_score_train = lg_full_fit.score(X_train, y_train)

print(y_score_train)
print(y_score_test)

####################################################################
#######select significant and important variavles as estimators
###### (final train and test data)#############################
######################################################################

got_data = got.loc[:,['age',
                      'male',
                      'new_house',
                      'new_title',
                      'new_culture',
                      'popularity',
                      'book1_A_Game_Of_Thrones',
                      'book2_A_Clash_Of_Kings',
                      'book4_A_Feast_For_Crows',
                      'book3_A_Storm_Of_Swords',
                      'alive_rate_house',
                      'alive_rate_culture',
                      'alive_rate_title'
                      ]]

got_target = got.loc[:, 'isAlive']
X_train, X_test, y_train, y_test = train_test_split(
                                               got_data,
                                               got_target,
                                               test_size = 0.10,
                                               random_state = 508,
                                               stratify = got_target)
lg_sig = LogisticRegression()

# Fit
lg_sig_fit = lg_sig.fit(X_train, y_train)

# Predict
lg_sig_pred = lg_sig_fit.predict(X_test)

# Score
y_score_test = lg_sig_fit.score(X_test, y_test)
y_score_train = lg_sig_fit.score(X_train, y_train)

print(y_score_train.round(2))
print(y_score_test.round(2))

##############AUC Score#########################
# Generating Predictions based on the optimal model
lg_fit_train = lg_sig_fit.predict(X_train)

lg_fit_train_test = lg_sig_fit.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,lg_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,lg_fit_train_test).round(4))

#######AUC
probs = lg_sig.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title('LG GoT isAlive Pred Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Lg_Cross-Validation  (cv = 3)

cv_lg_3 = cross_val_score(lg_sig,
                          got_data,
                          got_target,
                          cv = 3)

print(pd.np.mean(cv_lg_3).round(3))


#2.KNN model

#############KNN model####################


# Running the neighbor optimization code 
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()



# Looking for the highest test accuracy
print(test_accuracy)



# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)



# It looks like 1 neighbors is the most accurate
knn_clf = KNeighborsClassifier(algorithm = 'auto',n_neighbors = 6)



# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)


# Generating Predictions based on the optimal KNN model
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)

# Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))
y_pred_train=knn_clf_fit.predict(X_train)
y_pred_test=knn_clf_fit.predict(X_test)

# Let's also check our auc value
print('Training AUC Score', roc_auc_score(y_train, y_pred_train).round(4))
print('Testing AUC Score:', roc_auc_score(y_test, y_pred_test).round(4))

##############AUC plot#######
probs = knn_clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title('knn GoT isAlive Pred Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# knn_Cross-Validation  (cv = 3)

cv_knn_3 = cross_val_score(knn_clf,
                           got_data,
                           got_target,
                           cv = 3)

print(pd.np.mean(cv_knn_3).round(3))


# 3.Building Classification Trees

##############Decision Tree model##################


from sklearn.tree import DecisionTreeClassifier # Classification trees

c_tree = DecisionTreeClassifier(random_state = 508)

c_tree_fit = c_tree.fit(X_train, y_train)


print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))

###############################################################################
# Hyperparameter Tuning with RandomizedSearchCV!!!!!!!!!!!!!!!
###############################################################################
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 500)
split_space = pd.np.arange(0.1, 1.0)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'min_samples_split':split_space}


# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state = 508)


# Creating a GridSearchCV object
c_tree_2_hp_cv = RandomizedSearchCV(c_tree_2_hp, param_grid, cv = 3,n_iter=50)

# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", 
      c_tree_2_hp_cv.best_score_.round(4))
"""

########################
# max_depth and min_samples_leaf optimized model
########################

c_tree_3 = DecisionTreeClassifier(max_depth = 5,
                                  min_samples_leaf = 11,
                                  min_samples_split= 0.1,
                                  random_state = 508)

c_tree_3_fit = c_tree_3.fit(X_train, y_train)
c_tree_3_pred=c_tree_3_fit.predict(X_test)
c_tree_3_pred_train=c_tree_3_fit.predict(X_train)


print('Training Score', c_tree_3_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_3_fit.score(X_test, y_test).round(4))


# Let's also check our auc value
print('Training AUC Score', roc_auc_score(y_train, 
                                          c_tree_3_pred_train).round(4))
print('Testing AUC Score:', roc_auc_score(y_test, c_tree_3_pred).round(4))
##AUC##############
probs = c_tree_3.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title('CTree GoT isAlive Pred Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Ctree3_Cross-Validation  (cv = 3)

cv_ctree_3 = cross_val_score(c_tree_3,
                             got_data,
                             got_target,
                             cv = 3)

print(pd.np.mean(cv_ctree_3).round(3))
###############################################################################
# Visualizing the Tree
###############################################################################

# Importing the necessary libraries
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

dot_data = StringIO()


export_graphviz(decision_tree = c_tree_3_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train.columns)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)


# Saving the visualization in the working directory
graph.write_png("GOT_Optimal_Classification_Tree.png")

# Feature importance function
########################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_149_Feature_Importance.png')

plot_feature_importances(c_tree_3,
                         train = X_train,
                         export = True)

# 4.Random Forest

####################Random Forest #######################


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot


# Following the same procedure as other scikit-learn modeling techniques

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)



# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)

"""
########################
# Parameter tuning with RandomizedSearchCV
########################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}


# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)

# Creating a GridSearchCV object
full_forest_cv = RandomizedSearchCV(full_forest_grid, 
                                    param_grid, 
                                    cv = 3,
                                    n_iter=50)


# Fit it to the training data
full_forest_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", 
      full_forest_cv.best_score_.round(4))
"""
###########Optimal rf################
rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 1,
                                    n_estimators = 350,
                                    warm_start = True)



rf_op_fit=rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)

########################
# Feature importance function
########################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(20,15))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_,align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


plot_feature_importances(rf_op_fit,
                         train = X_train,
                         export = True)





#AUC Score 
# Generating Predictions based on the optimal Random Forest model
rf_optimal_pred_train = rf_optimal.predict(X_train)

rf_optimal_pred_test = rf_optimal.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,rf_optimal_pred_train).round(3))
print('Testing AUC Score:',roc_auc_score(
        y_test,rf_optimal_pred_test).round(3))


##AUC plot
probs = rf_optimal.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(15, 10))
plt.title('GoT isAlive Pred Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Cross-Validation on rf_optimal (cv = 3)

cv_rf_optimal = cross_val_score(rf_optimal,
                                got_data,
                                got_target,
                                cv = 3)

print(pd.np.mean(cv_rf_optimal).round(3))


#5.Gradient Boosted Machines

####################  GBM    ######################


from sklearn.ensemble import GradientBoostingClassifier

# Building a weak learner gbm
gbm_3 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )


gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)

"""
########################
# Applying GridSearchCV
########################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}



# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)


# Creating a RandomizedSearchCV object
gbm_grid_cv = RandomizedSearchCV(gbm_grid, param_grid, cv = 3)



# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))
"""

########################
# Building GBM Model Based on Best Parameters
########################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.2,
                                      max_depth =3,
                                      n_estimators = 150,
                                      random_state = 508)



gbm_op_fit=gbm_optimal.fit(X_train, y_train)


gbm_optimal_score = gbm_optimal.score(X_test, y_test)


gbm_optimal_pred = gbm_optimal.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))


gbm_optimal_train = gbm_optimal.score(X_train, y_train)
gmb_optimal_test  = gbm_optimal.score(X_test, y_test)


##############AUC Score#########################
# Generating Predictions based on the optimal model
gbm_fit_train = gbm_op_fit.predict(X_train)

gbm_fit_train_test = gbm_op_fit.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,gbm_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,gbm_fit_train_test).round(4))


##AUC plot
probs = gbm_optimal.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 5))
plt.title('GoT isAlive Pred Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# gbm_Cross-Validation  (cv = 3)


cv_gbm_3 = cross_val_score(gbm_optimal,
                           got_data,
                           got_target,
                           cv = 3)

print(pd.np.mean(cv_gbm_3).round(3))


########################
# Saving Results
########################

# Saving best model scores
print(f"""
         RF_Score_test: {rf_optimal.score(X_test, y_test).round(4)},
         RF_Score_train: {rf_optimal.score(X_train, y_train).round(4)},
         GBM_Score_test: {gbm_optimal.score(X_test, y_test).round(4)},
         GBM_Score_train:{gbm_optimal.score(X_train, y_train).round(4)},
         Logistic_Score_test:{lg_sig_fit.score(X_test, y_test).round(4)},
         Logistic_Score_train:{lg_sig_fit.score(X_train, y_train).round(4)},
         KNN_Score_test:{knn_clf_fit.score(X_test, y_test).round(4)},
         KNN_Score_train:{ knn_clf_fit.score(X_train, y_train).round(4)},
         Ctree_Score_test:{c_tree_3_fit.score(X_test, y_test).round(4)},
         Ctree_Score_train:{c_tree_3_fit.score(X_train, y_train).round(4)}
         """)


# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'GBM_Predicted': gbm_optimal_pred})


model_predictions_df.to_excel("JiweiModel_Predictions.xlsx")



