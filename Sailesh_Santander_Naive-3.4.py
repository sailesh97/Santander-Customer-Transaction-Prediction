#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt2
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report,roc_curve,auc

from inspect import signature
from sklearn.metrics import average_precision_score,precision_recall_curve

from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[3]:


os.chdir("C:/Users/Acesocloud/Downloads/Kaggle/Santander Customer Transaction Prediction/Sailesh Santander")


# In[4]:


df_santander = pd.read_csv("train.csv")


# In[5]:


df_santander_test = pd.read_csv("test.csv")


# In[6]:


print('Shape of our dataset:')
print(df_santander.shape,'\n')


# In[7]:


pd.options.display.max_columns = None


# # Exploratory Data Analysis

# In[8]:


print('*'*25,'Exploratory Data Analysis: ','*'*25,'\n')


# In[9]:


print('Showing 1st few rows of our dataset: \n')
print(df_santander.head(5))


# In[10]:


print("Basic info about dataset:\n")
print(df_santander.info())


# In[11]:


print("Data Description:\n")


# ### Target Class Count

# In[12]:


target_count = df_santander['target'].value_counts()


# In[13]:


print("Count of categories of the target variable:\n", target_count)


# In[14]:


print("Percentage of each category of the target variable:\n", ((target_count/df_santander.shape[0]))*100)


# ### Data Visualization

# In[15]:


f, ax = plt2.subplots(1,2,figsize=(15,8))
pie_data = df_santander['target'].value_counts()
pie_data.plot.pie(explode=[0,0.2], autopct='%1.2f%%', ax = ax[0], shadow = True)
ax[0].set_title('Training Set Target Distribution')
ax[0].set_ylabel('')

sns.countplot('target', data = df_santander, ax = ax[1])
plt2.show()


# ### Missing Value Analysis

# In[16]:


train_missing = df_santander.isnull().sum()


# In[17]:


print("No. of rows having missing values in train data:")
print(train_missing.loc[train_missing > 0].shape[0])


# In[18]:


test_missing = df_santander_test.isnull().sum()
print("No. of rows having missing values in test data:")
print(test_missing.loc[test_missing > 0].shape[0])


# ### Outlier Analysis

#      Can not perform as we have imbalance dataset

# ### Distribution of training data

# In[19]:


def plot_train_data_dist(cat_0,cat_1, label1, label2, columns):
    i = 0
    sns.set_style('darkgrid')
    
    fig = plt2.figure()
    ax = plt2.subplots(10,10,figsize=(22,18))
    
    for col in columns:
        i += 1
        plt2.subplot(10,10,i)
        sns.distplot(cat_0[col], hist=False, label=label1)
        sns.distplot(cat_1[col], hist=False, label=label2)
        plt2.legend()
        plt2.xlabel('Attribute',)
    plt2.show()


# In[20]:


cat_0 = df_santander.loc[df_santander['target'] == 0]
cat_1 = df_santander.loc[df_santander['target'] == 1]


# In[21]:


label1 = '0'
label2 = '1'


# In[22]:


columns = df_santander.columns.values[2:102]
plot_train_data_dist(cat_0, cat_1, label1, label2, columns)


# In[23]:


columns = df_santander.columns.values[102:202]
plot_train_data_dist(cat_0, cat_1, label1, label2, columns)


# ### Distribution of test data

# In[24]:


def plot_test_data_dist(test_attributes):
    i=0
    sns.set_style('whitegrid')
    
    fig=plt2.figure()
    ax=plt2.subplots(10,10,figsize=(22,18))
    
    for attribute in test_attributes:
        i+=1
        plt2.subplot(10,10,i)
        sns.distplot(df_santander_test[attribute],hist=False)
        plt2.xlabel('Attribute',)
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    plt2.show()


# In[25]:


test_attributes=df_santander_test.columns.values[1:101]
plot_test_data_dist(test_attributes)


# In[26]:


test_attributes=df_santander_test.columns.values[102:202]
plot_test_data_dist(test_attributes)


# ### Check for duplicate rows

# In[27]:


duplicateRowsDF = df_santander[df_santander.duplicated()]
 
print("No. of duplicate rows based on all columns are :")
print(duplicateRowsDF.shape[0])


# In[28]:


duplicateRowsDF = df_santander_test[df_santander_test.duplicated()]
 
print("No. of duplicate rows based on all columns are :")
print(duplicateRowsDF.shape[0])


# ### Correlation Analysis

# In[29]:


num_train = df_santander.columns.values[2:202]
num_test = df_santander_test.columns.values[1:201]


# #### Correlation between train data

# In[30]:


train_corr = df_santander[num_train].corr().abs()


# In[31]:


train_corr = train_corr.unstack()
train_corr


# In[32]:


train_corr = train_corr.sort_values(kind="quicksort")
train_corr


# In[33]:


train_corr = train_corr.reset_index()
train_corr


# In[34]:


train_corr


# #### Correlation between test data

# In[35]:


test_corr = df_santander_test[num_test].corr().abs()


# In[36]:


test_corr = test_corr.unstack()


# In[37]:


test_corr = test_corr.sort_values(kind="quicksort")


# In[38]:


test_corr = test_corr.reset_index()


# #### Excluding correlation between same variables as that will be 1 always

# In[39]:


train_corr = train_corr[train_corr['level_0']!=train_corr['level_1']]
train_corr


# In[40]:


test_corr = test_corr[test_corr['level_0']!=test_corr['level_1']]


# In[41]:


test_corr.iloc[:,2].describe()


# In[42]:


train_corr.iloc[:,2].describe()


# In[43]:


train_corr=df_santander[num_train].corr()
train_corr


# In[44]:


train_corr=train_corr.values.flatten()
train_corr


# In[45]:


train_corr=train_corr[train_corr!=1]


# In[46]:


test_corr=df_santander_test[num_test].corr()


# In[47]:


test_corr = test_corr.values.flatten()


# In[48]:


test_corr=test_corr[test_corr!=1]


# In[49]:


plt2.figure(figsize=(20,5))
sns.distplot(train_corr,color="blue",label="train")
sns.distplot(test_corr,color="red",label="test")
plt2.xlabel("Correlation values found in train & test data")
plt2.ylabel("Density")
plt2.title ("Correlation values in train & test data")
plt2.legend()


# ### Feature Importance

# In[50]:


X = df_santander.drop(columns=['ID_code', 'target'], axis=1)
y = df_santander['target']


# In[51]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)


# In[52]:


rf_model=RandomForestClassifier(n_estimators=10,random_state=42)
rf_model.fit(X_test,y_test)


# In[53]:


importance = pd.DataFrame(rf_model.feature_importances_, columns = ['Feature Importance']) 


# In[54]:


columns = pd.DataFrame(data=X.columns.values);


# In[55]:


columns['imporatance'] = importance


# In[56]:


columns = columns.rename(columns={0: "Variable"})


# In[57]:


columns = columns.rename(columns={'imporatance':'importance'})


# In[58]:


columns.sort_values(by=['importance'], inplace=True) 


# In[59]:


columns


# In[60]:


# Var_81 most important


# In[61]:


X=df_santander.drop(['ID_code','target'],axis=1)
y=df_santander['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2019)


# In[62]:


sm = SMOTE(random_state=42)
X_smote,y_smote=sm.fit_sample(X_train,y_train)
X_smote_v,y_smote_v=sm.fit_sample(X_test,y_test)


# In[63]:


x = pd.concat([X_smote,y_smote],axis=1)


# In[64]:


y = pd.concat([X_smote_v,y_smote_v], axis=1)


# In[65]:


xy = pd.concat([x,y],axis=0)


# In[66]:


X_train.head()


# ### Feature Scaling

# In[67]:


from sklearn.preprocessing import StandardScaler


# In[68]:


X_train = StandardScaler().fit_transform(X_train)


# In[69]:


X_test = StandardScaler().fit_transform(X_test)


# ## PCA

# In[70]:


from sklearn.preprocessing import StandardScaler


# In[71]:


x = StandardScaler().fit_transform(xy.drop(['target'],axis=1))


# In[72]:


from sklearn.decomposition import PCA
pca = PCA(n_components=170)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
print(sum(pca.explained_variance_))
print(sum(pca.explained_variance_ratio_))


# In[73]:


X1 = principalDf
y1 = xy['target']
X_train_PC,X_test_PC,y_train_PC,y_test_PC=train_test_split(X1,y1,random_state=42)


# In[74]:


plt2.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1],
            c=xy['target'], edgecolor='none', alpha=0.5)
plt2.xlabel('component 3')
plt2.ylabel('component 2')
plt2.colorbar();


# In[75]:


fig = plt2.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = principalDf.iloc[:,0]
ys = principalDf.iloc[:,1]
zs = principalDf.iloc[:,2]
# size = list(df_santander['target'])
ax.scatter(xs, ys, zs, alpha=0.6, edgecolors='w',c=xy['target'],s=50)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')


# In[76]:


per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)


# In[77]:


labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]


# In[78]:


fig= plt2.figure(figsize=(100,50))
plt2.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt2.ylabel('Percentage of Explained Variance')
plt2.xlabel('Principal Component')
plt2.title('Scree Plot')
plt2.show()


# In[79]:


plt2.figure(figsize=(26,9))
plt2.plot(pca.explained_variance_ratio_)
# plt2.xticks(range(80))
plt2.xlabel("Number of Features")
plt2.ylabel("Proportion of variance explained by additional feature")


# ## Model

# In[80]:


def draw_confusion_mx(y_test,y_pred):
    print('\n######### Confusion Matrix #########\n')
    cm=pd.crosstab(y_test,y_pred)
    print(cm)
    
def draw_classification_report(y_test,y_pred):
    print('\n######### Classification Report #########\n')
    print(classification_report(y_test,y_pred))

def draw_roc_auc(y_test,y_pred):  ##y_pred in form of probabilities
    ns_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    plt2.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt2.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    auc_score=auc(lr_fpr,lr_tpr)
    plt2.title('ROC(area=%0.3f)' %auc_score)
    
    plt2.xlabel('False Positive Rate')
    plt2.ylabel('True Positive Rate')
    
    plt2.legend()
    
    plt2.show()
    
def draw_precision_recall(y_test,y_pred):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt2.fill_between).parameters
               else {})
    plt2.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt2.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt2.xlabel('Recall')
    plt2.ylabel('Precision')
    plt2.ylim([0.0, 1.05])
    plt2.xlim([0.0, 1.0])
    plt2.title(' Precision-Recall curve: PR_AUC={0:0.3f}'.format( auc(recall, precision)))
    plt2.show() 


# In[81]:


def fit_N_predict(model,X_train,X_test,y_train,y_test,model_code,testData,PCA=0):
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred2 = model.predict_proba(X_test)
    y_pred2 = y_pred2[:,1]
    
    draw_confusion_mx(y_test,y_pred)
    
    draw_classification_report(y_test,y_pred)
    
    draw_roc_auc(y_test,y_pred2)
    
    draw_precision_recall(y_test,y_pred2) 
    if(PCA == 0):
        if(model_code!="XGB"):
            print('\n\nModel performance on test data:\n',)
            print(model.predict(testData.drop(['ID_code'],axis=1)))
        else:
            print('\n\nModel performance on test data:\n',)
            print(model.predict(testData.drop(['ID_code'],axis=1).values))
    
                                                                          


# ### Logistic Regression Model
# 

# In[173]:


lr_model=LogisticRegression(random_state=42,class_weight = 'balanced')


# In[174]:


print("LOGISTIC REGRESSION ON ORIGINAL DATASET\n\n")
fit_N_predict(lr_model,X_train,X_test,y_train,y_test,model_code='LR',testData=df_santander_test)


# ## Logistic Regression after applying SMOTE

# In[149]:


print("LOGISTIC REGRESSION SMOTE DATASET\n\n")
fit_N_predict(lr_model,X_smote,X_smote_v,y_smote,y_smote_v,model_code='LR',testData=df_santander_test)


# ## LR on SMOTE dataset and PCA

# In[152]:


print("LOGISTIC REGRESSION ON PCA+SMOTE DATASET\n\n")
fit_N_predict(lr_model,X_train_PC,X_test_PC,y_train_PC,y_test_PC,model_code='LR',testData = df_santander_test,PCA=1)


# # Decision Tree

# In[153]:


tree_clf = DecisionTreeClassifier(class_weight='balanced', random_state = 2019, 
                                  max_features = 0.7, min_samples_leaf = 80)


# In[154]:


print("DECISION TREE ON ORIGINAL DATASET\n\n")
fit_N_predict(tree_clf,X_train,X_test,y_train,y_test,model_code='DT',testData=df_santander_test)


# ### Decision Tree after applying SMOTE

# In[155]:


print("DECISION TREE ON SMOTE DATASET\n\n")
fit_N_predict(tree_clf,X_smote,X_smote_v,y_smote,y_smote_v,model_code='DT',testData=df_santander_test)


# ## DT + SMOTE + PCA

# In[158]:


print("DECISION TREE ON PCA+SMOTE DATASET\n\n")
fit_N_predict(tree_clf,X_train_PC,X_test_PC,y_train_PC,y_test_PC,model_code='DT',testData=df_santander_test,PCA=1)


# ## Random Forest

# In[159]:


random_forest = RandomForestClassifier(n_estimators=100, random_state=2019, verbose=1,
                                      class_weight='balanced', max_features = 0.5, 
                                       min_samples_leaf = 100,n_jobs=-1)


# In[160]:


print("RANDOM FOREST ON ORIGINAL DATASET\n\n")
fit_N_predict(random_forest,X_train,X_test,y_train,y_test,model_code='RF',testData=df_santander_test)


# In[161]:


print("RANDOM FOREST ON SMOTE DATASET\n\n")
fit_N_predict(random_forest,X_smote,X_smote_v,y_smote,y_smote_v,model_code='RF',testData=df_santander_test)


# ### RF + SMOTE + PCA

# In[162]:


print("RANDOM FOREST ON PCA+SMOTE DATASET\n\n")
fit_N_predict(random_forest,X_train_PC,X_test_PC,y_train_PC,y_test_PC,model_code='RF',testData=df_santander_test,PCA=1)


# ## NaiveBayes

# In[163]:


from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()


# In[164]:


print("NAIVE BAYES ON ORIGINAL DATASET\n\n")
fit_N_predict(NB_model,X_train,X_test,y_train,y_test,model_code='NB',testData=df_santander_test)


# In[165]:


print("NAIVE BAYES ON SMOTE DATASET\n\n")
fit_N_predict(NB_model,X_smote,X_smote_v,y_smote,y_smote_v,model_code='NB',testData=df_santander_test)


# In[166]:


print("NAIVE BAYES ON PCA+SMOTE DATASET\n\n")
fit_N_predict(NB_model,X_train_PC,X_test_PC,y_train_PC,y_test_PC,model_code='NB',testData=df_santander_test,PCA=1)


# ## XGBoost

# In[88]:


from xgboost import XGBClassifier


# In[89]:


XGB = XGBClassifier(learning_rate =0.1,
 n_estimators=800,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 seed=27,scale_pos_weight=2)


# In[90]:


print("XGBOOST CLASSIFIER ON ORIGINAL DATASET\n\n")
fit_N_predict(XGB,X_train,X_test,y_train,y_test,model_code='XGB',testData=df_santander_test)


# In[91]:


print("XGBOOST CLASSIFIER ON SMOTE DATASET\n\n")
fit_N_predict(XGB,X_smote,X_smote_v,y_smote,y_smote_v,model_code='XGB_SM',testData=df_santander_test)


# In[92]:


print("XGBOOST CLASSIFIER ON SMOTE ON PCA DATASET\n\n")
fit_N_predict(XGB,X_train_PC,X_test_PC,y_train_PC,y_test_PC,model_code='XGB',testData=df_santander_test,PCA=1)

