'''DecisionTreeClassifier(max_depth=10, min_samples_split=5)
0.8533149660874118
AdaBoostClassifier(learning_rate=1.5, n_estimators=500)
0.8691694759807789
RandomForestClassifier(criterion='entropy', min_samples_leaf=2,
                       min_samples_split=7, n_estimators=200)
0.8619907225387626
ExtraTreesClassifier(criterion='entropy', min_samples_leaf=2,
                     min_samples_split=7, n_estimators=200)
0.8565010801841932
GradientBoostingClassifier(max_depth=5, min_samples_leaf=5, min_samples_split=4,
                           n_estimators=300, subsample=1)
0.8718567266384145'''

# !/usr/bin/env python
# coding: utf-8

# # 1. 학습 가능한 형태로 데이터 변환

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns

from category_encoders.ordinal import OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# In[2]:


train_df = pd.read_csv('input/kakr-4th-competition/train.csv')
test_df = pd.read_csv('input/kakr-4th-competition/test.csv')

# In[3]:


train_df.drop(['id'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)

# 우선 예측하고자 하는 'income' 을 True/False 형태로 변환해준 뒤, X와 y를 분리했습니다.

# In[4]:


y = train_df['income'] != '<=50K'
X = train_df.drop(['income'], axis=1)

# Ordinal Encoder를 이용한 라벨링을 진행합니다.

# In[5]:


LE_encoder = OrdinalEncoder(list(X.columns))

X = LE_encoder.fit_transform(X, y)
test_df = LE_encoder.transform(test_df)

# 라벨링을 마치고 나면 아래와 같은 데이터로 정리됩니다.

# In[6]:


X['income'] = y
X.head(5)

# 'native_country' 열만 float 형태여서, 다른 열과 동일하게 형변환을 진행했습니다.

# In[7]:


test_df['native_country'] = test_df['native_country'].astype(np.int64)

# 이제 마지막으로 X_train, y_train, X_test를 나누어 저장해둡니다.

# In[8]:


y_train = X['income'].values
X_train = X.drop(['income'], axis=1).values
X_test = test_df.values

# # 2. Feature Tuning

# ## 1) Decision Trees

# In[9]:


dt_params = {
    'max_depth': [2, 3, 5, 7, 10, 20, 30],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [1, 2, 3, 5, 7],
}

# 5개의 모델을 아래와 같이 생성한 뒤, 결과값을 변수에 저장합니다.

# In[10]:


from sklearn.model_selection import RandomizedSearchCV

dt_model = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=dt_params, n_iter=100, cv=5)

# In[11]:


X_train

# In[12]:


y_train

# In[13]:


check = dt_model.fit(X_train, y_train)

# In[14]:


check.best_estimator_

# In[15]:


check.best_score_

# ## 2) AdaBoost

# In[16]:


ada_params = {
    'n_estimators': [25, 50, 75, 100, 200, 300, 500, 1000],
    'learning_rate': [0.1, 0.5, 1, 1.5, 2],
}

# In[17]:


ada_model = RandomizedSearchCV(AdaBoostClassifier(), param_distributions=ada_params, n_iter=30, cv=5)

# In[18]:


X_train

# In[19]:


y_train

# In[20]:


check = ada_model.fit(X_train, y_train)

# In[21]:


check.best_estimator_

# In[22]:


check.best_score_

# ## 3) RandomForest

# In[23]:


rf_params = {
    'n_estimators': [25, 50, 75, 100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [1, 2, 3, 4, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 5, 7]
}

# In[24]:


rf_model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_params, n_iter=30, cv=5)

# In[25]:


X_train

# In[26]:


y_train

# In[27]:


check = rf_model.fit(X_train, y_train)

# In[28]:


check.best_estimator_

# In[29]:


check.best_score_

# ## 4) ExtraTrees

# In[35]:


et_params = {
    'n_estimators': [25, 50, 75, 100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [1, 2, 3, 4, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 5, 7],
}

# In[36]:


et_model = RandomizedSearchCV(ExtraTreesClassifier(), param_distributions=et_params, n_iter=30, cv=5)

# In[37]:


X_train

# In[38]:


y_train

# In[39]:


check = et_model.fit(X_train, y_train)

# In[40]:


check.best_estimator_

# In[41]:


check.best_score_

# ## 5) GBM

# In[42]:


gb_params = {
    'n_estimators': [25, 50, 75, 100, 200, 300],
    'loss': ['deviance', 'exponential'],
    'subsample': [0.3, 0.5, 0.7, 1],
    'min_samples_split': [1, 2, 3, 4, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 5, 7],
    'max_depth': [1, 2, 3, 5, 7]
}

# In[43]:


gb_model = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=gb_params, n_iter=30, cv=5)

# In[44]:


X_train

# In[45]:


y_train

# In[46]:


check = gb_model.fit(X_train, y_train)

# In[47]:


check.best_estimator_

# In[48]:


check.best_score_

# In[ ]:




