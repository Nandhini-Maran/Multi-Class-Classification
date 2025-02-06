#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


# ## IMPORTING THE DATASET :

# In[2]:


source_data=pd.read_csv('Diabetes.csv')


# ## OVER VIEW OF DATASET :

# In[3]:


source_data.head()


# In[4]:


source_data.tail()


# In[5]:


source_data.shape


# ## TAKING SAMPLE DATA FROM DATASET FOR MODEL PREDICTION :

# In[6]:


df = source_data.sample(frac=0.1, random_state=42)


# In[7]:


df.shape


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.columns


# In[11]:


df.columns = df.columns.str.strip()


# In[12]:


df.columns.tolist()


# In[13]:


for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"{col} is Numeric")
    elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
        print(f"{col} is Categorical")


# In[14]:


df.nunique()


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


df.isna().sum()


# In[18]:


print("Count of duplicated entries:", df.duplicated().sum())


# In[19]:


df = df.drop_duplicates()
print("Count of duplicated entries:", df.duplicated().sum())


# In[20]:


df.shape


# ## ANALYSING AND VISUALIZING THE DATA : 

# In[21]:


numerical_columns = df.select_dtypes(include=['int64']).columns
plt.figure(figsize=(10,5))
df[numerical_columns].boxplot(rot=90,fontsize=10)
plt.title("Box Plot for Numerical Features",fontsize=10)
plt.show()


# In[22]:


df.corr(numeric_only=True)


# In[23]:


plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True) 
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=0.5)
plt.title('Correlation Matrix of Features', fontsize=14)
plt.show()


# ## HISTPLOT FOR FEATURES AND TARGET VALUE :

# In[24]:


num_columns = len(numerical_columns)
n_cols = 4 
n_rows = (num_columns // n_cols) + (num_columns % n_cols > 0)
plt.figure(figsize=(20, n_rows * 5))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=df[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# ## COUNT PLOT FOR TARGET :

# In[25]:


class_distribution = df['Diabetes_012'].value_counts()
plt.figure(figsize=(8, 5))
class_distribution.plot(kind='bar', color=['darkblue', 'blue', 'skyblue'])
plt.title('Class Distribution of Diabetes_012', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)
plt.show()


# ## DATA PREPROCESSING :

# In[26]:


from sklearn.preprocessing import MinMaxScaler,label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[27]:


df.drop(columns=['Education', 'Income'])
# THESE COLUMNS WERE DORPED BECAUSE THESE TWO ARE NOT GIVING TO MUCH INFORMATION REGARDING THE TARGET 


# In[28]:


def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# In[29]:


df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Diabetes_012'])
numerical_columns = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
df_train = remove_outliers(df_train, numerical_columns)


# In[30]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
source_data[numerical_columns].boxplot(ax=axes[0])
axes[0].set_title("Before Outlier Removal")
df_train[numerical_columns].boxplot(ax=axes[1])
axes[1].set_title("After Outlier Removal")
plt.show()


# In[31]:


scaler = MinMaxScaler()
df_train[numerical_columns] = scaler.fit_transform(df_train[numerical_columns])
df_test[numerical_columns] = scaler.transform(df_test[numerical_columns])


# In[32]:


df_train


# In[33]:


df_test


# In[34]:


plt.figure(figsize=(14, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df_train, x=column, kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[35]:


X_train, y_train = df_train.drop(columns=['Diabetes_012']), df_train['Diabetes_012']
X_test, y_test = df_test.drop(columns=['Diabetes_012']), df_test['Diabetes_012']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[36]:


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(x=y_train,hue=y_train,palette="ch:start=.2,rot=-.3")
plt.title('Class Distribution (Before SMOTE)')
plt.xlabel('Diabetes_012')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled,hue=y_resampled,palette="ch:start=.2,rot=-.3")
plt.title('Class Distribution (After SMOTE)')
plt.xlabel('Diabetes_012')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[37]:


X_resampled


# In[38]:


y_resampled 


# ## MODEL BUILDING :

# In[39]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,accuracy_score


# In[40]:


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    blue_shades = ['darkblue', 'blue', 'skyblue']
    
    print(f"{model_name} Model\n")
    print('Accuracy :',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {i}',color=blue_shades[i % len(blue_shades)])
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()
    
    return {"Model": model_name,"Accuracy": accuracy_score(y_test, y_pred), "AUC ROC": auc_score}


# ## RANDOM FOREST Classifier :

# In[41]:


rf = RandomForestClassifier(random_state=42)
rf.fit(X_resampled, y_resampled)


# In[42]:


rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")


# ## XGBClassifier :

# In[43]:


xgb = XGBClassifier(random_state=42)
xgb.fit(X_resampled, y_resampled)


# In[44]:


xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost")


# ## GradientBoostingClassifier :

# In[45]:


gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_resampled, y_resampled)


# In[46]:


gb_results = evaluate_model(gb, X_test, y_test, "Gradient Boosting")


# ## LGBMClassifier :

# In[47]:


lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_resampled, y_resampled)


# In[48]:


lgbm_results = evaluate_model(lgbm, X_test, y_test, "LightGBM")


# In[49]:


results_df = pd.DataFrame([rf_results, xgb_results, gb_results, lgbm_results])
results_df


# ## HYPER PARAMETER TUNNING :

# In[50]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score


# In[51]:


def evaluate_model_tunned(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted'),
        "AUC ROC": roc_auc_score(y_test, y_proba, multi_class='ovr')
    }


# ## HYPER TUNNING FOR RANDOM FOREST :

# In[52]:


rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_result = evaluate_model_tunned(rf_search.best_estimator_, X_test, y_test)
rf_result["Model"] = "Random Forest"
rf_result["Best Params"] = rf_search.best_params_


# ## HYPER TUNNING FOR XGBCLASSIFIER :

# In[53]:


xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

xgb_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), xgb_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
xgb_result = evaluate_model_tunned(xgb_search.best_estimator_, X_test, y_test)
xgb_result["Model"] = "XGBoost"
xgb_result["Best Params"] = xgb_search.best_params_


# ## HYPER TUNNING FOR GRADIENT BOOSTING :

# In[54]:


gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gb_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
gb_search.fit(X_train, y_train)
gb_result = evaluate_model_tunned(gb_search.best_estimator_, X_test, y_test)
gb_result["Model"] = "Gradient Boosting"
gb_result["Best Params"] = gb_search.best_params_


# ## HYPER TUNNING FOR LGBMCLASSIFIER :

# In[55]:


lgbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [-1, 10, 20]
}

lgbm_search = RandomizedSearchCV(LGBMClassifier(random_state=42), lgbm_param_grid, n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=-1)
lgbm_search.fit(X_train, y_train)
lgbm_result = evaluate_model_tunned(lgbm_search.best_estimator_, X_test, y_test)
lgbm_result["Model"] = "LightGBM"
lgbm_result["Best Params"] = lgbm_search.best_params_


# In[56]:


result = pd.DataFrame([rf_result, gb_result, xgb_result, lgbm_result])
result


# In[57]:


results_df


# ## AFTER MODEL BUILDING FINALE RESULT:
# ### Without Hyper Tunning LightGBM Performs Good With An Accuracy of (0.837761): 83%
# ### With Hyper Tunning All the Model's Accuracy are Similar (0.84): 84%
# ### The model performance is good incase if we increase the parameters it may overfit

# In[ ]:




