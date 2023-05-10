#!/usr/bin/env python
# coding: utf-8

# # CMPINF 2100 Fall 2022 - Project (Main Report)
# 
# ## Siddhesh Shimpukade
# 
# ## Project: PPG Customer Churn Prediction

# ## I. Introduction

# My project was on the **PPG Customer Churn** dataset.
# 
# I was working on a classification problem. It was to predict the probability of a customer not buying a certain product, or a line of products, otherwise called as 'churning'.

# I found that the model which had interactions among all continuous variables was the best performing model, even so in the lasso regression model of it. The accuracy of the model was highest.
# 
# The inputs that I found that were highly influencing were `X19`, and variables from the correlated sets, `X10`, `X13`, and `X18`. The predictive models display the influence of these variables well. Clustering helped a little in observing that the correlated variable clusters were less fuzzy. EDA can help on a preliminary basis, but to see good difference, modeling is important.

# I learned data analysis skills, from basic data manipulation to predictive modeling. I learned the different ways how data can be represented to come towards a strong conclusion in any company/organization. This project helped me gain confidence in using Python for any data analysis project in the future.
# 
# In the future, I wish to get into geopolitical/cyber intelligence analysis, where I believe these skills would be of great help, as every piece of intelligence requires analyzing data or information, and representing it well to policymakers or other stakeholders.

# ## II. Exploratory Data Analysis (EDA)

# ### Import Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[3]:


from patsy import dmatrices
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


# ### Read Data

# In[4]:


churn_data = 'ppg_churn.csv'


# In[5]:


ppg_churn = pd.read_csv(churn_data)


# ### Basic Information

# In[6]:


ppg_churn.shape


# In[7]:


ppg_churn.info()


# In[8]:


ppg_churn.nunique()


# In[9]:


ppg_churn.isna().sum()


# There are 5000 rows, or customers, and 20 columns. The datatypes of the variables are object, integer, and float. This dataset does not have any missing values. There are variations of number of unique values across the variables.

# ### Visualizations: Categorical Variables

# In[10]:


sns.catplot(data = ppg_churn, x='state', kind='count', aspect=2.5)

plt.show()


# In[11]:


sns.catplot(data = ppg_churn, x='X03', kind='count', aspect=1)

plt.show()


# In[12]:


sns.catplot(data = ppg_churn, x='X04', kind='count', aspect=1)

plt.show()


# In[13]:


sns.catplot(data = ppg_churn, x='X05', kind='count', aspect=1)

plt.show()


# ### Visualizations: Binary Output

# In[14]:


sns.catplot(data = ppg_churn, x='churn', kind='count', aspect=1)

plt.show()


# ### Visualizations: Continuous Variables

# In[15]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X02', kde=True, ax=ax)

plt.show()


# In[16]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X06', kde=True, ax=ax)

plt.show()


# In[17]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X07', kde=True, ax=ax)

plt.show()


# In[18]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X08', kde=True, ax=ax)

plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X09', kde=True, ax=ax)

plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X10', kde=True, ax=ax)

plt.show()


# In[21]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X11', kde=True, ax=ax)

plt.show()


# In[22]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X12', kde=True, ax=ax)

plt.show()


# In[23]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X13', kde=True, ax=ax)

plt.show()


# In[24]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X14', kde=True, ax=ax)

plt.show()


# In[25]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X15', kde=True, ax=ax)

plt.show()


# In[26]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X16', kde=True, ax=ax)

plt.show()


# In[27]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X17', kde=True, ax=ax)

plt.show()


# In[28]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X18', kde=True, ax=ax)

plt.show()


# In[29]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X19', kde=True, ax=ax)

plt.show()


# In[30]:


sns.pairplot(data=ppg_churn, height=5, kind='hist', diag_kws={'common_norm':False})

plt.show()


# In[31]:


groups = ppg_churn.X03.unique().tolist()

print(groups)


# In[32]:


fig, axs = plt.subplots(1, len(groups), figsize = (30,10), sharex=True, sharey=True)

for ix, x in enumerate(groups):
    sns.heatmap(data=ppg_churn.groupby(['X03']).corr().loc[x],
               vmin=-1, vmax=1, center=0,
               cmap='coolwarm', cbar=False,
               annot=True, annot_kws={'size': 8},
               ax=axs[ix])
    
    axs[ix].set_title('X03: '+x)
    
plt.show()


# In the given correlation matrix, it can be clearly seen that `X07` & `X09`, `X10` & `X12`, `X13` & `X15`, and `X16` & `X18` are higly correlated. This helps in deciding variables for the model fitting.

# ### Visualizations: Continuous variables interacting with output variable

# In[33]:


fig, ax = plt.subplots(figsize=(12,6))

sns.histplot(data = ppg_churn, x = 'X02', hue='churn', kde=True, common_norm=False, ax=ax)

plt.show()


# We can observe all the continuous variables with respect to the binary output using a pairplot.

# In[34]:


sns.pairplot(data=ppg_churn, height=5, kind='hist', hue='churn', diag_kws={'common_norm':False})

plt.show()


# ### Visualizations: Categorical variables interacting with output variable

# In[35]:


sns.catplot(data = ppg_churn, x='churn', col='X03', kind='count')

plt.show()


# In[36]:


sns.catplot(data = ppg_churn, x='churn', col='X04', kind='count')

plt.show()


# In[37]:


sns.catplot(data = ppg_churn, x='churn', col='X05', kind='count')

plt.show()


# ## III. Clustering Analysis

# In[38]:


churn_clustering = ppg_churn[['X02', 'X06', 'X07', 'X08', 'X09','X10', 
                                'X11', 'X12', 'X13', 'X14', 'X15', 'X16','X17', 'X18', 'X19']].copy()


# In[39]:


churn_clustering_stan = StandardScaler().fit_transform(churn_clustering)


# In[40]:


tots_within = [] 
K=range(1,30)
for k in K:  
    km = KMeans(n_clusters=k, random_state=121, n_init=25, max_iter=500)
    km = km.fit(churn_clustering_stan) 
    tots_within.append(km.inertia_)


# Optimum number of clusters can be found using the silhouette coefficient method. I have taken the number of clusters as 2, with the highest silhouette coefficient. 

# In[41]:


from sklearn.metrics import silhouette_score


# In[42]:


sil_coef = []

K=range(2,30)

for k in K:
    k_label = KMeans(n_clusters=k, random_state=121, n_init=25, max_iter=500).fit_predict( churn_clustering_stan )
    sil_coef.append( silhouette_score(churn_clustering_stan, k_label) )


# In[43]:


fig, ax = plt.subplots(figsize=(12, 8)) 

ax.plot(K, sil_coef, 'o-')
ax.set_xlabel('number of clusters')
ax.set_ylabel('average silhouette coefficient')

plt.show()


# In[44]:


cluster_stan = StandardScaler().fit_transform(churn_clustering.select_dtypes('number'))


# In[45]:


clust_2 = KMeans(n_clusters=2, random_state=121, n_init=25, max_iter=500).fit_predict(cluster_stan)

churn_clustering['clust_2'] = pd.Series(clust_2, index=churn_clustering.index)
churn_clustering['clust_2'] = churn_clustering.clust_2.astype('category') 


# In[46]:


sns.pairplot(churn_clustering, hue='clust_2', height=5, diag_kws={'common_norm':False})

plt.show()


# ## IV. Model - Interpretation

# For fitting the models, I have taken all the numeric values as continuous variables, I converted the churn value to either 1 or 0, and there are three non-numeric categorical variables.
# 
# Since the continuous variables cannot be scaled together properly, I have used StandardScalar() to standardize the variables.

# In[47]:


churn_df_copy = ppg_churn.copy()


# In[48]:


churn_df_copy['y'] = np.where(churn_df_copy['churn'] == 'no', 0, 1)


# In[49]:


churn_cont = churn_df_copy[['X02', 'X06', 'X07', 'X08', 'X09', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19']].copy()


# In[50]:


X_cont = StandardScaler().fit_transform(churn_cont)


# In[51]:


X_df = pd.DataFrame(X_cont, columns=churn_cont.columns)


# In[52]:


X_cat =  churn_df_copy[['X03', 'X04', 'X05', 'churn', 'y']].copy()


# In[53]:


churn_new = pd.concat([X_df, X_cat], axis=1)


# ### Logistic Regression
# 
# Here I have created 8 formulas for the different models, with all combinations of additive, interactions and polynomials. In a few complex models, I have removed one of each pair of highly correlated variables.

# In[54]:


formula_list = ['y ~ X02 + X06 + X07 + X08 + X09 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19',
                'y ~ X03 + X04 + X05',
                'y ~ (X02 + X06 + X07 + X08 + X10 + X11 + X13 + X14 + X17 + X18 + X19) ** 2',
                'y ~ X02 + X06 + X07 + X08 + X09 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X03 + X04 + X05',
                'y ~ X03 * (X02 + X06 + X07 + X08 + X09 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19)',
                'y ~ X05 + (X02 + X06 + X09 + X08 + X11 + X12 + X14 + X15 + X16 + X17 + X19) ** 2',
                'y ~ X06 + X15 + X16 + X17 + X18 + X19 + np.power(X07,2) + np.power(X08,2) + np.power(X09,2) + np.power(X10,2)+ np.power(X13,2)+ np.power(X14,2)',
                'y ~ X19 * (X03 + X04 + X05 + np.power(X07,2) + np.power(X08,2) + np.power(X11,2)+ np.power(X12,2)+ np.power(X13,2)+ np.power(X14,2))']


# In[55]:


model_list = []

for a_formula in formula_list:
    model_list.append( smf.logit( formula = a_formula, data = churn_new).fit(maxiter=500000) )


# In[56]:


def my_coefplot(model_object, figsize_use=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize_use)
    
    ax.errorbar(y = model_object.params.index,
               x = model_object.params,
               fmt = 'o', color = 'black', ecolor='black',
               xerr = 2 * model_object.bse,
               elinewidth = 3, ms=10)
    
    ax.axvline(x = 0, linestyle='--', linewidth=5, color='grey')
    
    ax.set_xlabel('coefficient value')
    
    plt.show()


# In[57]:


print( model_list[0].summary() )


# In[58]:


my_coefplot(model_list[0])


# This is the summary statistics and coefficient plot visualization for the first model. For the other models, I have displayed the information in the supporting document **Shimpukade_Siddhesh_Model_Interpretation**.

# An in-depth process and analysis of fitting the models and determining accuracy and AUC of the models is in the same document.

# In[ ]:





# ## V. Model - Performance and Validation

# The models that I selected based on accuracy and AUC:
# 
# Best : `y ~ (X02 + X06 + X07 + X08 + X10 + X11 + X13 + X14 + X17 + X18 + X19) ** 2`
# 
# Simple: `y ~ X03 + X04 + X05`
# 
# Complex: `y ~ X03 * (X02 + X06 + X07 + X08 + X09 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19)`

# I have used Stratified K-fold for validation.

# In[59]:


my_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=101)


# ### 1) Best Model

# In[60]:


best_model = formula_list[2]


# In[61]:


y_best, X_best = dmatrices(best_model, churn_new, return_type='dataframe')


# **Ridge and Lasso Model**

# In[62]:


best_model_ridge = LogisticRegressionCV(penalty='l2', Cs = 101, cv=my_cv, solver='lbfgs', max_iter=25001, fit_intercept=False).                                        fit(X_best, y_best.values.ravel())

best_model_lasso = LogisticRegressionCV(penalty='l1', Cs = 101, cv=my_cv, solver='saga', max_iter=25001, fit_intercept=False).                                        fit(X_best, y_best.values.ravel())


# **Optimum regularization strength**

# In[63]:


best_model_ridge_op = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=25001, fit_intercept=False, C=2.51188643).                                         fit(X_best, y_best.values.ravel())

best_model_lasso_op = LogisticRegression(penalty='l1', solver='saga', max_iter=25001, fit_intercept=False, C=2.08929613).                                         fit(X_best, y_best.values.ravel())

best_model_nopenalty_op = LogisticRegression(penalty='none', solver='lbfgs', max_iter=25001, fit_intercept=False).                                             fit(X_best, y_best.values.ravel())


# In[64]:


best_model_performance = pd.DataFrame({'Best Model': ['Ridge Model', 'Lasso Model', 'No Penalty Model'],
                                       'Accuracy': [best_model_ridge_op.score(X_best, y_best.values.ravel()),
                                                    best_model_lasso_op.score(X_best, y_best.values.ravel()),
                                                    best_model_nopenalty_op.score(X_best, y_best.values.ravel())]})


# In[65]:


best_model_performance


# Similar accuracies are seen in ridge, lasso and no penalty models.

# **Coefficent Comparison**

# In[66]:


coef_compare_best = pd.DataFrame({'Coefficient': list(X_best.columns),
                                  'Ridge Model': list(best_model_ridge_op.coef_.ravel()),
                                  'Lasso Model': list(best_model_lasso_op.coef_.ravel()),
                                   'No Penalty' : list(best_model_nopenalty_op.coef_.ravel())})


# In[67]:


coef_compare_best


# In[68]:


fig, ax = plt.subplots(figsize=(12,6))

ax.plot(np.log(best_model_ridge.Cs_), best_model_ridge.scores_[1.0].mean(axis=0), color='green', label='Ridge Model (Best)')
ax.plot(np.log(best_model_lasso.Cs_), best_model_lasso.scores_[1.0].mean(axis=0), color='grey', label='Lasso Model (Best)')

ax.set_xlabel('log(C)')
ax.set_ylabel('Cross-validation Accuracy')
ax.legend()

plt.show()


# In[69]:


coef_compare_best.loc[np.abs(coef_compare_best['Lasso Model']) > 0, :].shape[0]


# In[70]:


coef_compare_best['No Penalty'].shape[0]


# Hence, the lasso model turns the no penalty model from 67 features to 64.

# This is the process and coefficient comparison for the best model. An in-depth process and analysis regarding the other two models are in the supporting document **Shimpukade_Siddhesh_Model_Performance**.

# In this document I have also displayed the overall performance visualization of the 6 models.

# From all the models, the lasso version of the best model is the best performing one.

# ## VI. Model - Prediction

# For prediction, I have considered the lasso version of the best model, with the optimum regularization strength.

# In[71]:


best_model_lasso_op


# In[72]:


best_model


# In[85]:


input_grid_a = pd.DataFrame([(X02, X06, X07, X08, X10, X11, X13, X14, X17, X18, X19) for X02 in np.linspace(-4, 4, num=25)
                             for X06 in [-2., 2.]
                             for X07 in [-3., 3.]
                             for X08 in [-5., 5.]
                             for X10 in [-2., 0., 2.]
                             for X11 in [-3., 3.]
                             for X13 in [-4., 0., 4.]
                             for X14 in [-6., 0., 6.]
                             for X17 in [-2., 2.]
                             for X18 in [-2., 2.]
                             for X19 in [-4., 4.]],
                             columns = ['X02', 'X06', 'X07', 'X08', 'X10', 'X11', 'X13', 'X14', 'X17', 'X18', 'X19'])


# In[86]:


input_grid_a.info()


# In[87]:


input_grid_a.nunique()


# In[96]:


best_model_fit = smf.logit(best_model, data=churn_new).fit()


# In[97]:


input_grid_a['pred_probability'] = best_model_fit.predict(input_grid_a)


# In[98]:


sns.relplot(data = input_grid_a, x='X02', y='pred_probability', hue='X10',
           col='X18', row='X17', kind='line')

plt.show()


# In[99]:


sns.relplot(data = input_grid_a, x='X06', y='pred_probability', hue='X14',
           col='X13', row='X19', kind='line')

plt.show()

