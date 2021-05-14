#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import pickle
import random


# In[2]:


with open('project/data/listings_final.pickle', 'rb') as read_file:
    listings_all = pickle.load(read_file)
    
listings_all.head()


# In[3]:


#Clean up regions
def region_cleaned(region):
    if region == 'kirkwood':
        return 'Kirkwood'
    elif region == 'INCLINE VILLAGE':
        return 'Incline Village'
    elif region == 'Olympic valley':
        return "Olympic Valley"
    else: return region


# In[4]:


listings_all['region_cleaned'] = listings_all['region'].apply(region_cleaned)
listings_all.drop(columns ='region', inplace = True)


# In[5]:


listings_allv1 = listings_all[(listings_all['cleanliness'].notnull()) &
                              (listings_all['num_reviews'].notnull()) &
                              (listings_all['response_rate'].notnull())]


# In[6]:


listings_allv1.describe()


# In[7]:


listings_allv1 = listings_allv1[listings_all['price'] < 600] #remove extreme high price values


# In[8]:


#Remapping property types
def type_remapping(type_desc):
    if "Entire house" in type_desc:
        return "House"
    elif "Entire apartment" in type_desc:
        return "Apartment"
    elif "Entire serviced apartment" in type_desc:
        return "Apartment"
    elif "Entire townhouse" in type_desc:
        return "House"
    elif "Entire loft" in type_desc:
        return "Apartment"
    elif "cottage" in type_desc:
        return "House"
    elif "Entire chalet" in type_desc:
        return "House"
    elif "Entire condominium" in type_desc:
        return "Condominium"
    else:
        return type_desc


# In[9]:


listings_allv1['type_desc_remapping'] = listings_allv1['type_desc'].map(lambda x: type_remapping(x))


# In[10]:


# Regroup property types into three major categories to reduce the number of features
listings_allv1.loc[~listings_allv1.type_desc_remapping.isin(["House","Condominium"]), "type_desc_remapping"] = "Other_Type"


# In[11]:


listings_allv1.type_desc_remapping.value_counts()


# In[12]:


type_dummies = pd.get_dummies(listings_allv1['type_desc_remapping'])


# In[13]:


listings_allv1 = listings_allv1.join(type_dummies)
listings_allv1.drop(columns = ['Other_Type'], inplace = True)


# In[14]:


identify_dummies = pd.get_dummies(listings_allv1['identify_verify'])
listings_allv1 = listings_allv1.join(identify_dummies)

listings_allv1.drop(columns = ['Not Verified'], inplace = True)


# In[15]:


host_status_dummies = pd.get_dummies(listings_allv1['host_status'])
listings_allv1 = listings_allv1.join(host_status_dummies)

listings_allv1.drop(columns = ['Host'], inplace = True)


# In[16]:


listings_allv1['region_cleaned'].value_counts()


# In[17]:


by_lake = ['South Lake Tahoe','Kings Beach', 'Carnelian Bay', 'Tahoe City','Incline Village', 'Stateline', 'Tahoe Vista', 'Zephyr Cove', 'Tahoma', 'Sunnyside-Tahoe City', 'Dayton', 'Glenbrook','Placer County', 'Homewood']


# In[18]:


listings_allv1["neigborhood"] = np.where(listings_allv1["region_cleaned"].isin(by_lake), "By_lake_neighbor", "Other_neighbor")


# In[19]:


neigbor_dummies = pd.get_dummies(listings_allv1['neigborhood'])
listings_allv1 = listings_allv1.join(neigbor_dummies)
listings_allv1.drop(columns = ['Other_neighbor'], inplace = True)


# In[20]:


listings_allv1.info()


# In[21]:


regions_count = listings_allv1['region_cleaned'].value_counts()
region_others = list(regions_count[regions_count <= 8].index)

listings_allv1['region'] = listings_allv1['region_cleaned'].replace(region_others, 'Other_regions')


# In[22]:


region_dummies = pd.get_dummies(listings_allv1['region'])
listings_allv1 = listings_allv1.join(region_dummies)
listings_allv1.drop(columns = ['Stateline'], inplace = True)


# In[23]:


listings_allv1.info()


# In[24]:


columns = ['reviews','num_reviews','num_guets','num_bedrooms','num_baths','num_beds',
            'cleanliness','accuracy','communication','location','checkin',
            'value','response_rate','TV','Free_Parking','Wifi','Kitchen','Heating', 'Air_conditioning',
            'Identity verified','Superhost','House','Condominium','By_lake_neighbor','Carnelian Bay','Incline Village',
            'Kings Beach','Kirkwood','Olympic Valley','Other_regions','Reno','South Lake Tahoe','Tahoe City','Tahoe Vista',
            'Truckee','price']

listings_allv1 = listings_allv1[columns]


# In[25]:


sns.pairplot(listings_allv1, corner = True)


# In[27]:


corr_df = listings_allv1.corr()
corr_df


# ### Iteration 1: Base

# In[77]:


features = ['reviews','num_reviews','num_guets','num_bedrooms','num_baths','num_beds',
            'cleanliness','accuracy','communication','location','checkin',
            'value','response_rate','TV','Free_Parking','Wifi','Kitchen','Heating', 'Air_conditioning',
            'Identity verified','Superhost','House','Condominium','By_lake_neighbor','Carnelian Bay','Incline Village',
            'Kings Beach','Kirkwood','Olympic Valley','Other_regions','Reno','South Lake Tahoe','Tahoe City','Tahoe Vista',
            'Truckee']


y = listings_allv1['price']
X = listings_allv1[features]


# In[78]:


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.25, random_state = 40)


# In[79]:


#fit model
lr_model_1 = LinearRegression()
lr_model_1.fit(X_train, y_train)

#Scores
train_score = lr_model_1.score(X_train, y_train)
val_score = lr_model_1.score(X_val, y_val)

#Adj R2
adjusted_r2 = 1 - (1-lr_model_1.score(X_val, y_val))*(len(y_val)-1)/(len(y_val)-X_val.shape[1]-1)


# In[80]:


#print outputs
print('\nTrain R^2 score was:', train_score)
print('\nValidation R^2 score was:', val_score)
print('\nValidation adjusted R^2 was:',adjusted_r2)
print('\nFeature coefficient results: \n')
for feature, coef in zip(X_train.columns, lr_model_1.coef_):
    print(feature, ':', f'{coef:.2f}') 


# In[81]:


preds = lr_model_1.predict(X_test)

sns.jointplot(x=preds,y=y_test, kind='reg')


# In[82]:


#RMSE
def RMSE(actuals, preds): #root mean squared error
    return np.sqrt(np.mean((actuals - preds)**2))

RMSE(y_test,preds)


# In[83]:


import scipy.stats as stats


# In[84]:


res = y_test - preds

stats.probplot(res, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")


# In[85]:


plt.scatter(preds, res)
plt.title("Residual plot")
plt.xlabel("prediction")
plt.ylabel("residuals")


# In[ ]:





# ### Final Model

# In[86]:


X3_train = X_train.copy()
y3_train = y_train.copy()
X3_val = X_val.copy()
y3_val = y_val.copy()
X3_test = X_test.copy()
y3_test = y_test.copy()


# In[87]:


#Create a combo score of cleanliness,accuracy,communication and value to reduce collinearity
X3_train['score_combo'] = np.mean(X_train[['cleanliness','accuracy','communication','value','checkin']], axis = 1)
X3_val['score_combo'] = np.mean(X_val[['cleanliness','accuracy','communication','value','checkin']], axis = 1)
X3_test['score_combo'] = np.mean(X_test[['cleanliness','accuracy','communication','value','checkin']], axis = 1)


# In[88]:


features = ['num_guets','num_bedrooms','num_baths','num_beds',
            'location','score_combo','response_rate','TV','Free_Parking','Wifi','Kitchen','Heating', 'Air_conditioning',
            'Identity verified','Superhost','House','Condominium','By_lake_neighbor']


X3_train = X3_train[features]
X3_val = X3_val[features]


# In[182]:


from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler


# In[183]:


X, y = np.array(X3_train), np.array(y3_train)


# In[188]:


kf = KFold(n_splits=5, shuffle=True, random_state = 20)
cv_lm_r2s, cv_lm_lasso_r2s = [], [] #collect the validation results for both models

for train_ind, val_ind in kf.split(X,y):
    
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind] 
    
    #simple linear regression
    lm = LinearRegression()
    lm_lasso = LassoCV()

    lm.fit(X_train, y_train)
    cv_lm_r2s.append(lm.score(X_val, y_val).round(3))
    
    #ridge with feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lm_lasso.fit(X_train_scaled, y_train)
    cv_lm_lasso_r2s.append(lm_lasso.score(X_val_scaled, y_val).round(3))

print(f'Simple scores: \t{cv_lm_r2s}')
print(f'Lasso scores: \t{cv_lm_lasso_r2s} \n')

print(f'Simple mean cv r^2: {np.mean(cv_lm_r2s):.3f} +- {np.std(cv_lm_r2s):.3f}')
print(f'Lasso mean cv r^2: {np.mean(cv_lm_lasso_r2s):.3f} +- {np.std(cv_lm_lasso_r2s):.3f}')


# Lasso shows slightly higher R^2

# ### Lasso

# In[189]:


scaler = StandardScaler()
X3_train_scaled = scaler.fit_transform(X3_train)
X3_val_scaled = scaler.transform(X3_val)


# In[190]:


#Create the model
lasso = LassoCV()
lasso.fit(X3_train_scaled,y3_train)
lasso.score(X3_train_scaled,y3_train)


# In[191]:


# Score on validation set
lasso.score(X3_val_scaled, y3_val)


# In[192]:


X3_test = X3_test[features]


# In[193]:


X3_test_scaled = scaler.transform(X3_test)
preds3_lasso = lasso.predict(X3_test_scaled)

h = sns.jointplot(x=preds3_lasso,y=y3_test, kind='reg')

h.set_axis_labels('Predicted Price', 'Actual Price', fontsize=12)

h.fig.suptitle('Predicted vs. Actual',fontsize=18)
h.fig.subplots_adjust(top=0.90)
("")


# In[194]:


#Calculate RMSE
RMSE(y3_test,preds3_lasso)


# In[195]:


res3_lasso = y3_test - preds3_lasso

stats.probplot(res3_lasso, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")


# In[196]:


plt.scatter(preds3_lasso, res3_lasso, alpha=0.4)
plt.title("Residual plot")
plt.xlabel("prediction")
plt.ylabel("residuals")


# In[198]:


list(zip(X3_train.columns,lasso.coef_))


# In[ ]:





# ### Try regular linear regression

# In[120]:


#fit model
lr_model_3 = LinearRegression()
lr_model_3.fit(X3_train, y3_train)

#Scores
train_score3 = lr_model_3.score(X3_train, y3_train)
val_score3 = lr_model_3.score(X3_val, y3_val)


# In[121]:


#print features
print('Feature coefficient results: \n')
for feature, coef in zip(X3_train.columns, lr_model_3.coef_):
    print(feature, ':', f'{coef:.2f}') 


# In[122]:


X3_test = X3_test[features]

preds3 = lr_model_3.predict(X3_test)

sns.jointplot(x=preds3,y=y3_test, kind='reg');


# In[123]:


#Calculate RMSE
RMSE(y3_test,preds3)


# In[124]:


import scipy.stats as stats


# In[125]:


res3 = y3_test - preds3

stats.probplot(res3, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")


# In[180]:


plt.scatter(preds3, res3,alpha = 0.4)
plt.title("Residual plot")
plt.xlabel("prediction")
plt.ylabel("residuals")
    


# In[ ]:




