#!/usr/bin/env python
# coding: utf-8

# # Marketing Campaign For Banking Products
Bank is has a growing customer base. The bank wants to increase borrowers (asset customers) base to bring in more loan business and earn more through the interest on loans. So , bank wants to convert the liability based customers to personal loan customers. (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. The department wants to build a model that will help them identify the potential customers who have higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign.


# # 1 Importing the Required Libraries and the Dataset
# 

# In[3]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_excel(r'C:\Users\Nayan Kumar\Desktop\Bank_Personal_Loan_Modelling.xlsx','Data')


# In[5]:


#To display top 5 row
data.head()


# In[6]:


#to dislpays the columns
data.columns


# In[7]:


#to display the last 5 rows
data.tail()


# There are 12 features.The aim is to construct a model that can  identify potential customers who have a higher probability of purchasing loan. Output column is personal loan. Features are detailed below:
ID:                      Customer ID
Age:                     Customer's age in completed years
Experience:              Number years of professional experience
Income:                  Annual income of the customer
ZIPCode:                 Home Address ZIP code.
Family:                  Family size of the customer
CCAvg :                  Avg. Spending on credit cards per month
Education:               Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional
Mortgage:                Value of house mortgage if any.
Personal Loan:           Did this customer accept the personal loan offered in the last campaign?
Securities Account:      Does the customer have a securities account with the bank?
CD Account:              Does the customer have a certificate of deposit (CD) account with the bank?
Online:                  Does the customer use internet banking facilities?
CreditCard:              Does the customer use a credit card issued by UniversalBank?


# # 1.1 Checking the type of data and basic summary

# In[8]:


data.shape


# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


data.describe().transpose()


# In[12]:


data.isnull().sum()


# # 2. Dropping Irrelevant Columns
In a machine learning,it is neccessary to seperate signal from the noice.Hence the ID column which definitely doesn't have any signal is being dropped.Also ,the experiance column seens to have faulty data as soon values are negative.We can replace and impute those values,but i choose to drop this column as well ,as it seems to be highly correlated with the age column.

# In[13]:


experience=data['Experience']
age=data['Age']
correlation=experience.corr(age)
correlation


# In[14]:


data=data.drop(['ID','Experience'],axis=1)
data.head()


# # 3. EDA 

# In[15]:


#number of unique in each column
data.nunique()

ZIP Code has 467 distinct values.
It is nominal variable which has too many levels.
It's better to drop ZIP Code as well.
# In[16]:


data.drop('ZIP Code', axis=1)


# In[17]:


#Number of people with zero mortage
(data.Mortgage==0).sum()


# In[18]:


#Number of people with 0 credit card spending  per month
(data.CCAvg==0).sum()


# In[19]:


# Value count for all categorial columns
data.Family.value_counts()


# In[20]:


data.Education.value_counts()


# In[21]:


data['Securities Account'].value_counts()


# In[22]:


data['CD Account'].value_counts()


# In[23]:


data['CreditCard'].value_counts()


# In[24]:


data.Online.value_counts()


# # Univariate Analysis 

# In[25]:


# Univariate Analysis
# Age seems to have a symmetric distribution
sns.distplot(data.Age);


# In[26]:


# Income is right skewed Distribution
sns.distplot(data.Income);


# In[27]:


# Credit Card Average is right skewed Distribution
sns.distplot(data.CCAvg);


# In[28]:


# Mortgage seems to be highly skewed
sns.distplot(data.Mortgage);


# In[29]:


sns.countplot(data.Family);


# In[30]:


sns.countplot(data.Education);


# # MultiVariate Analysis
# 

# In[31]:


# It seems that the customers who has more Income is granted loan across each Education level
sns.boxplot(x='Education', y='Income',hue='Personal Loan', data=data);


# In[32]:


# Majority of people having Security Account don't have Personal loan
sns.countplot(x='Securities Account', hue='Personal Loan',data=data);


# In[33]:


# After dropping experience column earlier doesn't seem to be significant correlation between other object variable
# Credit Card Average and Income
fig,ax=plot.subplots(figsize=(15,10))
sns.heatmap(data.corr(),cmap='plasma',annot=True);


# In[34]:


sns.pairplot(data)


# In[37]:


data_X=data.loc[:,data.columns != "Personal Loan"]
data_Y=data[["Personal Loan"]]


# # 4. TRANSFORMATION OF FEATURE VARIABLE

# In[51]:


from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method="yeo-johnson", standardize=False)
pt.fit(data_X["Income"].values.reshape(-1,1))
sns.distplot(pt.transform(data_X["Income"].values.reshape(-1,1)))


# In[52]:


pt=PowerTransformer(method="yeo-johnson", standardize=False)
pt.fit(data_X["CCAvg"].values.reshape(-1,1))
sns.distplot(pt.transform(data_X["CCAvg"].values.reshape(-1,1)))


# In[40]:


data_X["Mortgage_Int"]=pd.cut(data_X["Mortgage"],
                              bins=[0,100,200,300,400,500,600,700],
                              labels=[0,1,2,3,4,5,6],
                              include_lowest=True)
data_X.drop("Mortgage",axis=1,inplace=True)


# In[41]:


data_X.head()


# In[45]:


## Univariate analysis
# 9.6% of all the applicants get approved for Personal Loan
tempDF=pd.DataFrame(data["Personal Loan"].value_counts()).reset_index()
tempDF.columns=["Labels","Personal Loan"]
fig1,ax1=plot.subplots(figsize=(10,8))
explode=(0,0.15)
ax1.pie(tempDF["Personal Loan"],explode=explode,autopct='%1.1f%%',
        shadow=True,startangle=70)
ax1.axis('equal')   #Equal aspect ratio ensures that pie is drawn as a circle
plot.title("Personal Loan Percentage")
plot.show()


# # 5. Splitting the data using stratified sampling

# In[46]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_Y,test_Y =train_test_split(data_X,data_Y,test_size=0.3,stratify=data_Y,random_state=0)


# In[47]:


train_X.reset_index(drop=True, inplace=True);
test_X.reset_index(drop=True, inplace=True);
train_Y.reset_index(drop=True, inplace=True);
test_Y.reset_index(drop=True, inplace=True);


# In[48]:


train_X.head()


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
scy = StandardScaler()
train_X,test_X,train_Y,test_Y = train_test_split(data_X,data_Y,test_size = 0.3, random_state = 0,stratify = data_Y)
scx.fit_transform(train_X)
scx.transform(test_X)


# 
# # LOGISTIC REGRESSION

# In[54]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[56]:


model.fit(train_X,train_Y)


# In[57]:


y_pred = model.predict(test_X)


# In[58]:


from sklearn import metrics


# In[63]:


print("MAE:",metrics.mean_absolute_error(test_Y,y_pred))
print("R2 score:",metrics.r2_score(test_Y,y_pred))


# In[64]:


print(metrics.accuracy_score(test_Y,y_pred))


# In[66]:


print(metrics.accuracy_score(train_Y,model.predict(train_X)))


# In[67]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y,y_pred))


# In[68]:


from sklearn.metrics import classification_report
print(classification_report(test_Y,y_pred))

 So we can see that accuracy score for our test data is 91.07%.
# # DECISION TREE

# In[69]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(train_X,train_Y)


# In[76]:


import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=['Age','Income','Family','CCAvg','Online',
                                                                  'CreditCard','Education', 'Mortgage', 'Securities Account', 
                                                                  'CD Account','Personal Loan'],filled=True, rounded=True) 
graph= graphviz.Source(dot_data)


# In[77]:


graph


# In[79]:


y_pred = clf.predict(test_X)


# In[80]:


from sklearn import metrics
print("MAE:",metrics.mean_absolute_error(test_Y,y_pred))
print("R2 score:",metrics.r2_score(test_Y,y_pred))
print("Accuracy score for test data",metrics.accuracy_score(test_Y,y_pred))


# In[81]:


print("Accuracy score for train data",metrics.accuracy_score(train_Y, model.predict(train_X)))


# In[82]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y,y_pred))


# In[83]:


from sklearn.metrics import classification_report
print(classification_report(test_Y,y_pred))

It is working very much better than previous algorithm. It is showing 97% accuracy score
# # RANDOM FOREST

# In[84]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(train_X,train_Y)


# In[85]:


y_pred = clf.predict(test_X)


# In[86]:


print("MAE:",metrics.mean_absolute_error(test_Y,y_pred))
print("R2 score:",metrics.r2_score(test_Y,y_pred))
print("Accuracy score for test data",metrics.accuracy_score(test_Y,y_pred))


# In[90]:


print("Accuracy score for train data",metrics.accuracy_score(train_Y,model.predict(train_X)))


# In[88]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y,y_pred))


# In[89]:


from sklearn.metrics import classification_report
print(classification_report(test_Y,y_pred))

It is clearly showing that the model of random forest is working very good. 
The accuracy score is 98% for test data which is highest in all the algorithm we have used till now.
Also the confusion matrix is also better than the previous results.
# # NAIVE BAYES

# In[91]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_X,train_Y)


# In[92]:


y_pred = model.predict(test_X)


# In[93]:


print("MAE:",metrics.mean_absolute_error(test_Y,y_pred))
print("R2 score:",metrics.r2_score(test_Y,y_pred))
print("Accuracy score for test data",metrics.accuracy_score(test_Y,y_pred))


# In[94]:


print("Accuracy score for train data",metrics.accuracy_score(train_Y,model.predict(train_X)))


# In[95]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y,y_pred))


# In[96]:


from sklearn.metrics import classification_report
print(classification_report(test_Y,y_pred))

The accuracy score for NAIVE BAYES algorithm  is not very good.
From all the models it is least till now. 
It is 88% which is less than other algorithms.
# # KNN ALOGORITHM

# In[97]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_X,train_Y)


# In[98]:


y_pred = neigh.predict(test_X)


# In[99]:


print("MAE:",metrics.mean_absolute_error(test_Y,y_pred))
print("R2 score:",metrics.r2_score(test_Y,y_pred))
print("Accuracy score for test data",metrics.accuracy_score(test_Y,y_pred))


# In[100]:


print("Accuracy score for train data",metrics.accuracy_score(train_Y,model.predict(train_X)))


# In[101]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y,y_pred))


# In[102]:


from sklearn.metrics import classification_report
print(classification_report(test_Y,y_pred))

KNN ALGORITHM is working better than NAIVE BAYES algorithm. 
But it's performance is not so good when compared with other previous algorithms.
# # MODEL COMPARISON

# In[106]:


X=data.drop(['Personal Loan'],axis=1)
y=data.pop('Personal Loan')


# In[115]:


from sklearn import model_selection

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=12345)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plot.boxplot(results)
ax.set_xticklabels(names)
plot.show()


# # CONCLUSION: 
In the first step of this project we imported various libraries and our data. Than we found out various things about our data.

1) We have to make the model to predict whether a person will take personal loan or not.
2) We found that age and experience are highly correlated so we droped the experience column.
3) ID and ZIPcode were not contributing factors for a person to take loan so we dropped them.
4) The Income and CCAvg column were left skewed so we applied Power transformation to them to normalize them.
5) The mortgage column was also skewed but since it was discrete so rather than power transformation, we use binning technique.
# After this we used several models to make predictions.
Accuracy Score of the predicted models are as follows:
RANDOM FOREST:
RF:  0.986600

DECISION TREE:
DT:  0.981400

KNN ALGORITHM:
KNN: 0.898600

NAIVE BAYES :
NB:  0.885600 
 
# The aim of the universal bank is to convert there liability customers into loan customers. 
# They want to set up a new marketing campaign.
# Hence, they need information about the connection between the variables given in the data.
# Four classification algorithms were used in this study. 
# From the above graph , it seems like DECISION TREE AND RANDOM FOREST are among those algorithm 
# which have the highest accuracy and we can choose those as our final model

# In[ ]:




