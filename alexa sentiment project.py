#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re


# In[4]:


data = pd.read_csv(
    "C:/Users/gilld/Downloads/archive (13)/amazon_alexa.tsv",
    delimiter="\t",
    quoting=3
)

print(f"Dataset shape : {data.shape}")


# In[5]:


data.head()


# In[6]:


#column names

print(f"Features names : {data.columns.values}")


# In[7]:


# check for null values

data.isnull().sum()


# In[10]:


# getting the record where' verified-reviews' is null

data[data['verified_reviews'].isna() ==True]


# In[11]:


# we will drop the null record

data.dropna(inplace=True)


# In[12]:


print(f"Dataset shape after dropping null values : {data.shape}")


# In[13]:


# creating a new column 'length' that will contain the length of the string in 'verified_reviews' column

data['length']= data['verified_reviews'].apply(len)


# In[14]:


data.head()


# In[15]:


# The 'length' column is new generated column - stores the length of 'verified_reviews' for that record.
# Let's check for some sample records

# Randomly checking for 10th record
print(f"'verified_reviews' column value: {data.iloc[10]['verified_reviews']}")  # original value
print(f"Length of review : {len(data.iloc[10]['verified_reviews'])}")            # length of review using len()
print(f"'length' column value : {data.iloc[10]['length']}")                      # value of the column 'length'


# In[16]:


data.dtypes


# In[18]:


len(data)


# In[17]:


# distint values of 'rating' and its count

print(f"Rating value count: \n{data['rating'].value_counts()}")


# In[19]:


# Bar plot to visualize the total counts of each rating
data['rating'].value_counts().plot.bar(color='red')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()


# In[20]:


# Finding the percentage distribution of each rating
# we'll divide the number of records for each rating by total number of records

print(
    f"Rating value count - percentage distribution:\n"
    f"{round(data['rating'].value_counts() / data.shape[0] * 100, 2)}"
)


# In[21]:


fig = plt.figure(figsize=(7, 7))

colors = ('red', 'green', 'blue', 'orange', 'yellow')
wp = {'linewidth': 1, 'edgecolor': 'black'}

tags = data['rating'].value_counts() / data.shape[0]

explode = (0.1, 0.1, 0.1, 0.1, 0.1)

tags.plot(
    kind='pie',
    autopct='%1.1f%%',
    shadow=True,
    colors=colors,
    startangle=90,
    wedgeprops=wp,
    explode=explode,
    label='Percentage'
)

from io import BytesIO
graph = BytesIO()

fig.savefig(graph, format="png")


# In[22]:


# Distinct values of 'feedback' and its count
print(f"Feedback value count:\n{data['feedback'].value_counts()}")


# In[23]:


# Extracting the 'verified_reviews' value for one record with feedback = 0
review_0 = data[data['feedback'] == 0].iloc[1]['verified_reviews']
print(review_0)


# In[24]:


# Extracting the 'verified_reviews' value for one record with feedback = 0
review_1 = data[data['feedback'] == 1].iloc[1]['verified_reviews']
print(review_1)


# In[25]:


# Bar graph to visualize the total counts of each feedback
data['feedback'].value_counts().plot.bar(color='blue')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()


# In[26]:


# Finding the percentage distribution of each feedback
# we'll divide the number of records for each feedback by total number of records

print(
    f"Feedback value count - percentage distribution:\n"
    f"{round(data['feedback'].value_counts() / data.shape[0] * 100, 2)}"
)


# In[27]:


fig = plt.figure(figsize=(7,7))

colors = ('red', 'green')
wp = {'linewidth':1, 'edgecolor':'black'}

tags = data['feedback'].value_counts()/data.shape[0]

explode=(0.1,0.1)

tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage')


# In[28]:


#Feedback = 0
data[data['feedback'] == 0]['rating'].value_counts()


# In[29]:


#Feedback = 1
data[data['feedback'] == 1]['rating'].value_counts()


# In[31]:


# distinct values of 'variation' and its count 

print(f"variation value count: \n{data['variation'].value_counts()}")


# In[32]:


# bar graph to visualize the total counts of each variation

data['variation'].value_counts().plot.bar(color = 'orange')
plt.title('Variation distribution count')
plt.xlabel('variation')
plt.ylabel('count')
plt.show()


# In[33]:


# finding the % distribution of each variation - we'll divide the number ofrecords for each variation by total number of

print(f"Variation value count - percentage distribution: \n{round(data['variation'].value_counts()/data.shape[0]*100,2)}")


# In[34]:


data.groupby('variation')['rating'].mean()


# In[36]:


data.groupby('variation')['rating'].mean().sort_values().plot.bar(color = 'brown', figsize=(11,6))
plt.title("Mean rating according to variation")
plt.xlabel('Variation')
plt.ylabel('Mean rating')
plt.show()


# In[37]:


data['length'].describe()


# In[39]:


sns.histplot(data['length'],color='blue').set(title='Distribution of length of review')


# In[41]:


sns.histplot(data[data['feedback']==0]['length'],color= 'red').set(title='Distribution of length of review if feedback = 0')


# In[43]:


sns.histplot(data[data['feedback']==1]['length'],color= 'green').set(title='Distribution of length of review if feedback = 1')


# In[44]:


data.groupby('length')['rating'].mean().plot.hist(color = 'blue', figsize=(7,6),bins = 20)
plt.title("Review lenght wise mean rating")
plt.xlabel('rating')
plt.ylabel('length')
plt.show()


# In[45]:


cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(data.verified_reviews)


# In[47]:


# Combine all reviews
reviews = " ".join([review for review in data['verified_reviews']])

# Initialize wordcloud object
wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[48]:


# Combine all reviews for each feedback category and splitting them into individual words
neg_reviews = " ".join(review for review in data[data['feedback'] == 0]['verified_reviews'])
neg_reviews = neg_reviews.lower().split()

pos_reviews = " ".join(review for review in data[data['feedback'] == 1]['verified_reviews'])
pos_reviews = pos_reviews.lower().split()

# Finding words from reviews which are present in that feedback category only
unique_negative = [x for x in neg_reviews if x not in pos_reviews]
unique_negative = " ".join(unique_negative)

unique_positive = [x for x in pos_reviews if x not in neg_reviews]
unique_positive = " ".join(unique_positive)


# In[49]:


wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('Wordcloud for negative reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[ ]:


# wc = WordCloud(background_color='white', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('Wordcloud for positive reviews', fontsize=10)
plt.axis('off')
plt.show()


# # Preprocessing and modelling
# 
# To build the corpus from the'verfied_reviews' we perfom the following -
# 1. Replace any non alphabet character with space
# 2. convert to lower case and split into words
# 3. iterate over the undividual words and if it is not a stopword then add the stemmed from of the word to the corpus

# In[52]:


import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
stemmer = PorterStemmer()

for i in range(0, data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    corpus.append(review)


# In[55]:


import os

if not os.path.exists('Models'):
    os.makedirs('Models') 

pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
import pickle

cv = CountVectorizer(max_features = 2500)

# Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# Saving the Count Vectorizer
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Splitting data into train and test set with 30% data with testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")

# We'll scale X_train and X_test so that all values are between 0 and 1.
scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# Saving the scaler model
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = 15)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {X_test.shape}")


# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Fitting scaled X_train and y_train on Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)

# Accuracy of the model on training and testing data
print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))

# Predicting on the test set
y_preds = model_rf.predict(X_test_scl)

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_rf.classes_)
cm_display.plot()
plt.show()


# # k fold cross validation

# In[61]:


from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = model_rf, X = X_train_scl, y = y_train, cv = 10)

print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())


# In[66]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}

cv_object = StratifiedKFold(n_splits = 2)

grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_scl, y_train.ravel())


# In[67]:


# Getting the best parameters from the grid search
print("Best Parameter Combination : {}".format(grid_search.best_params_))


# In[68]:


print("Cross validation mean accuracy on train set : {}".format(grid_search.cv_results_['mean_train_score'].mean()*100))
print("Cross validation mean accuracy on test set : {}".format(grid_search.cv_results_['mean_test_score'].mean()*100))
print("Accuracy score for test set :", accuracy_score(y_test, y_preds))


# # XGBoost Classifier 

# In[69]:


from xgboost import XGBClassifier

# Fitting scaled X_train and y_train on XGBoost Classifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)


# In[70]:


# Accuracy of the model on training and testing data
print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))


# In[73]:


y_preds = model_xgb.predict(X_test)


# In[74]:


# confusion matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)


# In[76]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
cm_display.plot()
plt.show()


# In[79]:


# saving the XGBoost classifier
pickle.dump(model_xgb, open('Models/model_xgb.pkl', 'wb'))


# # decision tree classifier 

# In[80]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)


# In[81]:


# accuracy of the model on training and testing data

print("Training Accuracy :", model_dt.score(X_train_scl, y_train))
print("Testing Accuracy :", model_dt.score(X_test_scl, y_test))


# In[82]:


y_preds = model_dt.predict(X_test)


# In[83]:


#confusion matrix
cm = confusion_matrix(y_test,y_preds)
print(cm)


# In[84]:


cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
cm_display.plot()
plt.show()







