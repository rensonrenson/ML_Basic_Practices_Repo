# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# COMMAND ----------

df = spark.read.csv('/FileStore/tables/mail_data.csv',header = True)

# COMMAND ----------

df_initial = df.toPandas()

# COMMAND ----------

print(df_initial)

# COMMAND ----------

data=df_initial.where((pd.notnull(df_initial)),'')

# COMMAND ----------

data.head()

# COMMAND ----------

data.info()

# COMMAND ----------

data.shape

# COMMAND ----------

data.loc[data['Category']=='spam','Category']==0
data.loc[data['Category']=='ham','Category']==1

# COMMAND ----------

X= data['Message']

Y=data['Category']

# COMMAND ----------

print(X)

# COMMAND ----------

print(Y)

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# COMMAND ----------

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

# COMMAND ----------

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)


# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# Instantiate the label encoder
label_encoder = LabelEncoder()

# Fit and transform the training set
Y_train_encoded = label_encoder.fit_transform(Y_train)

# Transform the test set (use the same label encoder instance)
Y_test_encoded = label_encoder.transform(Y_test)


# COMMAND ----------

print(X_train)

# COMMAND ----------

print(X_train_feature)

# COMMAND ----------

model = LogisticRegression()

model.fit(X_train_feature, Y_train)


# COMMAND ----------

prediction_on_training_data = model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# COMMAND ----------

print('Accuracy training data:', accuracy_on_training_data)

# COMMAND ----------

prediction_on_test_data = model.predict(X_test_feature)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

# COMMAND ----------

print('Accuracy training data:', accuracy_on_test_data)

# COMMAND ----------

input_your_mail = ["You have been selected to stay in one of 250 British hotels. For a delightful holiday valued at [Ase678mlji] (Dail )87123453 to claim"]
input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)


# COMMAND ----------

print(prediction)

# COMMAND ----------


