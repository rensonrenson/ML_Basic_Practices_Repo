from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import mlflow

# create a spark session
spark = SparkSession.builder.master("local[1]") \
                    .appName('SparkByExamples.com') \
                    .getOrCreate()

# read the source file
df=spark.read.option("header",True).csv('../Source/german_cerdit_data.csv')

# convert spark dataframe to Pandas.
data_df = df.toPandas()

# set y axis value
Y = data_df['Sex']

# set x axis value
X = data_df.drop('Sex',axis = 1)

# change the source null value to none in the dataframe.
data_df['Saving_accounts'] = data_df["Saving_accounts"].replace("NA","none")
data_df["Checking_account"] = data_df["Checking_account"].replace("NA","none")

# check the null value count
data_df.isnull().sum()

# encode the  source
data_df['Sex']=data_df['Sex'].replace({'female':0,'male':1})

data_df['Housing']=data_df['Housing'].replace({'own':3,'rent':2,'free':1})

data_df['Saving_accounts']=data_df['Saving_accounts'].replace({'little':5,'none':4,'moderate':3,'quite rich':2,'rich':1})

data_df["Checking_account"]=data_df["Checking_account"].replace({'none':4,'little':3,'moderate':2,'rich':1})

data_df["Purpose"]=data_df["Purpose"].replace({'car':8,'radio/TV':7,'furniture/equipment':6,'business':5,'education':4,'repairs':3,'domestic appliances':2,'vacation/others':1})

# train and test the source data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# collecting operation define mlflow

with mlflow.start_run():
    mlflow.set_tag("dev", "Renson")
    mlflow.set_tag("algo", "DecisionTree")
    # log the data for each run using log_param, log_metric, log_model
    mlflow.log_param("data-path", "../Source/german_cerdit_data.csv")
    depth = 3
    mlflow.log_param("max_depth", depth)
    dt_classifier = DecisionTreeClassifier(max_depth = depth)
    dt_classifier.fit(X_train, y_train)
    y_test_pred = dt_classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_test_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(dt_classifier, artifact_path="models")
