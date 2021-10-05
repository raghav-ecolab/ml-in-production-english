# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Model Management
# MAGIC 
# MAGIC A MLflow `pyfunc` allows for fully customizable deployments. This lesson provides a generalizable way of handling machine learning models created in and deployed to a variety of environments.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Introduce model management best practices
# MAGIC  - Build a model with preprocessing logic, a loader module, side artifacts, a training method, and a custom environment
# MAGIC  - Apply the custom ML model

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Managing Machine Learning Models
# MAGIC 
# MAGIC Once a model has been trained and bundled with the environment it was trained in the next step is to package the model so that it can be used by a variety of serving tools. **Packaging the final model in a platform-agnostic way offers the most flexibility in deployment options and allows for model reuse across a number of platforms.**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **MLflow models is a convention for packaging machine learning models that offers self-contained code, environments, and models.**<br><br>
# MAGIC 
# MAGIC * The main abstraction in this package is the concept of **flavors**
# MAGIC   - A flavor is a different ways the model can be used
# MAGIC   - For instance, a TensorFlow model can be loaded as a TensorFlow DAG or as a Python function
# MAGIC   - Using an MLflow model convention allows for both of these flavors
# MAGIC * The `python_function` or `pyfunc` flavor of models gives a generic way of bundling models
# MAGIC * We can thereby deploy a python function without worrying about the underlying format of the model
# MAGIC 
# MAGIC **MLflow therefore maps any training framework to any deployment** using these platform-agnostic representations, massively reducing the complexity of inference.
# MAGIC 
# MAGIC Arbitrary logic including pre and post-processing steps, arbitrary code executed when loading the model, side artifacts, and more can be included in the pipeline to customize it as needed.  This means that the full pipeline, not just the model, can be preserved as a single object that works with the rest of the MLflow ecosystem.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models-enviornments.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md Some of the most popular built-in flavors include the following:<br><br>
# MAGIC 
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.keras.html#module-mlflow.keras" target="_blank">mlflow.keras</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#module-mlflow.sklearn" target="_blank">mlflow.sklearn</a>
# MAGIC * <a href="https://mlflow.org/docs/latest/python_api/mlflow.spark.html#module-mlflow.spark" target="_blank">mlflow.spark</a>
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/python_api/index.html" target="_blank">You can see all of the flavors and modules here.</a>
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-models.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md ### Custom Models using `pyfunc`
# MAGIC 
# MAGIC A `pyfunc` is a generic python model that can define any arbitrary logic, regardless of the libraries used to train it. **This object interoperates with any MLflow functionality, especially downstream scoring tools.**  As such, it's defined as a class with a related directory structure with all of the dependencies.  It is then "just an object" with a various methods such as a predict method.  Since it makes very few assumptions, it can be deployed using MLflow, SageMaker, a Spark UDF, or in any other environment.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Check out <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-create-custom" target="_blank">the `pyfunc` documentation for details</a><br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Check out <a href="https://github.com/mlflow/mlflow/blob/master/docs/source/models.rst#example-saving-an-xgboost-model-in-mlflow-format" target="_blank">this README for generic example code and integration with `XGBoost`</a><br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Check out <a href="https://mlflow.org/docs/latest/models.html#example-creating-a-custom-add-n-model" target="_blank">this eaxmple that creates a basic class that adds `n` to the input values</a>

# COMMAND ----------

# MAGIC %md
# MAGIC We will train a random forest model using a pre-processed training set. The input dataframes are processed with the following steps:<br><br>
# MAGIC 
# MAGIC 1. Create an additional feature (`review_scores_sum`) aggregated from various review scores
# MAGIC 2. Enforce the proper data types
# MAGIC 
# MAGIC When creating predictions from our model, we need to re-apply the same pre-processing logic to the data each time we use our model. 
# MAGIC 
# MAGIC To streamline the steps, we define a custom `RFWithPreprocess` model class that uses a `preprocess_input(self, model_input)` helper method to automatically pre-processes the raw input it receives before executing a custom `fit()` method or before passing that input into the trained model's `.predict()` function. This way, in future applications of our model we will no longer have to handle arbitrary pre-processing logic for every batch of data.

# COMMAND ----------

# MAGIC %md Import the data.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)

# Change column types to simulate not having control over upstream changes
X_train["latitude"] = X_train["latitude"].astype(str)
X_train["longitude"] = X_train["longitude"].astype(str)

# COMMAND ----------

# MAGIC %md Examine the scripting version of the model we want to deploy. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> This code is **NOT** recommended

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Copy dataset
X_train_processed = X_train.copy()

# Feature engineer an aggregate feature
X_train_processed["review_scores_sum"] = (
   X_train["review_scores_accuracy"] + 
   X_train["review_scores_cleanliness"]+
   X_train["review_scores_checkin"] + 
   X_train["review_scores_communication"] + 
   X_train["review_scores_location"] + 
   X_train["review_scores_value"]
)
X_train_processed = X_train_processed.drop([
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
  ], axis=1)

# Enforce data types
X_train_processed["latitude_cleaned"] = X_train["latitude"].astype(float)
X_train_processed["longitude_cleaned"] = X_train["longitude"].astype(float)
X_train_processed = X_train_processed.drop(["latitude", "longitude"], axis=1)

## Repeat the same on the test datset
# Copy dataset
X_test_processed = X_test.copy()

# Feature engineer an aggregate feature
X_test_processed["review_scores_sum"] = (
   X_test["review_scores_accuracy"] + 
   X_test["review_scores_cleanliness"]+
   X_test["review_scores_checkin"] + 
   X_test["review_scores_communication"] + 
   X_test["review_scores_location"] + 
   X_test["review_scores_value"]
)
X_test_processed = X_test_processed.drop([
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
  ], axis=1)

# Enforce data types
X_test_processed["latitude_cleaned"] = X_test["latitude"].astype(float)
X_test_processed["longitude_cleaned"] = X_test["longitude"].astype(float)
X_test_processed = X_test_processed.drop(["latitude", "longitude"], axis=1)

# Fit and evaluate a random forest model
rf_model = RandomForestRegressor(n_estimators=15, max_depth=5)

rf_model.fit(X_train_processed, y_train)

# COMMAND ----------

# MAGIC %md Lots of repeitive code, right? And what happens when we need to replicate this in a deployment system? Take a look at the following code instead.

# COMMAND ----------

import mlflow

class RFWithPreprocess(mlflow.pyfunc.PythonModel):

  def __init__(self, params):
    '''
    Initialize with just the model hyperparameters
    '''
    self.params = params
    self.rf_model = None
    self.config = None
        
  def load_context(self, context=None, config_path=None):
    '''
    When loading a pyfunc, this method runs automatically with the related
    context.  This method is designed to perform the same functionality when
    run in a notebook or a downstream operation (like a REST endpoint).
    If the `context` object is provided, it will load the path to a config from 
    that object (this happens with `mlflow.pyfunc.load_model()` is called).
    If the `config_path` argument is provided instead, it uses this argument
    in order to load in the config.
    '''
    if context: # This block executes for server run
      config_path = context.artifacts["config_path"]
    else: # This block executes for notebook run
      pass
      
    self.config = json.load(open(config_path))
      
  def preprocess_input(self, model_input):
    '''
    Return pre-processed model_input
    '''
    processed_input = model_input.copy()
    processed_input["review_scores_sum"] = (
      processed_input["review_scores_accuracy"] + 
      processed_input["review_scores_cleanliness"]+
      processed_input["review_scores_checkin"] + 
      processed_input["review_scores_communication"] + 
      processed_input["review_scores_location"] + 
      processed_input["review_scores_value"]
    )
    processed_input = processed_input.drop([
      "review_scores_accuracy",
      "review_scores_cleanliness",
      "review_scores_checkin",
      "review_scores_communication",
      "review_scores_location",
      "review_scores_value"
    ], axis=1)
      
    processed_input["latitude_cleaned"] = processed_input["latitude"].astype(float)
    processed_input["longitude_cleaned"] = processed_input["longitude"].astype(float)
    processed_input = processed_input.drop(["latitude", "longitude"], axis=1)
    return processed_input
  
  def fit(self, X_train, y_train):
    '''
    Uses the same preprocessing logic to fit the model
    '''
    from sklearn.ensemble import RandomForestRegressor
    
    processed_model_input = self.preprocess_input(X_train)
    rf_model = RandomForestRegressor(**self.params)
    rf_model.fit(processed_model_input, y_train)
    
    self.rf_model = rf_model
    
  def predict(self, context, model_input):
    '''
    This is the main entrance to the model in deployment systems
    '''
    processed_model_input = self.preprocess_input(model_input.copy())
    return self.rf_model.predict(processed_model_input)

# COMMAND ----------

# MAGIC %md The `context` parameter is provided automatically by MLflow in downstream tools. This can be used to add custom dependent objects such as models that are not easily serialized (e.g. `keras` models) or custom configuration files.
# MAGIC 
# MAGIC Use the following to provide a config file. Note the steps:<br><br>
# MAGIC 
# MAGIC - Save out any file we want to load into the class
# MAGIC - Create an artifact dictionary of key/value pairs where the value is the path to that object
# MAGIC - When saving the model, all artifacts will be copied over into the same directory for downstream use
# MAGIC 
# MAGIC In our case, we'll save some model hyperparameters as our config.

# COMMAND ----------

import json 

params = {
  "n_estimators": 15, 
  "max_depth": 5
}

# Designate a path
config_path = f"{working_dir}/data.json".replace("dbfs:/", "/dbfs/")

# Save the results
with open(config_path, "w") as f:
  json.dump(params, f)

# Generate an artifact object to saved
# All paths to the associated values will be copied over when saving
artifacts = {"config_path": config_path} 

# COMMAND ----------

# MAGIC %md Instantiate the class. Run `load_context` to load the config. This automatically runs in downstream serving tools.

# COMMAND ----------

model = RFWithPreprocess(params)

# Run manually (this happens automatically in serving integrations)
model.load_context(config_path=config_path) 

# Confirm the config has loaded
model.config

# COMMAND ----------

# MAGIC %md Train the model. Note that this runs the preprocessing logic for us automatically.

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md Generate predictions.

# COMMAND ----------

predictions = model.predict(context=None, model_input=X_test)

predictions

# COMMAND ----------

# MAGIC %md Generate the model signature.

# COMMAND ----------

from mlflow.models.signature import infer_signature

signature = infer_signature(X_test, predictions)

signature

# COMMAND ----------

# MAGIC %md Generate the conda environment. This can be arbitrarily complex. This is necessary because when we use `mlflow.sklearn`, we automatically log the appropriate version of `sklearn`. With a `pyfunc`, we must manually construct our deployment environment.

# COMMAND ----------

from sys import version_info
import sklearn

conda_env = {
  "channels": ["defaults"],
  "dependencies": [
    f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
    "pip",
    {"pip": [
      "mlflow",
      f"sklearn=={sklearn.__version__}"
      ],
     },
  ],
  "name": "sklearn_env"
}

conda_env

# COMMAND ----------

# MAGIC %md Save the model.

# COMMAND ----------

import shutil

mlflow_pyfunc_model_path = f"{working_dir}/RFWithPreprocess".replace("dbfs:", "/dbfs")

try:
  shutil.rmtree(mlflow_pyfunc_model_path) # remove folder if already exists
except:
  None

mlflow.pyfunc.save_model(
  path=mlflow_pyfunc_model_path, 
  python_model=model, 
  artifacts=artifacts,
  conda_env=conda_env, # Note: this could fail if running in a repo. Comment out this line for workaround.
  signature=signature,
  input_example=X_test[:3] 
)

# COMMAND ----------

# MAGIC %md Load the model in `python_function` format.

# COMMAND ----------

loaded_preprocess_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# COMMAND ----------

# MAGIC %md Apply the model.

# COMMAND ----------

loaded_preprocess_model.predict(X_test)

# COMMAND ----------

# MAGIC %md **Note that `pyfunc`'s interoperate with any downstream serving tool. It allows you to use arbitrary code, niche libraries, and complex side information.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC **Question:** How do MLflow projects differ from models?
# MAGIC **Answer:** The focus of MLflow projects is reproducibility of runs and packaging of code.  MLflow models focuses on various deployment environments.
# MAGIC 
# MAGIC **Question:** What is a ML model flavor?
# MAGIC **Answer:** Flavors are a convention that deployment tools can use to understand the model, which makes it possible to write tools that work with models from any ML library without having to integrate each tool with each library.  Instead of having to map each training environment to a deployment environment, ML model flavors manages this mapping for you.
# MAGIC 
# MAGIC **Question:** How do I add pre and post processing logic to my models?
# MAGIC **Answer:** A model class that extends `mlflow.pyfunc.PythonModel` allows you to have load, pre-processing, and post-processing logic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the labs for this lesson, [Model Management Lab]($./Labs/01-Model-Management-Lab)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Models?
# MAGIC **A:** Check out <a href="https://www.mlflow.org/docs/latest/models.html" target="_blank">the MLflow documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>