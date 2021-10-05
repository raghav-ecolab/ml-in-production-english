# Databricks notebook source
# Does any work to reset the environment prior to testing.
try:
  dbutils.fs.unmount("/mnt/training")
except:
  pass

username = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(dbutils.entry_point.getDbutils().notebook().getContext().tags())["user"]
dbutils.fs.rm(f"dbfs:/user/{username}/ml_in_production", True)

# COMMAND ----------

# MAGIC %run ./Classroom-Setup
