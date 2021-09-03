# Databricks notebook source
# Does any work to reset the environment prior to testing.
try:
  dbutils.fs.unmount("/mnt/training")
except:
  pass

# COMMAND ----------

# MAGIC %run ./Classroom-Setup
