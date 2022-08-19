# Databricks notebook source
# MAGIC %pip install \
# MAGIC git+https://github.com/databricks-academy/dbacademy-gems@c3032c2df47472f1600d368523f052d2920b406d \
# MAGIC git+https://github.com/databricks-academy/dbacademy-rest@04a7a66df15a54460f01ee98003d244819281ab1 \
# MAGIC git+https://github.com/databricks-academy/dbacademy-helper@0206e867312f6efc3af117f3742878c65f406ee8 \
# MAGIC --quiet --disable-pip-version-check

# COMMAND ----------

# MAGIC %run ./_dataset_index

# COMMAND ----------

from dbacademy_gems import dbgems
from dbacademy_helper import DBAcademyHelper, Paths

helper_arguments = {
    "course_code" : "mlip",          # The abreviated version of the course
    "course_name" : "ml-in-production",      # The full name of the course, hyphenated
    "data_source_name" : "ml-in-production", # Should be the same as the course
    "data_source_version" : "v01",     # New courses would start with 01
    "enable_streaming_support": False, # This couse uses stream and thus needs checkpoint directories
    "install_min_time" : "1 min",      # The minimum amount of time to install the datasets (e.g. from Oregon)
    "install_max_time" : "5 min",      # The maximum amount of time to install the datasets (e.g. from India)
    "remote_files": remote_files,      # The enumerated list of files in the datasets
}

