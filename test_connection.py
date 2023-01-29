# Databricks notebook source
!pip install adlfs

# COMMAND ----------

dsArtifacts_storage_options_DEV = {
'account_name': 'mapdatalakestore002d',
'tenant_id': 'c1eb5112-7946-4c9d-bc57-40040cfe3a91',
'client_id': '1712d803-ec6d-4cc2-b426-2b3d571a263a',
'client_secret': 'X2c7Q~.YnQxaL6vUNotS~2WjNt0KGOAGuJ3iz'
}
dsArtifacts_storage_options_STAGE = { 'account_name': 'mapdatalakestore002p',
'tenant_id' :"c1eb5112-7946-4c9d-bc57-40040cfe3a91",
'client_secret': "Quu7llbEIh8_ba0516OfpTa_at~1KmL92_",
'client_id' : "3001b296-5a1d-41de-9729-945dbd580dc0" }

dsArtifacts_storage_options_PROD = { 'account_name': 'mapdatalakestore002s',
                                'tenant_id' :"c1eb5112-7946-4c9d-bc57-40040cfe3a91",
                                'client_secret': "RSs7Q~RMYPU6QWtcI~xP3gytxREHXv3ep0ubs",
                                'client_id' : "9ed55bcd-0cf9-466b-8ab8-bf6209717682" }   


# COMMAND ----------

import pandas as pd
#https://mapdatalakestore002p.blob.core.windows.net/ds-artifacts/wsi/ex-us/model_artifacts/model_metaData_ANZ.csv
df_dev=pd.read_csv('abfs://ds-artifacts/wsi/ex-us/model_artifacts/model_metaData_ANZ.csv',storage_options=dsArtifacts_storage_options_DEV)

# COMMAND ----------

df_dev

# COMMAND ----------

df_stage=pd.read_csv('abfs://ds-artifacts/wsi/ex-us/model_artifacts/model_metaData_ANZ.csv',storage_options=dsArtifacts_storage_options_STAGE)

# COMMAND ----------

df_stage

# COMMAND ----------

df_prod=pd.read_csv('abfs://ds-artifacts/wsi/ex-us/model_artifacts/',storage_options=dsArtifacts_storage_options_PROD)

# COMMAND ----------

df_prod

# COMMAND ----------


