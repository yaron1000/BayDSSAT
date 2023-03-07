"""
Connecting to Bayer's Crop Sciense Warehouse (CSW).

Main project:
    location360-datasets.
    product360-datasets.

Main datasets:
    environmental_data_cube.
        schemas: isric_global_soil_250, growth_stage_predictions_corn, growth_stage_predictions_soybean.
    historical_weather:
        routines query: historical_weather_daily_blend, historical_weather_hourly_blend.
"""

import hvac
import base64
import json
from google.oauth2 import service_account
from google.cloud import bigquery
import os
import pandas as pd

APPROLE_ID = os.environ['APPROLE_ID']
APPROLE_SECRET = os.environ['APPROLE_SECRET']

client = hvac.Client(url='https://vault.agro.services')
#client.auth_approle(APPROLE_ID, APPROLE_SECRET)
client.auth.approle.login(APPROLE_ID, APPROLE_SECRET)
secrets = client.read('secret/csw/service-identities/marketdevtrialinglatam')

if 'data' in secrets and type(secrets['data']['data']) == str:
    service_account_creds = json.loads(
        base64.b64decode(secrets['data']['data']))
else:
    service_account_creds = secrets

credentials = service_account.Credentials.from_service_account_info(
    service_account_creds)


class CSWconnect:
    """Class to connect to the Crop Science Warehouse (CSW).

    Attributes
    ----------
    project  
        name of the project which contain the datasets (e.g., 'location360-datasets').
    

    Methods
    -------
    load()
        get pandas dataframe from query used.
    save()
        save pandas dataframe as a table in CSW.
    """

    def __init__(self, project: str) -> None:
        self.credentials = credentials
        self.project = project
        self.bq_client = bigquery.Client(
            project=project, credentials=credentials)

    def load(self, query: str) -> pd.DataFrame:
        """Method to load CSW tables as pandas dataframe.
        
        Attributes
        ----------
        query  
            query to retrieve data from dataset project.
        
        Returns
        ------
        df 
            Dataframe obtained from CSW query.

        """

        df = pd.read_gbq(query,
                         project_id=self.project,
                         credentials=credentials,
                         use_bqstorage_api=True)
        return df

    def save(self, DF: pd.DataFrame, BQ_table: str, append: bool = False) -> None:
        """Method to save a pandas dataframe in CSW.

        Attributes
        ----------
        DF  
            pandas dataframe to upload in CSW.
        
        BQ_table  
            table name, e.g., 'latam_datasets.dssat_brazil_soybeans'.
        
        Append
            choose if append (True) or replace (False) the CSW table.
        
        Returns
        ------
        None 

        """
        BQ_table_up = self.project + '.' + BQ_table
        
        if append:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND")
        else:
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE")

        self.bq_client.load_table_from_dataframe(
            DF, BQ_table_up, job_config=job_config)



