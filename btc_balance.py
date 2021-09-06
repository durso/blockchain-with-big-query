# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 09:35:20 2021

@author: Rodrigo
"""

import pandas as pd
import os
#You should change it using the json file you downloaded from google
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="PATH TO JSON FILE"


from google.cloud import bigquery

client = bigquery.Client()

#This is a simple query to check the btc balance
query = """
WITH ledger AS (
   SELECT
     array_to_string(inputs.addresses, ",") as wallet_address,
     -inputs.value as satoshis
   FROM `bigquery-public-data.crypto_bitcoin.inputs` as inputs
   UNION ALL
   SELECT
     array_to_string(outputs.addresses, ",") as wallet_address,
	 outputs.value as satoshis
   FROM `bigquery-public-data.crypto_bitcoin.outputs` as outputs
)
--Return the top 10,000 wallets in terms of balance in BTC 
SELECT
   wallet_address, 
   sum(satoshis)/100000000 as btc_balance
FROM ledger
GROUP BY wallet_address
ORDER BY btc_balance DESC
LIMIT 10000
"""


sbytes = 2**30

#173 is the query size in GB (checked in Google Big Query console). You should change it for other queries or you can automate it. sbytes is a constant
safe_config = bigquery.QueryJobConfig(
    maximum_bytes_billed=173 * sbytes
)

query_job = client.query(query, job_config=safe_config)
result = query_job.result()
df = result.to_dataframe()

#Save for future use as you are billed for each new query
df.to_csv("btc-balance060921.csv")
