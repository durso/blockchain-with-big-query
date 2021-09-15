# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:46:59 2021

@author: Rodrigo
"""

import pyarrow
import numpy as np
import pandas as pd


from google.cloud import bigquery
client = bigquery.Client()



query = """
WITH ledger AS (
    SELECT 
        wallet_address, 
        block_timestamp, 
        -- aggregate by wallet address
        SUM(satoshis) OVER (PARTITION BY wallet_address) AS satoshis,
        -- rank by date
        row_number() OVER(PARTITION BY wallet_address Order BY block_timestamp) AS rownumber
    FROM 
        (
        -- debits along with time stamp    
        SELECT
                array_to_string(inputs.addresses, ",") as wallet_address,
                -inputs.value as satoshis,
                block_timestamp 
        FROM `bigquery-public-data.crypto_bitcoin.inputs` as inputs
        UNION ALL
         -- credits along with time stamp    
        SELECT
                array_to_string(outputs.addresses, ",") as wallet_address,
                outputs.value as satoshis,
                block_timestamp
            FROM `bigquery-public-data.crypto_bitcoin.outputs` as outputs
        )
)
--Return the top 100,000 wallets in terms of balance in BTC along with the number of days since the first transaction
SELECT wallet_address, TIMESTAMP_DIFF(CURRENT_TIMESTAMP(),block_timestamp,DAY) AS age, satoshis/100000000 as btc_balance 
FROM  ledger
WHERE rownumber=1
ORDER BY btc_balance DESC
LIMIT 100000
"""

sbytes = 2**30

#179 is the query size in GB (checked in Google Big Query console). You should change it for other queries or you can automate it. sbytes is a constant
safe_config = bigquery.QueryJobConfig(
    maximum_bytes_billed=179 * sbytes
)
#This will take a while
query_job = client.query(query, job_config=safe_config)
result = query_job.result()
df = result.to_dataframe()

#Save for future use as your are billed for each new query
df.to_csv("btc-balance-age-150921.csv")

#Plot histogram of wallet age
df["age"].plot.hist(bins=100, title= "Histogram of wallet age in days");

#get oldest wallet
df.iloc[np.argmax(df["age"])]
#No surprise 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. Check out the owner yourself =]