# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:19:24 2021

@author: Rodrigo

Get the cumulative BTC balance of a wallet
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
        SUM(satoshis) OVER (PARTITION BY wallet_address order by block_timestamp) as satoshis
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
--Return the historical cumulative sum for binance cold wallet
SELECT block_timestamp, satoshis/100000000 as btc_balance 
FROM  ledger
WHERE wallet_address="34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo"
ORDER BY block_timestamp
"""

sbytes = 2**30

#180 is the query size in GB (checked in Google Big Query console).
safe_config = bigquery.QueryJobConfig(
    maximum_bytes_billed=180 * sbytes
)
#This may take a while
query_job = client.query(query, job_config=safe_config)
result = query_job.result()
df = result.to_dataframe()

#Save for future use as your are billed for each new query
df.to_csv("btc-balance-age-220921.csv")
#Change type
df["btc_balance"] = pd.to_numeric(df["btc_balance"])

#Note that first they sent a small amount to the wallet, just to test
df.head()

#Plot line graph of btc balance over time
df.plot(x="block_timestamp",y="btc_balance",figsize=(12,10))

