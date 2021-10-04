# Blockchain with big query

We are using Google Big Query to do some on-chain analysis. We will start with Bitcoin. You could also run a full Bitcoin node, but in my case I only have 500GB disk space, so I would not be able to run a node and then export it to SQL local server.

We are staring with the file balance_btc.py that returns the balance of the top wallets.

The second file (btc_balance_and_wallet_age.py) shows the balance along with the number of days since the first transaction (wallet age).

Plot produced by btc_balance_and_wallet_age.py

![Histogram](https://github.com/durso/blockchain-with-big-query/blob/main/img/Hist_age.png?raw=true)

The third file (wallet_cumbalance.py) shows the cumulative balance of a specific wallet (Binance cold wallet). You can amend the WHERE clause to get the balance of a different wallet.

Plot produced by wallet_cumbalance.py

![Line Chart](https://github.com/durso/blockchain-with-big-query/blob/main/img/btc_cumbalance.png?raw=true)
