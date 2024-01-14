import pandas as pd

files = [
    "eth_raw_data/Ethereum_8_1_2015-12_31_2015_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2016-12_31_2016_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2017-12_31_2017_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2018-12_31_2018_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2019-12_31_2019_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2020-12_31_2020_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2021-12_31_2021_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2022-12_31_2022_historical_data_coinmarketcap.csv",
    "eth_raw_data/Ethereum_1_1_2023-12_27_2023_historical_data_coinmarketcap.csv",
]

dfs = []
for filename in files:
    df = pd.read_csv(filename, sep=";")
    df["timeOpen"] = pd.to_datetime(df["timeOpen"])
    df["timeClose"] = pd.to_datetime(df["timeClose"])
    df["timeOpen"] = df["timeOpen"].dt.date
    df["timeClose"] = df["timeClose"].dt.date

    df.rename(
        columns={
            "timeOpen": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.sort_values(by="Date", inplace=True)
combined_df.to_csv("combined_data_sorted.csv", index=False)
