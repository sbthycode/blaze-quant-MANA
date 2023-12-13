import pandas as pd

# Create an empty list to store DataFrames
all_dfs = []

tokens = {
    "apecoin-ape": {
        "include": True,
        "symbol": "APE",
        "name": "ApeCoin",
        "ranknow": 87,
        "file_path": "raw_data\ApeCoin_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
    "internet-computer": {
        "include": True,
        "symbol": "ICP",
        "name": "Internet Computer",
        "ranknow": 28,
        "file_path": "raw_data\Internet Computer_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
    "stacks": {
        "include": True,
        "symbol": "STX",
        "name": "Stacks",
        "ranknow": 50,
        "file_path": "raw_data\Stacks_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
    "decentraland": {
        "include": True,
        "symbol": "MANA",
        "name": "Decentraland",
        "ranknow": 67,
        "file_path": "raw_data\Decentraland_12_12_2019-12_13_2023_historical_data_coinmarketcap.csv",
    },
    "theta-network": {
        "include": True,
        "symbol": "THETA",
        "name": "Theta Network",
        "ranknow": 61,
        "file_path": "raw_data\Theta Network_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
    "axie-infinity": {
        "include": True,
        "symbol": "AXS",
        "name": "Axie Infinity",
        "ranknow": 66,
        "file_path": "raw_data\Axie Infinity_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
    "the-sandbox": {
        "include": True,
        "symbol": "SAND",
        "name": "The Sandbox",
        "ranknow": 62,
        "file_path": "raw_data\The Sandbox_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
    "bitcoin": {
        "include": False,
        "symbol": "BTC",
        "name": "Bitcoin",
        "ranknow": 1,
        "file_path": "raw_data\Bitcoin_12_12_2019-12_13_2023_historical_data_coinmarketcap.csv",
    },
    "ethereum": {
        "include": False,
        "symbol": "ETH",
        "name": "Ethereum",
        "ranknow": 2,
        "file_path": "raw_data\Ethereum_12_12_2019-12_13_2023_historical_data_coinmarketcap.csv",
    },
    "illuvium": {
        "include": True,
        "symbol": "ILV",
        "name": "Illuvium",
        "ranknow": 124,
        "file_path": "raw_data\Illuvium_12_13_2019-12_14_2023_historical_data_coinmarketcap.csv",
    },
}

for k, v in tokens.items():
    if v["include"] == False:
        continue
    # Read the CSV file into a pandas DataFrame
    csv_file = v["file_path"]
    df = pd.read_csv(csv_file, sep=";")

    # Convert the 'timeClose' column to a pandas datetime object
    df["timeClose"] = pd.to_datetime(df["timeClose"])

    # Create a new column 'date' with the date extracted from 'timeClose'
    df["date"] = df["timeClose"].dt.strftime("%m/%d/%Y")

    # Add 'close_ratio' and 'spread' columns
    df["close_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["spread"] = df["high"] - df["low"]

    # Add constant columns 'slug', 'symbol', and 'name'
    df["slug"] = k
    df["symbol"] = v["symbol"]
    df["name"] = v["name"]
    df["ranknow"] = v["ranknow"]

    # Drop the columns 'timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'timestamp'
    df = df.drop(["timeOpen", "timeClose", "timeHigh", "timeLow", "timestamp"], axis=1)

    df = df.rename(columns={"marketCap": "market"})

    # Reorder the columns
    column_order = [
        "slug",
        "symbol",
        "name",
        "date",
        "ranknow",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market",
        "close_ratio",
        "spread",
    ]
    df = df[column_order]

    # Append the DataFrame to the list
    all_dfs.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv("modified\eight_comparison_data.csv", index=False)
