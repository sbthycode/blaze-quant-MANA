import pandas as pd

filename = "eth_raw.csv"
df = pd.read_csv(filename)
df["Change"] = df["Close"].pct_change() * 100
df.rename(columns={"Close": "Price"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df.sort_values(by="Date", inplace=True)
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
df["Vol"] = df["Volume"]
df = df[["Date", "Price", "Open", "High", "Low", "Vol", "Change"]]
df.to_csv("eth_modified.csv", index=False)
