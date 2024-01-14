from datetime import datetime
import pandas as pd

date_2023 = datetime(2023, 12, 27)
date_2015 = datetime(2015, 8, 7)
days_difference = (date_2023 - date_2015).days
print(f"The number of days between the two dates is: {days_difference} days")

csv_file = "combined_data_sorted.csv"

df = pd.read_csv(csv_file, parse_dates=["Date"])
start_date = df["Date"].min()
end_date = df["Date"].max()
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

missing_dates = set(date_range) - set(df["Date"])

print("Missing Dates:")
for date in sorted(missing_dates):
    print(date.strftime("%Y-%m-%d"))
