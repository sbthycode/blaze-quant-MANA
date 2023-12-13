# %% [markdown]
# # Beginner's Exploratory Data Analysis for Crypto Market

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import cross_decomposition
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

sns.set()

# Make charts a bit bolder
#sns.set_context("talk")

%matplotlib inline

# Default figure size
sns.set(rc={"figure.figsize": (12, 6)})

# This actually makes autocomplete WAY faster ...
%config Completer.use_jedi = False

# Show only 2 decimals for floating point numbers
pd.options.display.float_format = "{:.2f}".format

sns.set_style('whitegrid')

# %%
data = pd.read_csv('../input/seven-comparison-data/seven_comparison_data.csv')

# %%
data.head()

# %%
data.describe()

# %%
data.shape, data.info()

# %% [markdown]
# Here are the descriptions for some of the columns that I wasn't really sure about:
# 
# * volume - Transactions volume
# * market - Market Cap
# * ranknow - Currency rank
# * spread - Spread between high and low
# 
# Also, one thing I noticed -- market caps are quite huge. For ease of observing, let's introduce a new column - *market_billion*, which will represent currencies Market Cap in billion

# %% [markdown]
# ## Data Wrangle & Cleanup

# %%
# Convert date to real date
data['date'] = pd.to_datetime(data['date'])
data['market_billion'] = data['market'] / 1000000000
data['volume_million'] = data['volume'] / 1000000000
data['volume_billion'] = data['volume']

# %%
# Let's prepare one dataframe where we will observe closing prices for each currency
wide_format = data.groupby(['date', 'name'])['close'].last().unstack()
wide_format.head(3)

# %%
wide_format.shape

# %%
wide_format.describe()

# %% [markdown]
# ## Data Exploration
# 
# ### Top 10 cryptocurrencies in 2018

# %%
ax = data.groupby(['name'])['market_billion'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh');
ax.set_xlabel("Market cap (in billion USD)");
plt.title("Top 10 Currencies by Market Cap");

# %%
ax = data.groupby(['name'])['volume_million'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh');
ax.set_xlabel("Transaction Volume (in million)");
plt.title("Top 10 Currencies by Transaction Volume");

# %%
# For sake of convenience, let's define the top 5 currencies

top_5_currency_names = data.groupby(['name'])['market'].last().sort_values(ascending=False).head(7).index
data_top_5_currencies = data[data['name'].isin(top_5_currency_names)]
data_top_5_currencies.head(5)
print(top_5_currency_names)

# %%
data_top_5_currencies.describe()

# %% [markdown]
# ## Trend Charts

# %%
ax = data_top_5_currencies.groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency");

# %%
ax = data_top_5_currencies.groupby(['date', 'name'])['market_billion'].mean().unstack().plot();
ax.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency");

# %%
ax = data_top_5_currencies.groupby(['date', 'name'])['volume_million'].mean().unstack().plot();
ax.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency");

# %% [markdown]
# ## Trend Charts in 2017

# %%
ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency (from 2017th)");

# %%
ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2017].groupby(['date', 'name'])['market_billion'].mean().unstack().plot();
ax.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency (from 2017th)");

# %%
ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2017].groupby(['date', 'name'])['volume_million'].mean().unstack().plot();
ax.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency (from 2017th)");

# %% [markdown]
# ## Trend Charts in 2018

# %%
ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2023].groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency (from 2018th)");

# %%
ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2023].groupby(['date', 'name'])['market_billion'].mean().unstack().plot();
ax.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency (from 2018th)");

# %%
ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2023].groupby(['date', 'name'])['volume_million'].mean().unstack().plot();
ax.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency (from 2018th)");

# %% [markdown]
# ## Correlation

# %%
plt.figure(figsize=(14,8))
sns.heatmap(wide_format[top_5_currency_names].corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True);

# %% [markdown]
# ## Experiments
# 
# Small experiment - let's assume that we invested some amount (say - 1000 USD) at some point. Let's see what ROI would we have.

# %%
def plot_roi(amount, df):
    ((amount / df.iloc[0]) * df).plot(figsize=(12,8))

# %%
plot_roi(1000, wide_format[['Decentraland']])

# %%
wide_format_2017th = wide_format[(wide_format.index.year >= 2017)]
plot_roi(1000, wide_format_2017th[top_5_currency_names])

# %%
wide_format_late_2017th = wide_format[(wide_format.index.year >= 2017) & (wide_format.index.month >= 10)]
plot_roi(1000, wide_format_late_2017th[top_5_currency_names])

# %%
wide_format_2018th = wide_format[(wide_format.index.year >= 2018)]
plot_roi(1000, wide_format_2018th[top_5_currency_names])

# %%
len(data.slug.unique())

# %%
# Some common filters that we might be using
# is_bitcoin = data['symbol'] == 'BTC'
# is_ethereum = data['symbol'] == 'ETH'
# is_ripple  = data['symbol'] == 'XRP'

is_axs = data['symbol'] == 'AXS'
is_mana = data['symbol'] == 'MANA'
is_sand = data['symbol'] == 'SAND'

# Pull out a part of dataset that only has the most interesting currencies
# data_top_currencies = data[is_bitcoin | is_ethereum | is_ripple]
data_top_currencies = data[is_axs | is_mana | is_sand]

# %% [markdown]
# Let's chart out Top cryptocurrencies according to latest reported Market Cap

# %%
top10Currencies = data.groupby('name')['market_billion'].last().sort_values(ascending=False).head(10)

# %%
ax = top10Currencies.sort_values().plot(kind='barh')
ax.set_xlabel("Market cap in Billion");
ax.set_ylabel("Currency");

# %% [markdown]
# As we can see, and as it was expected, Bitcoin has the highest market cap. Let's see the trend for couple of top currencies.

# %%
ax = data_top_currencies.groupby(['date', 'name'])['close'].mean().unstack().plot()
ax.set_ylabel("Price per 1 unit (in USD)")

# %% [markdown]
# That's rather amusing. Let's see focus on trend starting in 2018th.

# %%
data_top_currencies[data_top_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['close'].mean().unstack().plot()
ax.set_ylabel("Price per 1 unit (in USD)")

# %% [markdown]
# We can see that prices have jumped enormously in start and then decreases monotonically with a sharp increase in between Feb and March of 2018th. The cause? Apparently, there are lots of causes. From people's awareness about crypto currencies, to introduction of other currencies that increased the overal need.
# 
# ## Let's see a trend of Trading Volume for top currencies now

# %%
ax = data_top_currencies[data_top_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['volume_billion'].mean().unstack().plot()
ax.set_ylabel("Trading volume (in billion)");

# %% [markdown]
# There seems to be a correlation in trading between currencies. Which probably makes sense as, if I understood correctly, most of the currencies are actually traded using Bitcoin (i.e. you have to purchase Bitcoin in order to purchase Ripple). For sake of visibility, I'll plot Bitcoin and other currencies separately. Thing is that Bitcoin prices are actually masking other currencies.

# %% [markdown]
# # Experiments
# 
# 
# Let's do a small experiment. Let's say that we invested 1000$ in each crypto currency 5 years ago. Let's see how much money would you have now.
# 
# First, let's start by drawing a diagram of closing prices for each year for each currency.

# %%
def plot_with_textvalue(df):
    ax = df.plot(kind='bar')
    
    ax.set_ylabel("Yearly closing prices (in USD)")

    for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')

# %%
top10Currencies

# %%
# closing_prices_bitcoin_and_ethereum = data[is_bitcoin | is_ethereum].groupby(['date','name'])['close'].last().unstack().reset_index()
# closing_prices_other_currencies = data[data['name'].isin(top10Currencies.index) & ~is_bitcoin & ~is_ethereum].groupby(['date','name'])['close'].last().unstack().reset_index()

closing_prices_bitcoin_and_ethereum = data[is_mana | is_sand].groupby(['date','name'])['close'].last().unstack().reset_index()
closing_prices_other_currencies = data[data['name'].isin(top10Currencies.index) & ~is_sand & ~is_mana].groupby(['date','name'])['close'].last().unstack().reset_index()

# %%
yearly_closing_prices_bitcoin_and_ethereum = closing_prices_bitcoin_and_ethereum.groupby(closing_prices_bitcoin_and_ethereum.date.dt.year).last()
yearly_closing_prices_bitcoin_and_ethereum.drop(columns='date', inplace=True)
plot_with_textvalue(yearly_closing_prices_bitcoin_and_ethereum)

# %%
yearly_closing_prices_other_currencies = closing_prices_other_currencies.groupby(closing_prices_other_currencies.date.dt.year).last()
yearly_closing_prices_other_currencies.drop(columns='date', inplace=True)
yearly_closing_prices_other_currencies.plot(kind='bar')

# %%
closing_prices_other_currencies.head()

# %% [markdown]
# Let's plot the closing prices.

# %%
closing_prices_bitcoin_and_ethereum.head()

# %%
closing_prices_other_currencies.head()

# %%
def calc_earnings(currency_name, df):
    #print("Displaying stats for "+currency_name)

    closing_prices = df[(df['name'] == currency_name) & (~df['close'].isnull())][['date', 'close']]

    # Num. currency purchased for 1000$
    #print("Closing price at the beginning: " + str(closing_prices.iloc[0]['close']))

    num_units_purchased = 1000 / closing_prices.iloc[0]['close']
    num_units_purchased

    #print("Num. units purchased: " + str(num_units_purchased))

    # Current value
    last_price = closing_prices.iloc[-1]['close']
    #print("Last price: " + str(last_price))

    amount_earned = (num_units_purchased * last_price) - 1000

    #print("Amount you would have earned: " + str(amount_earned) + "$")
    
    return amount_earned
    
# Borrow the index :-)
top_10_currencies_earnings = top10Currencies

for currency in top10Currencies.index:
    top_10_currencies_earnings[currency] = calc_earnings(currency, data)
    
ax = top_10_currencies_earnings.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')

# %%
# Borrow the index :-)
top_10_currencies_earnings_2018 = top10Currencies

for currency in top10Currencies.index:
    top_10_currencies_earnings_2018[currency] = calc_earnings(currency, data[data.date.dt.year >= 2018])
    
top_10_currencies_earnings_2018

ax = top_10_currencies_earnings_2018.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')

# %%
# Borrow the index :-)
top_10_currencies_earnings_2018 = top10Currencies

for currency in top10Currencies.index:
    top_10_currencies_earnings_2018[currency] = calc_earnings(currency, data[data.date.dt.year >= 2018])
    
top_10_currencies_earnings_2018

ax = top_10_currencies_earnings_2018.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')

# %%
top_10_currencies_earnings_without_nem = top_10_currencies_earnings[top_10_currencies_earnings.index != 'NEM']

ax = top_10_currencies_earnings_without_nem.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')

# %%
top10Currencies = data.groupby('name')['market_billion'].last().sort_values(ascending=False).head(5)
closing_prices_top10 = data[data['name'].isin(top10Currencies.index)].groupby(['date', 'name'])['close'].mean().unstack()
closing_prices_top10.corr()

plt.figure(figsize=(12,6))
sns.heatmap(closing_prices_top10.corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap between Bitcoin and other top 5 Crypto')

# %%
plt.figure(figsize=(12,6))
sns.heatmap(closing_prices_top10.corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap between Bitcoin and other top 4 Crypto')
