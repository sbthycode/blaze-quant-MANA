import pandas as pd
import nest_asyncio
import re
import twint
import datetime

nest_asyncio.apply()


def url_filter(text: str) -> str:
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.search(regex, text)

    if url and len(url[0]) / len(text) > 0.50:
        return "SPAM"
    else:
        return text


def remove_emojis(text: str) -> str:
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", text)


def filter_scam_tweets(text: str) -> str:
    word_black_list = [
        "giving away",
        "Giving away",
        "GIVING AWAY",
        "PRE-GIVEAWAY",
        "Giveaway",
        "GIVEAWAY",
        "giveaway",
        "follow me",
        "Follow me",
        "FOLLOW ME",
        "retweet",
        "Retweet",
        "RETWEET",
        "LIKE",
        "airdrop",
        "AIRDROP",
        "Airdrop",
        "free",
        "FREE",
        "Free",
        "-follow",
        "-Follow",
        "-rt",
        "-Rt",
        "Requesting faucet funds",
    ]
    if any(ext in text for ext in word_black_list):
        return "SPAM"
    else:
        return text


def clean_text(text: str) -> str:
    text = str(text)

    text = text.replace("\n", "")
    text = url_filter(text)
    text = filter_scam_tweets(text)
    text = remove_emojis(text)

    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = text.replace("#", "")

    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, "", text)

    return text.strip()


def create_date_list(
    start_year: int, start_month: int, start_day: int, number_of_days: int
) -> list[str]:
    start_date = datetime.date(start_year, start_month, start_day)
    date_list = []

    for day in range(number_of_days):
        date_str = (start_date + datetime.timedelta(days=day)).isoformat()
        date_list.append(date_str)

    return date_list


def scrape_tweets(day: str, topic: str, num_of_tweets: int) -> pd.DataFrame:
    config = twint.Config()

    config.Search = topic
    config.Limit = num_of_tweets
    config.Lang = "en"
    config.Since = f"{day} 00:00:00"
    config.Until = f"{day} 23:59:59"
    print("done1")
    config.Pandas = True
    config.Store_object = True
    print("done2")
    twint.run.Search(config)
    print("done3")
    df = twint.storage.panda.Tweets_df
    print("done4")
    return df


df = scrape_tweets(day="2023-12-25", topic="solana", num_of_tweets=10)
print(df)
