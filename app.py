import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import altair as alt


# Function to fetch Ethereum price data
def fetch_data(coin="decentraland"):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # Last two years
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range?vs_currency=usd&from={start_date.timestamp()}&to={end_date.timestamp()}"
    response = requests.get(url)
    data = response.json()
    prices = data["prices"]
    return pd.DataFrame(prices, columns=["Date", "Price"]).set_index("Date")


def display_multiple_plots():
    st.title("Cryptocurrency Price Comparison")
    selected_coins = st.multiselect(
        "Select Cryptocurrencies", ["ethereum", "bitcoin", "decentraland"]
    )

    if not selected_coins:
        st.warning("Please select at least one cryptocurrency.")
        return

    charts = []
    for coin in selected_coins:
        data = fetch_data(coin)
        chart = (
            alt.Chart(data.reset_index())
            .mark_line()
            .encode(
                x="Date:T",
                y="Price:Q",
                color=alt.value(coin.capitalize()),
                tooltip=["Date:T", "Price:Q"],
            )
            .properties(title=f"{coin.capitalize()} Price Over Time")
        )

        charts.append(chart)
    combined_chart = alt.layer(*charts).resolve_scale(y="independent")
    st.altair_chart(combined_chart, use_container_width=True)


# Set background to light
st.set_page_config(layout="wide")
st.title("Decentraland (MANA) Price Analysis")

# Section 1: Introduction to Decentraland
with st.container():
    st.header("1. Introduction to Decentraland")
    st.write(
        """
    Decentraland is a virtual reality platform powered by the Ethereum blockchain.
    It allows users to create, experience, and monetize content and applications.
    """
    )

# Section 2: Initial Analysis and Tokenomics (Orange background)
with st.container():
    st.header("2. Initial Analysis and Tokenomics")
    st.markdown(
        """
        <div style="background-color: #FFA500; padding: 10px; border-radius: 5px;">
            <p>Decentraland's native token is MANA, which serves various purposes within the virtual ecosystem.</p>
            <p>Explore the tokenomics and initial analysis to understand its role in the platform.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

# Section 3: Price-related Analysis
with st.container():
    st.header("3. Price-related Analysis")
    df = fetch_data("decentraland")
    df.index = pd.to_datetime(df.index, unit="ms")
    st.line_chart(df)

    # Additional analysis or features can be added here
    st.write(
        """
    Explore additional price-related analysis or features in this section.
    You can analyze trends, identify key events, or present other relevant insights.
    """
    )

# Section 4: Competition Analysis
with st.container():
    st.header("3. Market Competition Analysis")
    df = display_multiple_plots()
    df.index = pd.to_datetime(df.index, unit="ms")
    st.line_chart(df)

    # Additional analysis or features can be added here
    st.write(
        """
    Explore additional market competition analysis or features in this section.
    You can analyze trends, identify key events, or present other relevant insights.
    """
    )
