import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# Function to fetch Ethereum price data
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # Last two years
    url = f"https://api.coingecko.com/api/v3/coins/decentraland/market_chart/range?vs_currency=usd&from={start_date.timestamp()}&to={end_date.timestamp()}"
    response = requests.get(url)
    data = response.json()
    prices = data["prices"]
    return pd.DataFrame(prices, columns=["Date", "Price"]).set_index("Date")

# Set background to light
st.set_page_config(layout="wide")
st.title("Decentraland (MANA) Price Analysis")

# Section 1: Introduction to Decentraland
with st.container():
    st.header("1. Introduction to Decentraland")
    st.write("""
    Decentraland is a virtual reality platform powered by the Ethereum blockchain.
    It allows users to create, experience, and monetize content and applications.
    """)

# Section 2: Initial Analysis and Tokenomics (Orange background)
with st.container():
    st.header("2. Initial Analysis and Tokenomics")
    st.markdown("""
        <div style="background-color: #FFA500; padding: 10px; border-radius: 5px;">
            <p>Decentraland's native token is MANA, which serves various purposes within the virtual ecosystem.</p>
            <p>Explore the tokenomics and initial analysis to understand its role in the platform.</p>
        </div>
    """, unsafe_allow_html=True)

# Section 3: Price-related Analysis
with st.container():
    st.header("3. Price-related Analysis")
    df = fetch_data()
    df.index = pd.to_datetime(df.index, unit='ms')
    st.line_chart(df)

    # Additional analysis or features can be added here
    st.write("""
    Explore additional price-related analysis or features in this section.
    You can analyze trends, identify key events, or present other relevant insights.
    """)
