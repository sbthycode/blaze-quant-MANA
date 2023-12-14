import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import altair as alt
from PIL import Image


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
    selected_coins = st.multiselect("Select Cryptocurrencies", ["ethereum", "bitcoin"])
    selected_coins.append("decentraland")
    if not selected_coins:
        st.warning("Please select at least one cryptocurrency.")
        return

    coin_colors = {
        "ethereum": "blue",
        "bitcoin": "orange",
        "decentraland": "green",
    }

    charts = []
    for coin in selected_coins:
        data = fetch_data(coin)
        chart = (
            alt.Chart(data.reset_index())
            .mark_line()
            .encode(
                x="Date:T",
                y="Price:Q",
                color=alt.value(coin_colors[coin]),
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
    st.markdown(
        """
    
    ### Whitepaper Analysis
    
    Decentraland is an Ethereum-based virtual ecosystem. It has revolutionized the digital landscape by offering users the ability to buy and sell digital properties, engage in games, trade collectibles, and participate in social interactions within a shared virtual world. Decentraland is Governed by three distinctive tokens – MANA, LAND, and Estate. Decentraland operates on a decentralized model, allowing users to create DAOs through the Argon software on the Ethereum blockchain.

    Originating in 2015 and officially launched in 2017, Decentraland gained prominence in 2021 as the cryptocurrency market and NFTs surged in popularity. The MANA tokens, initially priced at 0.02 USD, saw a substantial increase, reaching values between $6000 and $100,000 per parcel. Notably, major brands like Samsung, Adidas, Atari, and Miller Lite joined the platform, acquiring virtual "properties."
    
    Decentraland's architecture revolves around three key concepts: The World, consisting of 3D units called Parcels; The District, formed through DAO voting; and The Marketplace, the hub of Decentraland's economy. The technical aspects involve three layers – Consensus, Assets, and Real-time – ensuring efficient operation of the decentralized virtual space.
    
    Tokenomics in Decentraland involve two distinct tokens: LAND, necessary for purchasing parcels, and MANA, an ERC20 token serving as a governance token. The project faces competition from platforms like The Sandbox, Rarible, Vault Hill, and Odyssey, each offering unique features within the blockchain and metaverse spaces.

    """
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
    # df = display_multiple_plots()
    df.index = pd.to_datetime(df.index, unit="ms")
    st.line_chart(df)

    # Additional analysis or features can be added here
    st.write(
        """
    Explore additional market competition analysis or features in this section.
    You can analyze trends, identify key events, or present other relevant insights.
    """
    )

    st.subheader("Overview")
    st.markdown(
        """
    To understand the competition related Decentraland MANA we can analyse the data corresponding to similar cryptocurrencies. These are the top five cryptocurrencies similar to MANA which are direct competitors in the market:
    #### [SAND](https://coinmarketcap.com/currencies/the-sandbox/)
    - The Sandbox essentially operate on the same idea. The Sandbox is a community-driven platform where creators can monetize voxel assets and gaming experiences on the blockchain. Similar to various Metaverse real estate, users can engage, create virtual structures, and get incentives by selling certain goods on the platform. Briefly, The Sandbox is a virtual world where players can build, own, and monetize their gaming experiences in the Ethereum blockchain using the platform’s utility token SAND.
    
    #### [AXS](https://coinmarketcap.com/currencies/axie-infinity/)
    - Axie Infinity is another metaverse platform where gamers earn rewards based on their participation. The platform is built on the Ethereum blockchain and it is a decentralized application (dApp) that allows players to collect, breed, raise, battle, and trade token-based creatures known as Axies. These Axies are adorable creatures that are similar to Pokemon and CryptoKitties. 
    - They can as well play against AI-controlled Axies to earn reward. The platform’s utility token is AXS. The token is used for governance and staking on the platform. It is also used to purchase in-game items and pay for transaction fees.
    
    #### [ILV](https://coinmarketcap.com/currencies/illuvium/)
    - Illuvium as another competitor against Decentraland has a similar concept to Axie Infinity. Players combat Illuvials with the intention of capturing them to battle other players.
    - The game structure provides an option of combining Illuvials to make them stronger, putting their owners at an advantage during duels with other players. Lastly, these Illuvials can be traded as NFTs and the platform uses ILV as its utility token.
    
    ### [STX](https://coinmarketcap.com/currencies/stacks/)
    - Stacks is a blockchain that enables smart contracts and apps on Bitcoin. It is a layer-1 blockchain that uses the security of the Bitcoin network to secure its smart contracts. The platform’s utility token is STX. It is used for governance and staking on the platform.
    
    ### [THETA](https://coinmarketcap.com/currencies/theta-network/)
    - Theta aims to decentralize video streaming, operating a peer-to-peer video delivery network. The company’s promises are like many metaverse business plans: reduce costs, transfer power from companies to the masses and eliminate intermediaries. 
    - According to Theta, this vision would give a bigger piece of the pie to content creators and make video cheaper for consumers.
    
    *Data source: Coinbase, Forbes*
    
    In this section, we will be comparing the Decentraland MANA token with these 5 competitors. We will be comparing the price, market capitalization, and trading volume of these tokens. We will also perform some experiments to analyse the correlation and return on investmen between these tokens.
    """
    )

    # Exploratory Data Analysis Section
    st.subheader("Exploratory Data Analysis")
    image_path = "Images/data_example.png"
    image = Image.open(image_path)
    st.image(image, caption="Example Data")
    col1, col2 = st.columns(2)
    metric = """
    The data used for this analysis is taken from CoinMarketCap. The data is taken for the starting from the December of 2019 till date. The data is sample on monthly basis for each of the tokens. The data is then cleaned and preprocessed to remove any null values and outliers. The data is then visualized using various plots to understand the trends and patterns in the data.
    
    #### Metrics
    - Price: The price of the token at the end of the month.
    - Open: The price of the token at the start of the month.
    - High: The highest price of the token in the month.
    - Low: The lowest price of the token in the month.
    - Close: The price of the token at the end of the month.
    - Spread: The difference between the highest and lowest price of the token in the month.
    - Close Ratio: The ratio of the closing price to the highest price of the token in the month.
    - Maket Cap: The market capitalization of the token at the end of the month.
    - Volume: The trading volume of the token at the end of the month.
    - Rank: Current rank of the token in the market.
    
    """
    # col2.markdown(metric)
    st.markdown(metric)

    st.write("*Data source: CoinMarketCap*")

    col1, col2 = st.columns(2)
    col1.markdown("**Currencies by Market Capitalization:**")
    image_path_cap = "Images/currencies_by_market_cap.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Currencies by Transaction Volume:**")
    image_path_volume = "Images/currencies_by_transactin_volume.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Market Cap
        
        - From the stream plot, we see that the market cap of Theta Network is consistently higher than all the other tokens.
        - The market cap of Decentraland MANA at ATH is __0.67 times__ the market cap of Theta Network at ATH.
        - Decentraland MANA ranks consisntently at around 3rd or 4th position in terms of market cap.
        - One important observation is the consistent dominance of SAND over MANA in terms of market cap.
        - The crpto market __bull__ run of 2021 is clearly visible in both the plots.
        
        - __Qualitatively__, we can reason that Theta Network which address the singular problem of video streaming has a higher market cap than Decentraland MANA which is a virtual reality platform providing streaming, gaming, and other services.
            - This focussed approach of Theta Network is an indicator for better investments and thus a higher market cap.
            - This is also the reason for its dominance over SAND and various other tokens which are aiming to provide a virtual reality platform.
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Market Cap**")
    image_path_cap = "Images/market_cap_per_curr.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Market Cap (Stream)**")
    image_path_volume = "Images/market_cap_per_curr_stream.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Transaction Volume
        
        - In the line chart, we observe that, although similar, the transaction volume of The Sandbox is slightly higher than that of Decentraland MANA.
        - The peak in transaction volumes is observed in the last quarter of 2021 for MANA and almost all other tokens. This was due to the announcement of the __Metaverse__ by Facebook. Numerous new users joined the crypto market and the transaction volume increased.
        - Decentraland MANA ranks consisntently at around 2nd or 3rd position in terms of transaction volume.
        
        - __Qualitatively__, we see that even though the use case of both MANA and SAND is similar, the transaction volume of SAND is higher than that of MANA. Following are the possible speculations for this:
            - SAND was launched much earlier in 2011 as compare to MANA in 2017 and is a more mature platform and has been in the market for a longer time.
            - Compared to MANA, SAND has a higher real estate supply than Decentraland. Moreover, due the maturity, it has better options for users to purchase virtual land.
            - Decentraland MANA has lesser user generated content than SAND, and the gaming experience is not as good as SAND. Thus, the transaction volume is lower.
            
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Transaction Volume**")
    image_path_cap = "Images/trans_volume_per_curr.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Transaction Volume (Stream)**")
    image_path_volume = "Images/trans_volume_per_curr_stream.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Price
        
        - From the line charts and stream plot, we observe that there is one outlier, ILV (Illuvium), whose peak price is much higher than other tokens and is also much higher than the ATH of Decentraland MANA. ILV also has consistently higher prices than other tokens.
        - Excluding ILV, Decentraland MANA ranks around 2nd consistently in terms of price.
        - The latest price of Decentraland MANA is around 0.5 USD as compred to ILV which is around 100 USD. All the other tokens are in price range similar to MANA.
        - The peak in price plot is obtained around last quarter of 2021 which was due to the announcement of the __Metaverse__ and the bull run of the crypto market.
        
        - __Qualitatively__, we can analyse the huge difference between ILV and rest of tokens including Decentraland MANA. Following are the possible speculations for this:
            - ILV is a very new token. It was launched in the mid of 2021. Thus, it has not yet been in the market for a long time resulting in a higher price.
            - Further, ILV is anticipated to be a breakthrough in AAA crypto gaming. This anticipation has led to a huge demand for the token and thus the price is very high.
            - However, overall ILV has smaller market cap and transaction volume than MANA. Once, ILV is more mature and has more users, it will be able to compete with MANA. According to current context, MANA is a better investment than ILV.
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Price per unit of Currency**")
    image_path_cap = "Images/price_per_unit_curr.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Price per unit of Currency (Stream)**")
    image_path_volume = "Images/price_per_unit_curr_stream.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.subheader("Competition in 2023")
    st.markdown(
        """
        Trends observed in 2023 were consistend with overall trends of market capitalization, tranasaction volume and price per unit currency.Excluding, the crypto market bull run of 2021, there was not significant change in trends from 2020 to 2023. Let us analyse the trends in 2023.
        
        """
    )

    st.markdown(
        """
        ### Market Cap
        
        - Decentraland MANA ranks at around 4th position in terms of market cap in 2023. MANA's ATH is 0.5 times the ATH of Theta Network.
        - All tokens saw an increase in market cap in last three months of 2023. This is due to the fact that the crypto market is growing and more and more people are investing in crypto.
        - From the stream plot, the market cap of Theta Network in 2023 is consistent with overall trend. Due to its focussed approach towards use-case, it has a higher market cap than all the other tokens.
        - One interesting observation is the convergence of market cap at the end of 2023 comapred to the start of 2023.
        
        Qualitatively, the analysis of the overall trend applies to the year of 2023 as well. For improvement in market cap, Decentraland MANA needs to focus on its use-case and provide better user experience.
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Market Cap in 2023**")
    image_path_cap = "Images/market_cap_per_curr_2023.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Market Cap in 2023 (Stream)**")
    image_path_volume = "Images/market_cap_per_curr_2023_stream.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Transaction Volume
        
        - Decentraland MANA ranks at around 3rd or 4th position in terms of market cap in 2023. MANA observed a decrease in the first half of 2023 and then an increase in the second half of 2023.
        - From the stream plot, The Sandbox performed better than Decentraland MANA in 2023 consistently having around 1.5 times the transaction volume of MANA.
        - All tokens saw an increase in transaction volume in last three months of 2023. This is due to the fact that the crypto market is growing and new users are adopting crypto.
        - One interesting observation is the convergence of transaction volume at the end of 2023 comapred to the start of 2023.
        
        Qualitatively, the analysis of the overall trend applies to the year of 2023 as well. For improvement in market cap, Decentraland MANA needs to focus on its use-case and provide better user experience.
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Transaction Volume in 2023**")
    image_path_cap = "Images/transact_volume_per_curr_2023.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Transaction Volume in 2023 (Stream)**")
    image_path_volume = "Images/transact_volume_per_curr_2023_stream.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Price
        
        - Excluding ILV, Decentraland MANA ranks at around 2nd position in terms of market cap in 2023. MANA observed a lower price in the end of 2023 as compared to the start of 2023.
        - The latest price of Decentraland MANA is around 0.5 USD as compred to ILV which is around 100 USD. All the other tokens are in price range similar to MANA.
        - In 2023 too, ILV is an outlier with a much higher price than all the other tokens.
        
        Qualitatively, the analysis of the overall trend applies to the year of 2023 as well.
        
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Price per unit of Currency in 2023**")
    image_path_cap = "Images/price_per_unit_curr_2023.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Price per unit of Currency in 2023 (Stream)**")
    image_path_volume = "Images/price_per_unit_curr_2023_stream.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Future of Decentraland MANA
        
        According to _Analytics Insight_, Decentraland stands out as one of the most prominent and promising metaverse projects, earning it a featured spot on our list of upcoming cryptocurrencies poised for significant growth.

        Being the first metaverse project established, Decentraland benefits from a pioneering advantage, evolving into the largest and most valuable virtual world. This position has facilitated strategic partnerships with renowned brands, both on and off the blockchain.

        As a pioneer in the metaverse realm, Decentraland is not only widely recognized but also actively sought after by various brands, such as Samsung, JP Morgan, Adidas, Coca-Cola, Starbucks, and others. These brands have invested in virtual land within Decentraland, intending to develop projects, thereby contributing to increased investor interest and the success of MANA tokens.

        These partnerships have played a crucial role in maintaining an upward trajectory for MANA token prices since their inception. Anticipating a resurgence in the market, we foresee MANA's price recovering in 2023, reclaiming its all-time high, and potentially establishing a new record. This anticipation serves as the primary rationale for its inclusion in our selection of cryptocurrencies expected to experience significant growth this year.

        Moreover, we feature Decentraland due to our expectation that its extensive support base and strategic utilization of various cryptocurrency trends will propel MANA tokens to unprecedented heights in 2023. The incorporation of NFT technology within the metaverse and the anticipated growth in play-to-earn games on its network are poised to initiate substantial trends likely to drive MANA token prices to unprecedented levels.

        """
    )

    st.write("*source: AnalyticsInsight*")

    st.markdown(
        """
        ### Correlation between the currencies
        
        - In the correlation plot below, we can use these rules to get an idea of the interdependence between various currencies:
            - Darker colors indicate stronger correlations, while lighter colors indicate weaker correlations.
            - Positive correlations (when one variable increases, the other variable tends to increase) represented by warm colors.
            - Negative correlations (when one variable increases, the other variable tends to decrease) represented by cool colors.
        
        
        - From the correlation plot, we observe that the correlation between MANA and SAND is __0.94__ which is very high. This is due to the fact that both the tokens have similar use-case and are direct competitors.
        
        - Except for Theta Network, all the other tokens have a correlation of more than __0.8__ with MANA. This is due to the fact that all the tokens are competitors and have similar use-case.
        
        """
    )
    st.markdown("**Correlation between the currencies**")
    image_path_cap = "Images/correlation.png"
    image_cap = Image.open(image_path_cap)
    st.image(image_cap)

    st.markdown(
        """
        ## Empirical Analysis
        
        We will be conducting two analysis based on the data from 2020 to 2023.
        
        ### Return on Investment
        
        The plots show the return on investment for each token in the period of 2020 to 2023. The plot for the year 2023 shows the return on an investment made in the beginning of 2023. The plots are calculated using the closing prices at the end of the particular month. The following plots assume an initial investment of 1000 USD in the each token at the beginning. 
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Return on Investment**")
    image_path_cap = "Images/roi_curr.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Return on Investment in 2023**")
    image_path_volume = "Images/roi_curr_2023.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)

    st.markdown(
        """
        ### Earnings
        
        The plots shows earnings for each token in the period of 2020 to 2023. The plot for the year 2023 shows the earnings made in 2023. The plots are calculated using the closing prices at the end of the particular month. The following plots assume an initial investment of 1000 USD in the each token at the beginning. 
        
        Even though the overall earning of Decentraland MANA is negative, the positive earnings in 2023 indicate that the token in growing and could be a good investment in the future.
        """
    )
    col1, col2 = st.columns(2)
    col1.markdown("**Earnings**")
    image_path_cap = "Images/earnings.png"
    image_cap = Image.open(image_path_cap)
    col1.image(image_cap)
    col2.markdown("**Earnings in 2023**")
    image_path_volume = "Images/earnings_2023.png"
    image_volume = Image.open(image_path_volume)
    col2.image(image_volume)
