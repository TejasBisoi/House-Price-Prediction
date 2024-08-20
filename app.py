import yfinance as yf
import streamlit as st
import pandas as pd



st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":tada:",
    layout="centered",
    initial_sidebar_state="auto",
    theme={
        "primaryColor": "blue",
        "backgroundColor": "white",
        "secondaryBackgroundColor": "gray",
        "textColor": "black",
        "font": "sans serif"
    }
)




st.write("""
# Simple Stock Price App
Shown below are the stock closing price and volume of Apple!
         
""")

tickerSymbol='AAPL'
tickerData=yf.Ticker(tickerSymbol)
tickerDf=tickerData.history(period='1d',start='2010-5-31',end='2020-5-31')

st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)
