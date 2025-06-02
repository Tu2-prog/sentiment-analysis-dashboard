import streamlit as st
from .models import SentimentAnalysisPipeline


pipeline = SentimentAnalysisPipeline(
    model_name="cardiffnlp/twitter-roberta-base-sentiment",
    label_mapping={
        "LABEL_0": -1.0,  # negative
        "LABEL_1": 0.0,  # neutral
        "LABEL_2": 1.0,  # positive
    },
)


def main():
    st.title("Sentiment Analysis Dashboard")
    st.write("Welcome to the Sentiment Analysis Dashboard!")
    # Add an input field for the user input
    user_input = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze"):
        # Call the sentiment analysis function and display the result
        result = pipeline.run(user_input)
        st.write("Sentiment Analysis Result:")
        st.write(result)
