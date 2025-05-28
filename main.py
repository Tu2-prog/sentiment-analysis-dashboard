from src.models import SentimentAnalysisPipeline

pipeline = SentimentAnalysisPipeline(
    model_name="cardiffnlp/twitter-roberta-base-sentiment",
    label_mapping={
        "LABEL_0": -1.0,  # negative
        "LABEL_1": 0.0,  # neutral
        "LABEL_2": 1.0,  # positive
    },
)

# ...existing code...
if __name__ == "__main__":
    text = "I like this product!"
    raw_output = pipeline.sentiment_pipeline(text)[
        0
    ]  # or however you call the model inside your pipeline
    print("Raw model output:", raw_output)
    sentiment_score = pipeline.run(text)
    print(f"Sentiment score for '{text}': {sentiment_score}")
# ...existing code...
