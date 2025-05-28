import numpy as np
from transformers import pipeline


class SentimentAnalysisPipeline:
    """
    A class to handle sentiment analysis using a pre-trained model.
    """

    def __init__(
        self,
        model_name: str,
        label_mapping: dict[str, float],
    ) -> None:
        self.model_name = model_name
        self.label_mapping = label_mapping
        self.sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            top_k=3,
        )

    def run(self, text: str) -> float:
        """
        Run the sentiment analysis pipeline on the input text.
        Args:
            text: The input text to analyze.
        Returns:
            A float value representing the sentiment score.
        """

        sentiment: list[dict[str]] = self.sentiment_pipeline(text)[0]

        positivity = 0.0
        for label_score_dict in sentiment:
            label: str = label_score_dict["label"]
            score: float = label_score_dict["score"]

            if label in self.label_mapping:
                positivity += self.label_mapping[label] * score
        positivity = np.clip(positivity, -1, 1)
        return positivity
