"""
Text Classification (Sentiment Analysis)
Publisher: Hugging Face
This script uses DistilBERT (fine-tuned on SST-2) to classify text sentiment.
"""

from transformers import pipeline

class TextClassifier:
    def __init__(self, model_id="distilbert-base-uncased-finetuned-sst-2-english"):
        # Load sentiment analysis pipeline from Hugging Face
        self.classifier = pipeline("sentiment-analysis", model=model_id)

    def classify(self, text):
        """
        Classify the sentiment of the input text.
        Returns label (Positive/Negative) and confidence score.
        """
        result = self.classifier(text)[0]
        return {"label": result["label"], "confidence": round(result["score"], 2)}


# Run test when file is executed directly
if __name__ == "__main__":
    clf = TextClassifier()
    example = "I really enjoy learning new programming skills!"
    print(clf.classify(example))
