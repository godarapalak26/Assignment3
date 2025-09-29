"""
This module provides two classes:
1. TextToImage  -> Generates an image from a text description (Stable Diffusion v2.1)
2. TextClassifier -> Performs sentiment analysis on text (DistilBERT fine-tuned on SST-2)
"""

from diffusers import StableDiffusionPipeline
from transformers import pipeline
import torch


class TextToImage:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1"):
        """
        Initialize Stable Diffusion text-to-image generator.
        Uses GPU if available, otherwise CPU.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_image(self, prompt, save_path="generated_image.png"):
        """
        Generate an image from a text prompt.
        Saves image to `save_path` and also returns that path.
        """
        image = self.pipe(prompt).images[0]
        image.save(save_path)
        return save_path


class TextClassifier:
    def __init__(self, model_id="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize a sentiment analysis classifier using DistilBERT.
        """
        self.classifier = pipeline("sentiment-analysis", model=model_id)

    def classify(self, text):
        """
        Classify sentiment of input text.
        Returns dictionary with label and confidence.
        """
        result = self.classifier(text)[0]
        return {"label": result["label"], "confidence": round(result["score"], 2)}


# Standalone testing
if __name__ == "__main__":
    # Test text-to-image
    print("Testing Text-to-Image...")
    tti = TextToImage()
    img_path = tti.generate_image("A futuristic city skyline at sunset")
    print("Image saved at:", img_path)

    # Test sentiment analysis
    print("\nTesting Sentiment Analysis...")
    clf = TextClassifier()
    test_text = "I absolutely love working on creative projects!"
    print(clf.classify(test_text))
