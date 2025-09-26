"""
Text-to-Image Generator
Publisher: Stability AI
This script generates an image from a text description using Stable Diffusion v2.1.
"""

from diffusers import StableDiffusionPipeline
import torch

class TextToImage:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1"):
        # Use GPU if available, otherwise run on CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Stable Diffusion model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, prompt, save_path="generated_image.png"):
        """
        Generate an image from the given text prompt.
        The output is saved as a PNG file.
        """
        image = self.pipe(prompt).images[0]
        image.save(save_path)
        return save_path


# Run test when file is executed directly
if __name__ == "__main__":
    generator = TextToImage()
    path = generator.generate_image("A sunny beach with palm trees and blue water")
    print("Image saved at:", path)
