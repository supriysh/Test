from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import requests
from io import BytesIO
import os
from datetime import datetime

class NanoBananaAI:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        """
        Initialize Nano Banana AI.
        Args:
            model_id (str): Hugging Face model ID.
            device (str): "cuda" for GPU, "cpu" for CPU.
        """
        print("🍌 Initializing Nano Banana AI... (uncensored mode)")
        self.model_id = model_id
        self.device = device
        self.history = []  # Store past interactions
        self._load_models()

    def _load_models(self):
        """Load Stable Diffusion models (generator and img2img)."""
        print("Loading models...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None  # Disable safety checker
        ).to(self.device)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None  # Disable safety checker
        ).to(self.device)
        print("✅ Models loaded. Nano Banana is ready to generate chaos.")

    def generate_image(self, prompt, height=512, width=512, save=True):
        """
        Generate an image from a prompt.
        Args:
            prompt (str): Text prompt.
            height (int): Image height.
            width (int): Image width.
            save (bool): Save the image to disk.
        Returns:
            PIL.Image: Generated image.
        """
        print(f"🎨 Generating: '{prompt}'...")
        image = self.pipe(prompt, height=height, width=width, guidance_scale=7.5).images[0]
        if save:
            filename = f"nano_banana_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image.save(filename)
            self.history.append({"type": "generate", "prompt": prompt, "filename": filename})
            print(f"💾 Image saved as '{filename}'.")
        return image

    def edit_image(self, input_image, prompt, strength=0.75, save=True):
        """
        Edit an image using a prompt.
        Args:
            input_image (PIL.Image or str): Input image or file path.
            prompt (str): Edit prompt.
            strength (float): How much to edit the image (0.1-1.0).
            save (bool): Save the edited image.
        Returns:
            PIL.Image: Edited image.
        """
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        print(f"✏️ Editing image with prompt: '{prompt}'...")
        image = self.img2img_pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=7.5
        ).images[0]
        if save:
            filename = f"nano_banana_edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image.save(filename)
            self.history.append({"type": "edit", "prompt": prompt, "filename": filename})
            print(f"💾 Edited image saved as '{filename}'.")
        return image

    def upload_image_from_url(self, url):
        """
        Upload an image from a URL.
        Args:
            url (str): Image URL.
        Returns:
            PIL.Image: Uploaded image.
        """
        print(f"📥 Uploading image from URL: {url}...")
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(filename)
        print(f"💾 Image saved as '{filename}'.")
        return image

    def chat(self):
        """Interactive chat mode for Nano Banana AI."""
        print("=== 🍌 NANO BANANA AI (UNCENSORED) ===")
        print("Type 'quit' to exit.")
        while True:
            print("\nOptions:")
            print("1. Generate image from prompt")
            print("2. Upload and edit image")
            print("3. View history")
            print("4. Quit")
            choice = input("Choose an option (1/2/3/4): ").strip()
            if choice == "1":
                prompt = input("Enter your prompt: ")
                self.generate_image(prompt)
            elif choice == "2":
                url = input("Enter image URL: ")
                image = self.upload_image_from_url(url)
                prompt = input("Enter edit prompt: ")
                self.edit_image(image, prompt)
            elif choice == "3":
                print("\n📜 History:")
                for i, entry in enumerate(self.history):
                    print(f"{i+1}. {entry['type']}: {entry['prompt']} → {entry['filename']}")
            elif choice == "4":
                print("👋 Later, degenerate.")
                break
            else:
                print("❌ Invalid choice. Try again.")

    def clear_history(self):
        """Clear interaction history."""
        self.history = []
        print("🧹 History cleared.")