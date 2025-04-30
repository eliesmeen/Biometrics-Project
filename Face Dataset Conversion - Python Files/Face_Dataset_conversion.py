import os
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch

# Load the beard-generation model
print("Loading AI model")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Override the safety checker
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# Define folders - CHANGE TO YOUR OWN FILE PATH!
input_folder = r"Facial Hair Datasets\faces"
output_folder = r"Facial Hair Datasets\Faces_Beard_Adjusted"
os.makedirs(output_folder, exist_ok=True)

# Prompt for beard addition to dataset
prompt = "add realistic facial hair that matches the hair color of the person"

# Loop through each image and apply the beard
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            original_size = image.size  # Saved original dimensions

            print(f"[INFO] Processing: {filename}")
            edited = pipe(prompt=prompt, image=image).images[0]

            # Resize output to match original
            edited = edited.resize(original_size)

            # Save with high quality
            save_path = os.path.join(output_folder, filename)
            edited.save(save_path, quality=95)

            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Failed to process: {filename}: {e}")
