# test.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

# Assuming the Generator class is defined in chaos.py
from train import Generator, ChaosGenerator

def generate_image(model_path, latent_dim=100, device="cpu"):
    # Load the generator model
    generator = Generator(latent_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Generate chaotic noise
    chaos = ChaosGenerator(latent_dim)
    z = chaos.generate(1).to(device)

    # Generate the image
    with torch.no_grad():
        fake_image = generator(z)

    # Denormalize and convert to PIL Image
    fake_image = fake_image.squeeze(0).cpu().detach().numpy()
    fake_image = (fake_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    fake_image = np.transpose(fake_image, (1, 2, 0))
    fake_image = (fake_image * 255).astype(np.uint8)
    fake_image = Image.fromarray(fake_image)

    return fake_image

if __name__ == "__main__":
    # Replace with the path to your saved model
    model_path = "checkpoints/final_generator.pth"  
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
    else:
        generated_image = generate_image(model_path)
        generated_image.save("generated_image.png")
        print("Image generated and saved as generated_image.png")
