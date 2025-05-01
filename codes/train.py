import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------
# 1. Enhanced Chaos Governor
# --------------------------
class ChaosGovernor:
    """
    This class controls the chaotic behavior of the noise generation process.
    It uses Lorenz and Rössler attractors to dynamically adjust a key parameter
    (logistic map parameter) based on the diversity and similarity of generated images.
    """
    def __init__(self):
        # Initialize states for Lorenz and Rössler attractors
        self.lorenz_state = torch.tensor([0.1, 0.0, 0.0])
        self.rossler_state = torch.tensor([0.0, 0.0, 0.0])
        # Initial value for the logistic map parameter
        self.logistic_param = 3.99
        # Parameters for Lorenz and Rössler systems
        self.sigma, self.rho, self.beta = 10.0, 28.0, 2.667
        self.a, self.b, self.c = 0.2, 0.2, 5.7
        # Initial creative phase
        self.creative_phase = 0.0
        # History of similarity values for moving average calculation
        self.similarity_history = []

    def update(self, diversity, similarity):
        """
        Updates the chaotic systems and the logistic map parameter based on
        diversity and similarity metrics.

        Args:
            diversity (float): A measure of the diversity among generated images.
            similarity (float): A measure of the similarity between generated and real images.

        Returns:
            tuple: The updated logistic map parameter and creative phase.
        """
        # Update Lorenz system using its differential equations
        x, y, z = self.lorenz_state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        self.lorenz_state += torch.tensor([dx, dy, dz]) * 0.01  # Small time step

        # Update Rossler system using its differential equations
        x_r, y_r, z_r = self.rossler_state
        dx_r = -y_r - z_r
        dy_r = x_r + self.a * y_r
        dz_r = self.b + z_r * (x_r - self.c)
        self.rossler_state += torch.tensor([dx_r, dy_r, dz_r]) * 0.01  # Small time step

        # Adaptive parameter control for the logistic map
        self.similarity_history.append(similarity)
        # Calculate a moving average of similarity over the last 100 values
        similarity_ma = np.mean(self.similarity_history[-100:]) if self.similarity_history else 0.5

        # Combine Lorenz and Rössler states to influence the base parameter
        chaos_mix = torch.sigmoid(0.5 * (self.lorenz_state[0] + self.rossler_state[2]))
        base_param = 3.6 + 0.4 * chaos_mix.item()

        # Adjust the logistic parameter based on similarity and diversity
        # Reduce the parameter if similarity is too high (images are too similar)
        similarity_correction = 0.2 * torch.sigmoid(torch.tensor(5 * (0.6 - similarity_ma))).item()
        # Increase the parameter if diversity is low (images are too different)
        diversity_boost = 0.1 * torch.tanh(torch.tensor(10 * (diversity - 0.3))).item()

        # Clip the logistic parameter to a reasonable range
        self.logistic_param = np.clip(
            base_param - similarity_correction + diversity_boost,
            3.4, 4.0
        )

        # Evolve the creative phase using a combination of Lorenz and Rössler states
        self.creative_phase = (self.creative_phase + 0.01 *
                             (self.lorenz_state[1].item() + self.rossler_state[0].item())) % 1.0
        return self.logistic_param, self.creative_phase

# --------------------------
# 2. Quantum Chaos Generator
# --------------------------
class QuantumChaosGenerator:
    """
    Generates latent noise vectors using a quantum-inspired approach combined with
    chaotic control from the ChaosGovernor.
    """
    def __init__(self, latent_dim=256):
        self.latent_dim = latent_dim
        self.governor = ChaosGovernor()
        self.quantum_state = None

    def generate(self, batch_size, steps=100, diversity=0.5, similarity=0.5):
        """
        Generates a batch of latent noise vectors.

        Args:
            batch_size (int): The number of noise vectors to generate.
            steps (int): The number of iterations for the chaotic transformation.
            diversity (float): The diversity metric from the CreativityOracle.
            similarity (float): The similarity metric from the CreativityOracle.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, latent_dim, 1, 1) containing the generated noise vectors.
        """
        # Get chaotic parameters from the governor
        chaos_param, phase = self.governor.update(diversity, similarity)

        # Initialize the quantum state if it's not already initialized or the batch size changes
        if self.quantum_state is None or self.quantum_state.size(0) != batch_size:
            self.quantum_state = torch.randn(batch_size, self.latent_dim, 1, 1)
            
        noise = self.quantum_state.clone()
        for _ in range(steps):
            # Core chaotic transformations
            noise = chaos_param * noise * (1 - torch.abs(noise))
            phase_factor = np.exp(1j * np.pi * (phase + noise.mean().item()))
            phase_rot = torch.tensor(phase_factor.real, dtype=noise.dtype, device=noise.device)
            noise = noise * phase_rot
            noise += 0.1 * torch.randn_like(noise)
            noise = torch.clamp(noise, -1.0, 1.0)
            
        self.quantum_state = 0.9 * self.quantum_state + 0.1 * noise
        return self.quantum_state

# --------------------------
# 3. Generator Architecture
# --------------------------
class ChaoticResBlock(nn.Module):
    """
    A residual block with a chaotic gate, introducing a chaotic element into the generator network.
    """
    def __init__(self, channels):
        super().__init__()
        # Two convolutional layers with spectral normalization
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1)),
        )
        # Learnable parameter for the chaotic gate
        self.chaos_gate = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # Apply the residual connection with a chaotic modulation using a sine function
        return x + torch.sin(self.chaos_gate * np.pi * self.conv(x))

class MultiverseGenerator(nn.Module):
    """
    The generator network that transforms latent noise vectors into images.
    It uses a series of transposed convolutional layers and ChaoticResBlocks.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        # Initial transposed convolution to upsample the latent vector
        self.init_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Sequential blocks of ChaoticResBlocks and transposed convolutions for upsampling
        self.blocks = nn.Sequential(
            *[self._build_block(512, 256, 4)],  # change into (512,256,1) if you are using cpu
            *[self._build_block(256, 128, 4)],  # change into (256,128,1) if you are using cpu
            *[self._build_block(128, 64, 4)],   # change into (128,64,1) if you are using cpu
            *[self._build_block(64, 32, 4)],    # change into (64,32,1) if you are using cpu
            *[self._build_block(32, 16, 4)],    # change into (32,16,1) if you are using cpu
            *[self._build_block(16, 8, 4)]      # change into (16,8,1) if you are using cpu
        )
        
        # Final convolutional layer to produce the RGB image
        self.final = nn.Sequential(
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Tanh()  # Output pixel values in the range [-1, 1]
        )

    def _build_block(self, in_c, out_c, num_layers):
        """
        Builds a block of ChaoticResBlocks and a transposed convolution.

        Args:
            in_c (int): Number of input channels.
            out_c (int): Number of output channels.
            num_layers (int): Number of ChaoticResBlocks in the block.

        Returns:
            nn.Sequential: A sequential container of layers.
        """
        layers = []
        for _ in range(num_layers):
            layers += [
                ChaoticResBlock(in_c),
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            ]
            in_c = out_c  # Update input channels for the next layer
        return nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass through the generator network.

        Args:
            z (torch.Tensor): A latent noise vector of shape (batch_size, latent_dim, 1, 1).

        Returns:
            torch.Tensor: A generated image of shape (batch_size, 3, 256, 256).
        """
        x = self.init_conv(z)
        return self.final(self.blocks(x))

# --------------------------
# 4. Discriminator
# --------------------------
class ChaosAwareDiscriminator(nn.Module):
    """
    The discriminator network that distinguishes between real and generated images.
    It uses convolutional layers and ChaoticResBlocks.
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Initial convolutional layer with spectral normalization
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            ChaoticResBlock(64),
            # Subsequent convolutional layers with spectral normalization
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            ChaoticResBlock(128),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            ChaoticResBlock(256),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): An image of shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing the discriminator's predictions.
        """
        return self.main(x).view(x.size(0), -1).mean(1)

# --------------------------
# 5. Creativity Oracle
# --------------------------
class CreativityOracle(nn.Module):
    """
    Evaluates the creativity of generated images based on similarity to real images
    and diversity within the generated batch.  Uses a pre-trained ConvNeXt model
    as a feature extractor.
    """
    def __init__(self):
        super().__init__()
        # Use a pre-trained ConvNeXt-Base model as a fixed feature extractor
        self.feature_extractor = torchvision.models.convnext_base(pretrained=True).features
        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Separate networks for predicting similarity and diversity
        self.similarity = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.diversity = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the creativity oracle.

        Args:
            x (torch.Tensor): A batch of generated images of shape (batch_size, 3, 256, 256).

        Returns:
            tuple: Tensors of shape (batch_size,) representing similarity and diversity scores.
        """
        with torch.no_grad():
            # Extract features using the pre-trained ConvNeXt model
            features = self.feature_extractor((x+1)/2).mean([2,3])  # Normalize to [0, 1] and global average pooling
        # Predict similarity and diversity based on the extracted features
        return self.similarity(features).squeeze(), self.diversity(features).squeeze()

# --------------------------
# 6. Training System
# --------------------------
class CreativeAITrainer:
    """
    Orchestrates the training process of the GAN, including updating the generator,
    discriminator, and the chaos governor.
    """
    def __init__(self, dataset_path, device='cpu'):
        self.device = device
        # Load the dataset
        self.dataset = ArtDataset(dataset_path)
        # Create a data loader for batching and shuffling
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True, pin_memory=True)
        
        # Initialize the generator, discriminator, chaos generator, and creativity oracle
        self.G = MultiverseGenerator().to(device)
        self.D = ChaosAwareDiscriminator().to(device)
        self.chaos_gen = QuantumChaosGenerator()
        self.oracle = CreativityOracle().to(device)
        
        # Optimizers for the generator and discriminator
        self.opt_G = optim.AdamW(self.G.parameters(), lr=0.0002, betas=(0.5,0.999))
        self.opt_D = optim.AdamW(self.D.parameters(), lr=0.0001, betas=(0.5,0.999))

    def train(self, epochs=6000):
        """
        Trains the GAN for a specified number of epochs.

        Args:
            epochs (int): The number of training epochs.
        """
        # Create directories for saving checkpoints and sample outputs
        os.makedirs("../checkpoints", exist_ok=True)
        os.makedirs("../sample_input", exist_ok=True)
        os.makedirs("../sample_output", exist_ok=True)
        
        for epoch in range(epochs):
            for batch_idx, real_imgs in enumerate(self.loader):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                # --- Discriminator Update ---
                self.opt_D.zero_grad()
                
                # Generate latent codes
                with torch.no_grad():
                    z = self.chaos_gen.generate(batch_size)
                    fake_imgs = self.G(z)
                
                real_pred = self.D(real_imgs)
                fake_pred = self.D(fake_imgs.detach())
                
                # Calculate losses
                loss_real = -torch.log(real_pred + 1e-8).mean()
                loss_fake = -torch.log(1 - fake_pred + 1e-8).mean()
                gp = self._gradient_penalty(real_imgs, fake_imgs)
                
                loss_D = loss_real + loss_fake + 10*gp
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                self.opt_D.step()
                
                # --- Generator Update ---
                self.opt_G.zero_grad()
                z = self.chaos_gen.generate(batch_size)
                fake_imgs = self.G(z)
                fake_pred = self.D(fake_imgs)
                
                # Get creativity metrics
                with torch.no_grad():
                    similarity, diversity = self.oracle(fake_imgs)
                    similarity = similarity.mean()
                    diversity = diversity.mean()
                
                # Adaptive loss weights
                current_chaos = self.chaos_gen.governor.logistic_param
                chaos_weight = np.clip(current_chaos / 4.0, 0.3, 0.7)
                similarity_weight = 2.0 - chaos_weight
                
                # Loss components
                loss_adv = -torch.log(fake_pred + 1e-8).mean()
                loss_sim = (1 - similarity) * similarity_weight
                loss_div = (1 - diversity) * (1 - similarity_weight)
                
                loss_G = chaos_weight*loss_adv + 0.5*loss_sim + 0.3*loss_div
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
                self.opt_G.step()
                
                # Logging
                if batch_idx % 10 == 0:
                    print(f"\nEpoch {epoch+1}/{epochs} | Batch {batch_idx+1}")
                    print(f"Chaos: {current_chaos:.2f} | Similarity: {similarity.item():.4f}")
                    print(f"G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f}")
                    print(f"Real Score: {real_pred.mean().item():.4f} | Fake Score: {fake_pred.mean().item():.4f}")

            # Save progress
            if (epoch+1) % 10 == 0: # Save every 10 epochs
                self._save_sample(epoch)
                torch.save(self.G.state_dict(), f"../checkpoints/generator_epoch_{epoch+1}.pth")
                print(f"\n=== Saved checkpoint at epoch {epoch+1} ===")

    def _gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real + (1-alpha) * fake).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _save_sample(self, epoch):
        with torch.no_grad():
            z = self.chaos_gen.generate(1)
            fake = self.G(z).cpu()
            fake = (fake + 1)/2
            plt.imsave(f"../sample_output/sample_{epoch+1}.png",
                      fake.squeeze().permute(1,2,0).numpy(),
                      dpi=300)

# --------------------------
# 7. Dataset Class
# --------------------------
class ArtDataset(Dataset):
    def __init__(self, txt_file):
        with open(txt_file) as f:
            self.paths = [line for line in f.read().splitlines() if line.strip() != '']
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform(img)

# --------------------------
# 8. Main Execution
# --------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Creative AI on {device}")
    
    trainer = CreativeAITrainer(
        dataset_path="inputs.txt",
        device=device
    )
    
    try:
        trainer.train(epochs=6000)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        torch.save(trainer.G.state_dict(), "../checkpoints/final_generator.pth")
