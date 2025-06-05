# CreativeAI: Multilayered Chaotic GAN Image Generator

This project is a fun experiment on the generation of artistic images using multilayered chaotic functions (based on Lorenz and Rössler attractors) rather than traditional linear Gaussian functions on Generative Adversarial Network (GAN) architecture. This serves as a test to the philosophy of creativity: "creativity can be described as multilayered chaotic functions that are perfectly tuned. Slight change in the parameters of chaos functions leads to insanity and chaos rather than creativity." That philosophy is just my 3am thoughts btw :) Note that this project is just for fun and not for research purposes (although this might be a useful research in the future to develop an AI that is truly creative or even conscious) 

## Video Demo
This video shows the output images (timelapse) after several epochs.


https://github.com/user-attachments/assets/bbccea69-b7d4-461d-adc7-6c5be008c2ea



## System Components

1.  **ChaosGovernor:** Controls the chaotic behavior of the noise generation by dynamically adjusting a logistic map parameter based on diversity and similarity metrics.
2.  **QuantumChaosGenerator:** Generates latent noise vectors using a quantum-inspired approach, modulated by the ChaosGovernor.
3.  **MultiverseGenerator:** Transforms latent noise vectors into 256x256 images using transposed convolutional layers and ChaoticResBlocks.
4.  **ChaosAwareDiscriminator:** Distinguishes between real and generated images, also incorporating ChaoticResBlocks.
5.  **CreativityOracle:** Evaluates the creativity of generated images based on similarity to real images and diversity within the generated batch, using a pre-trained ConvNeXt model as a feature extractor.
6.  **CreativeAITrainer:** Orchestrates the training process, managing the generator, discriminator, chaos governor, and creativity oracle.
7.  **ArtDataset:** Handles loading and preprocessing of the training dataset, consisting of artistic images.

## Getting Started

1.  **Clone the repository:**
    ```bash
    cd ~
    git clone https://github.com/howardtse2005/Multilayered-Chaotic-GAN.git
    cd Multilayered-Chaotic-GAN/
    ```

2.  **Install dependencies:**
    The code was tested on Python 3.9 in Conda Environment
    ```bash
    conda create -n <env_name> python=3.9
    conda install pytorch::torchvision conda-forge::pytorch conda-forge::pillow "numpy<2"
    ```
    (You may need to adjust this command based on your CUDA configuration if using a GPU. See the PyTorch documentation for details.)

3.  **Prepare the image dataset:**
    Sample images can be found on Kaggle: https://www.kaggle.com/datasets/heyitsfahd/paintings. Create sample_input directory and put the images under sample_input directory.
    ```bash
    mkdir sample_input && cd ..
    ```
    To rename the image datasets, run the rename.py. To write the image paths to the inputs.txt, run the writer.py
    ```bash
    cd codes/
    python3 rename.py
    python3 writer.py
    ```
4.  **Train the model:**
    ```bash
    python3 train.py
    ```
    (Training can take a significant amount of time, especially on a CPU. Consider using a GPU for faster training.)

5.  **Generate images:**
    After training, you can see the generated images (every 10 epochs) under the sample_output. Or you can test it using test.py:
    ```bash
    python3 test.py
    ```
    This will save a generated image as `generated_image.png` in the project directory.

## Notes

*   **GPU Training:** For faster training, it is highly recommended to use a GPU with sufficient memory. The generator architecture can be memory-intensive, especially with the increased number of layers in the `_build_block` method. If you encounter out-of-memory errors, consider reducing the batch size in `CreativeAITrainer`.
*   **CPU Training:** If you are training on a CPU, you may need to modify the generator architecture in `train.py` to reduce memory usage.  Specifically, in the `MultiverseGenerator` class, you can change the `num_layers` argument in the `_build_block` calls to 1:
    ```python
    self.blocks = nn.Sequential(
        *[self._build_block(512, 256, 1)],
        # ... other blocks ...
    )
    ```
    This will significantly reduce the model size, but may also affect the quality of generated images.
*   **Dataset:** The quality and diversity of your training dataset will greatly influence the results.

## Future Directions

*   Explore more multilayers of chaos functions (chaos functions that determines the chaos parameters of another chaos functions)
*   Investigate different generator and discriminator architectures to further enhance image quality and diversity.
*   Most importantly, add text generation (Natural Language Processing) on each epoch to describe the "creative thinking" process of the model
