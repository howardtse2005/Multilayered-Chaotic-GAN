def generate_painting_txt(output_file, num_images):
    """
    Generates a text file with lines like "sample_input/1.jpg", "sample_input/2.jpg", ..., "sample_input/4877.jpg".

    Args:
        output_file: The path to the output text file (e.g., "inputs.txt").
        num_images: The number of image paths to generate.
    """

    try:
        with open(output_file, 'w') as f:
            for i in range(1, num_images + 1):
                f.write(f"../sample_input/{i}.jpg\n")
        print(f"Successfully created '{output_file}' with {num_images} image paths.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    output_file = "inputs.txt"  # Replace with your desired output file path
    num_images = 4877  # Number of images paths to write in inputs.txt
    generate_painting_txt(output_file, num_images)
