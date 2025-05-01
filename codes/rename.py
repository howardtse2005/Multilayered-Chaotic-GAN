import os

def rename_images(folder_path):
    """
    Renames all images in a folder to sequential numbers (1.jpg, 2.jpg, etc.).

    Args:
        folder_path: The path to the folder containing the images.
    """

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add more if needed
    image_count = 0

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_count += 1
            old_filepath = os.path.join(folder_path, filename)
            new_filename = f"{image_count}{os.path.splitext(filename)[1]}"  # Keep original extension
            new_filepath = os.path.join(folder_path, new_filename)

            os.rename(old_filepath, new_filepath)
            print(f"Renamed '{filename}' to '{new_filename}'")

    print(f"\nRenamed {image_count} images in '{folder_path}'.")

# --- Example Usage ---
if __name__ == "__main__":
    folder_path = "paintings"
    rename_images(folder_path)
