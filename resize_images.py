"""
Script to resize all images in the data/train folder and save them to a new folder.
"""
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
SOURCE_FOLDER = "data/train"
OUTPUT_FOLDER = "data/train_resized_518x518"  # Specify the output folder name here
TARGET_SIZE = (518, 518)  # Specify the target size (width, height) here
MAINTAIN_ASPECT_RATIO = False  # Set to True to maintain aspect ratio with padding

def resize_image(image_path, output_path, target_size, maintain_aspect=False):
    """
    Resize an image to the target size.
    
    Args:
        image_path: Path to the source image
        output_path: Path to save the resized image
        target_size: Tuple of (width, height)
        maintain_aspect: If True, maintains aspect ratio with padding
    """
    try:
        img = Image.open(image_path)
        
        if maintain_aspect:
            # Resize while maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            # Create a new image with padding
            new_img = Image.new("RGB", target_size, (0, 0, 0))
            paste_x = (target_size[0] - img.width) // 2
            paste_y = (target_size[1] - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        else:
            # Resize without maintaining aspect ratio
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save the resized image
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    """Main function to resize all images in the source folder."""
    # Create output folder if it doesn't exist
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from the source folder
    source_path = Path(SOURCE_FOLDER)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {SOURCE_FOLDER}")
    print(f"Resizing to {TARGET_SIZE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Maintain aspect ratio: {MAINTAIN_ASPECT_RATIO}")
    print("-" * 50)
    
    # Process all images
    success_count = 0
    for img_file in tqdm(image_files, desc="Resizing images"):
        output_file = output_path / img_file.name
        if resize_image(img_file, output_file, TARGET_SIZE, MAINTAIN_ASPECT_RATIO):
            success_count += 1
    
    print("-" * 50)
    print(f"Successfully resized {success_count}/{len(image_files)} images")
    print(f"Resized images saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
