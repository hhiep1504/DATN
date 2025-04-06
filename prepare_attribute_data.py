import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json

# Define column meanings based on read_annotations.py
COLUMN_MEANINGS = {
    0: "Frame",
    1: "ID",
    2: "x",
    3: "y",
    4: "h",
    5: "w",
    6: "flag",
    7: "yaw",
    8: "pitch",
    9: "roll",
    10: "Gender",  # 0=Male, 1=Female, 2=Unknown
    11: "Age",     # 0=0-11, 1=12-17, 2=18-24, 3=25-34, 4=35-44, 5=45-54, 6=55-64, 7=>65, 8=Unknown
    12: "Height",
    13: "Body_Volume",
    14: "Ethnicity",
    15: "Hair_Color",
    16: "Hairstyle",
    17: "Beard",
    18: "Moustache",
    19: "Glasses",  # 0=Normal glass, 1=Sun glass, 2=No, 3=Unknown
    20: "Head_Accessories",  # 0=Hat, 1=Scarf, 2=Neckless, 3=Cannot see, 4=Unknown
    21: "Upper_Body_Clothing",  # 0=T Shirt, 1=Blouse, 2=Sweater, 3=Coat...
    22: "Lower_Body_Clothing",  # 0=Jeans, 1=Leggins, 2=Pants, 3=Shorts, 4=Skirt...
    23: "Feet",
    24: "Accessories",  # 0=Bag, 1=Backpack Bag, 2=Rolling Bag...
    25: "Action"  # 0=Walking, 1=Running, 2=Standing, 3=Sitting...
}

# Define mapping from numeric values to human-readable labels
ATTRIBUTE_MAPPINGS = {
    "Gender": {0: "Male", 1: "Female", 2: "Unknown"},
    "Age": {
        0: "Child (0-11)", 
        1: "Teen (12-17)", 
        2: "Young (18-24)", 
        3: "Young Adult (25-34)", 
        4: "Adult (35-44)", 
        5: "Middle Age (45-54)", 
        6: "Senior (55-64)", 
        7: "Elderly (65+)", 
        8: "Unknown"
    },
    "Glasses": {0: "Normal", 1: "Sunglasses", 2: "No", 3: "Unknown"},
    "Head_Accessories": {0: "Hat", 1: "Scarf", 2: "Neckless", 3: "None", 4: "Unknown"},
    "Upper_Body_Clothing": {
        0: "T-Shirt", 1: "Blouse", 2: "Sweater", 3: "Coat", 4: "Bikini", 
        5: "Naked", 6: "Dress", 7: "Uniform", 8: "Shirt", 9: "Suit", 
        10: "Hoodie", 11: "Cardigan", 12: "Unknown"
    },
    "Lower_Body_Clothing": {
        0: "Jeans", 1: "Leggins", 2: "Pants", 3: "Shorts", 4: "Skirt", 
        5: "Bikini", 6: "Dress", 7: "Uniform", 8: "Suit", 9: "Unknown"
    },
    "Accessories": {
        0: "Bag", 1: "Backpack", 2: "Rolling Bag", 3: "Umbrella", 
        4: "Sport Bag", 5: "Market Bag", 6: "Nothing", 7: "Unknown"
    },
    "Action": {
        0: "Walking", 1: "Running", 2: "Standing", 3: "Sitting", 4: "Cycling", 
        5: "Exercising", 6: "Petting", 7: "Talking on Phone", 8: "Leaving Bag", 
        9: "Fall", 10: "Fighting", 11: "Dating", 12: "Offending", 13: "Trading"
    }
}

def process_annotations_and_images(annotations_file, images_dir, output_dir):
    """
    Match annotations with already cropped person images
    
    Args:
        annotations_file: Path to P-DESTRE annotations file
        images_dir: Directory containing cropped person images
        output_dir: Directory to save organized data
    """
    # Create output directories
    organized_dir = os.path.join(output_dir, 'organized')
    os.makedirs(organized_dir, exist_ok=True)
    
    # Read annotations - try different encodings
    print(f"Reading annotations from {annotations_file}")
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    content = None
    successful_encoding = None
    
    for encoding in encodings:
        try:
            with open(annotations_file, 'r', encoding=encoding) as f:
                content = f.readlines()
                successful_encoding = encoding
                break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print(f"Error: Unable to read the file with common encodings: {annotations_file}")
        return
    
    print(f"Successfully read file with encoding: {successful_encoding}")
    
    # Parse the content manually - the file is comma-separated
    annotations = []
    for line in content:
        # Split by comma
        values = line.strip().split(',')
        if len(values) >= 26:  # Make sure we have all required fields
            annotations.append(values)
    
    print(f"Found {len(annotations)} annotations")
    
    # Extract the annotation filename (used to construct image paths)
    annotation_basename = os.path.basename(annotations_file).split('.')[0]
    print(f"Annotation basename: {annotation_basename}")
    
    # Find the matching image directory in jpg_Extracted_PIDS
    image_dir_match = None
    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        if os.path.isdir(item_path) and annotation_basename in item:
            image_dir_match = item_path
            break
    
    if not image_dir_match:
        print(f"Error: Could not find matching image directory for {annotation_basename}")
        return
    
    print(f"Found matching image directory: {image_dir_match}")
    
    # Create a mapping from frame number to subdirectory
    frame_dirs = {}
    for item in os.listdir(image_dir_match):
        item_path = os.path.join(image_dir_match, item)
        if os.path.isdir(item_path):
            try:
                frame_num = int(item)
                frame_dirs[frame_num] = item_path
            except ValueError:
                continue
    
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Create a dictionary to store attribute information for each image
    attribute_data = {}
    
    # Process each annotation
    print("Processing annotations...")
    matched_count = 0
    
    # Create frame-to-annotation mapping
    frame_annotations = {}
    for row in annotations:
        frame_num = int(row[0])
        if frame_num not in frame_annotations:
            frame_annotations[frame_num] = []
        frame_annotations[frame_num].append(row)
    
    # Process each frame directory
    for frame_num, frame_dir in tqdm(frame_dirs.items()):
        # Get all images in this directory
        images = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images or frame_num not in frame_annotations:
            continue
        
        # Get the annotations for this frame
        frame_annots = frame_annotations[frame_num]
        
        # Usually there's only one image per frame directory (the person crop)
        # We'll assign the annotation directly without trying to match bounding boxes
        for img_file in images:
            img_path = os.path.join(frame_dir, img_file)
            
            # Choose the first annotation for this frame
            row = frame_annots[0]
            
            # Extract person ID for the filename
            person_id = row[1]
            
            # Generate a unique ID for the person
            person_unique_id = f"{annotation_basename}_frame{frame_num}_id{person_id}"
            
            # Create the organized filename
            organized_filename = f"{person_unique_id}.jpg"
            organized_path = os.path.join(organized_dir, organized_filename)
            
            # Copy the image to the organized directory
            shutil.copy(img_path, organized_path)
            matched_count += 1
            
            # Extract attributes from annotation
            attributes = {}
            for attr_name, mapping in ATTRIBUTE_MAPPINGS.items():
                col_idx = next((i for i, name in COLUMN_MEANINGS.items() if name == attr_name), None)
                if col_idx is not None and col_idx < len(row):
                    try:
                        attr_value = int(float(row[col_idx]))
                        # Convert numeric value to human-readable label
                        attributes[attr_name] = mapping.get(attr_value, f"Unknown ({attr_value})")
                    except (ValueError, IndexError):
                        attributes[attr_name] = "Unknown"
            
            # Store attributes for this image
            attribute_data[organized_filename] = attributes
    
    # Save attribute data as JSON
    with open(os.path.join(output_dir, 'attributes.json'), 'w') as f:
        json.dump(attribute_data, f, indent=2)
    
    print(f"Matched {matched_count} images with annotations, saved to {output_dir}")
    
    if matched_count > 0:
        # Create train/val split
        create_train_val_split(output_dir, organized_dir, attribute_data, val_ratio=0.2)
    else:
        print("No valid matches were found. Check the annotation format and image paths.")

def create_train_val_split(output_dir, organized_dir, attribute_data, val_ratio=0.2):
    """Create training and validation splits"""
    # Create train and val directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of all organized filenames
    all_images = list(attribute_data.keys())
    
    # Shuffle the list
    np.random.shuffle(all_images)
    
    # Split into train and validation sets
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Copy files to train and val directories
    train_attributes = {}
    val_attributes = {}
    
    print("Creating train/val split...")
    # Copy train images
    for img_filename in tqdm(train_images, desc="Train"):
        src = os.path.join(organized_dir, img_filename)
        dst = os.path.join(train_dir, img_filename)
        shutil.copy(src, dst)
        train_attributes[img_filename] = attribute_data[img_filename]
    
    # Copy val images
    for img_filename in tqdm(val_images, desc="Val"):
        src = os.path.join(organized_dir, img_filename)
        dst = os.path.join(val_dir, img_filename)
        shutil.copy(src, dst)
        val_attributes[img_filename] = attribute_data[img_filename]
    
    # Save train/val attributes
    with open(os.path.join(output_dir, 'train_attributes.json'), 'w') as f:
        json.dump(train_attributes, f, indent=2)
    
    with open(os.path.join(output_dir, 'val_attributes.json'), 'w') as f:
        json.dump(val_attributes, f, indent=2)
    
    print(f"Created split: {len(train_images)} training samples, {len(val_images)} validation samples")

if __name__ == "__main__":
    # Get the first annotation file from the annotation directory
    annotation_dir = Path("P-DESTRE/annotation")
    annotation_files = list(annotation_dir.glob("*.txt"))
    
    if not annotation_files:
        print("Error: No annotation files found in P-DESTRE/annotation directory")
        exit(1)
    
    # Use the specified annotation file
    first_annotation_file = str(annotation_files[75])
    print(f"Using annotation file: {first_annotation_file}")
    
    # Define paths
    images_dir = "jpg_Extracted_PIDS"  # Directory with original images
    output_dir = "attribute_data"  # Directory to save crops and attributes
    
    # Process annotations and images
    process_annotations_and_images(first_annotation_file, images_dir, output_dir) 